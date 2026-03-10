from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from src.utils import encode_with_markers, _parse_layers, _load_proj, phi, phi_inv

class SEKALLM:
    # ───────────── init ────────────────────────────────────────────────
    def __init__(self,
                 model_or_path: str,
                 *,
                 device: str | None = "auto",
                 marker_start: str = "**",
                 marker_end: str | None = None,
                 pos_pt: str = None,
                 neg_pt: str | None = None,
                 layers: str = "last10",
                 amplify_pos: float = 0.8,
                 amplify_neg: float = 0.2,
                 feature_function: str | None = None,
                 **hf_kwargs
                 ):
        # ----- extract custom params before passing to HF -----
        self.wd_seka_pt = hf_kwargs.pop('wd_seka_pt', None)
        self.wd_seka_gain = hf_kwargs.pop('wd_seka_gain', 1.0)
        self.wd_seka_uniform_weight = hf_kwargs.pop('wd_seka_uniform_weight', False)
        self.kv_seka_pt = hf_kwargs.pop('kv_seka_pt', None)
        self.kv_seka_gain_k = hf_kwargs.pop('kv_seka_gain_k', 0.4)
        self.kv_seka_gain_v = hf_kwargs.pop('kv_seka_gain_v', 0.2)

        # ----- device selection -----
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps"  if torch.backends.mps.is_available()
                      else "cpu")

        multi_gpu = torch.cuda.device_count() > 1 and str(device).startswith("cuda")

        # ----- HF objects -----
        self.name_or_path = f"SEKA-{model_or_path}"
        self.tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_or_path, padding_side="left", **hf_kwargs)

        if multi_gpu:
            # shard across all GPUs
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                device_map="auto",
                # use_cache=False,
                **hf_kwargs
            ).eval()
        else:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                # use_cache=False,
                **hf_kwargs
            ).to(device).eval()

        # markers & steering defaults
        self.m_start, self.m_end = marker_start, (marker_start if marker_end is None else marker_end)
        self.pos_pt, self.neg_pt = pos_pt,   neg_pt
        self.layers = layers
        self.amplify_pos, self.amplify_neg = amplify_pos, amplify_neg

        if feature_function is None:
            if "_tanh"     in str(pos_pt): self.feature_function = "tanh"
            elif "_elu"    in str(pos_pt): self.feature_function = "elu"
            elif "_squared" in str(pos_pt): self.feature_function = "squared-exponential"
            else:                          self.feature_function = None
        else:
            self.feature_function = feature_function

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        # expose everything from the HF model
        object.__setattr__(self, "__getattr__", lambda n: getattr(self.model, n))

    @property
    def device(self):  # convenience
        return self.model.device

    def generate(self,
                 ids: torch.LongTensor | str,
                 steer: bool = True,
                 steer_mask: torch.Tensor | None = None,
                 attention_mask: torch.Tensor | None = None,
                 return_raw: bool = False,
                 **gen_kw) -> str:

        if isinstance(ids, (str, list)):
            ids, steer_mask, attention_mask = encode_with_markers(ids, self.tok, self.m_start, self.m_end)
            ids = ids.to(self.device)
            # check decoded ids
            steer_mask = steer_mask.to(self.device)
            attention_mask = attention_mask.to(self.device)
        elif isinstance(ids, torch.Tensor):
            if steer:
                assert steer_mask is not None, "steer_mask must be provided if ids is a tensor"
                steer_mask = steer_mask.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
            attention_mask = attention_mask.to(self.device)

        # -------- optional steering --------
        if steer and self.kv_seka_pt:
            self.attach_kv_seka(steer_mask_tensor=steer_mask, silence=True)
        elif steer and self.wd_seka_pt:
            self.attach_wd_seka(steer_mask_tensor=steer_mask, silence=True)
        elif steer:
            self.attach_projection(steer_mask_tensor=steer_mask, silence=True)
        else:
            self.remove_projection()

        if "attention_mask" not in gen_kw and attention_mask is not None:
            gen_kw["attention_mask"] = attention_mask

        out = self.model.generate(
            ids,
            **gen_kw
        )

        if steer:
            self.remove_projection()

        if return_raw:
            return out
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(ids, out)
        ]
        generated = self.tok.batch_decode(generated_ids, skip_special_tokens=True)

        return generated[0] if len(generated) == 1 else generated

    # ───────────── steering ────────────────────────────────────────────
    def attach_projection(
        self,
        steer_mask_tensor=None,
        pos_pt=None,
        neg_pt=None,
        layers=None,
        amplify_pos=None,
        amplify_neg=None,
        feature_function=None,
        silence=False
    ):
        self.remove_projection()

        # defaults
        pos_pt         = self.pos_pt         if pos_pt  is None else pos_pt
        neg_pt         = self.neg_pt         if neg_pt  is None else neg_pt
        layers         = self.layers         if layers  is None else layers
        amplify_pos    = self.amplify_pos    if amplify_pos is None else amplify_pos
        amplify_neg    = self.amplify_neg    if amplify_neg is None else amplify_neg
        feature_function = self.feature_function if feature_function is None else feature_function

        n_layers = len(self.model.model.layers) if "gemma3" not in self.model.__class__.__name__.lower() else len(self.model.language_model.model.layers)
        dtype = torch.float32

        # load projections on first device (cheap to move later)
        first_dev = self.device
        file_layers, P_pos_stack = _load_proj(pos_pt, first_dev)  # → (L_sel,H,d,d)
        sel_layers = _parse_layers(layers, n_layers)
        if P_pos_stack.ndim != 4:
            raise ValueError("Expected 4‑D pos_proj")
        P_pos = {layer: P_pos_stack[i].to(dtype) for i, layer in enumerate(sel_layers)}

        P_neg = None
        if neg_pt:
            _, P_neg_stack = _load_proj(neg_pt, first_dev)
            if P_neg_stack.ndim != 4:
                raise ValueError("Expected 4‑D neg_proj")
            P_neg = {layer: P_neg_stack[i].to(dtype) for i, layer in enumerate(sel_layers)}

        # steering mask (will still be relocated inside hook)
        m_dev = (steer_mask_tensor if steer_mask_tensor is None
                 else steer_mask_tensor.unsqueeze(0) if steer_mask_tensor.dim() == 1
                 else steer_mask_tensor).to(first_dev) if steer_mask_tensor is not None else None
        

        # pick correct root (handles DataParallel etc.)
        root = self.model.module if hasattr(self.model, "module") else self.model

        # register hooks
        for L in sel_layers:
            attn = root.model.layers[L].self_attn if "gemma3" not in root.__class__.__name__.lower() else root.language_model.model.layers[L].self_attn
            mod  = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj

            # Move tensors to layer's device
            layer_device = next(mod.parameters()).device
            m_dev = m_dev.to(layer_device) if m_dev is not None else None
            Pp_layer = P_pos[L].to(layer_device) # (H, D, D)
            Pn_layer = P_neg[L].to(layer_device) if P_neg else None

            def _hook(_, __, k_in,
                      m=m_dev, Pp=Pp_layer, Pn=Pn_layer,
                      gp=amplify_pos, gn=amplify_neg
                      ):
                
                # k_in can be [B,T,H,D] or [B,T,D]
                need_transpose = False

                if k_in.dim() == 4:
                    B, T, H, D = k_in.shape
                    k_view = k_in
                elif k_in.dim() == 3:
                    B, T, D_all = k_in.shape
                    H = Pp.shape[0]
                    D = Pp.shape[1]
                    assert D_all == H * D, f"Dim mismatch: {D_all} != {H}*{D}"
                    k_view = k_in.view(B, T, H, D)
                else:
                    raise ValueError(f"Unsupported k_in shape: {k_in.shape}")

                if (Pp.sum() == 0 or m is None or m.sum() == 0):
                    return k_in
                

                if m.shape != (B, T):
                    if m.shape == (B, H):
                        # transpose k_in to [B, T, H, D]
                        k_view = k_view.transpose(1, 2)  # (B, H, T, D)
                        need_transpose = True
                    else:
                        return k_in  # no steering if mask shape is unexpected

                k_feat = phi(k_view, feature_function)  # (B, T, H, D)
                k_sel = k_feat[m].to(Pp.dtype)  # (N_sel, H, D)
                pos = torch.einsum('n h d, h d k -> n h k', k_sel, Pp)

                if Pn is not None:
                    neg = torch.einsum('n h d, h d k -> n h k', k_sel, Pn)
                    delta = (gp * pos + gn * neg) / 2
                else:
                    delta = gp * pos

                k_feat[m] += delta.to(k_feat.dtype)
                k_out = phi_inv(k_feat, feature_function)

                if need_transpose:
                    k_out = k_out.transpose(1, 2)

                # Return in original shape
                if k_in.dim() == 4:
                    return k_out
                else:
                    return k_out.contiguous().view(B, T, H * D)
                

            self._hooks.append(mod.register_forward_hook(_hook))

        if not silence:
            print(f"✅ Steering hooks attached on layers {sel_layers}")

    def attach_wd_seka(self, steer_mask_tensor=None, silence=False):
        """WD-SEKA: Attach differential projection with soft head weighting."""
        self.remove_projection()

        wd_pt = self.wd_seka_pt
        gain = self.wd_seka_gain
        layers_spec = self.layers
        feature_function = self.feature_function

        n_layers = len(self.model.model.layers) if "gemma3" not in self.model.__class__.__name__.lower() else len(self.model.language_model.model.layers)
        first_dev = self.device

        # Load differential projection file
        obj = torch.load(wd_pt, map_location=first_dev)
        P_diff_stack = obj['proj'].to(torch.float32)        # (L_sel, H, d, d)
        sel_layers = _parse_layers(layers_spec, n_layers)

        # Precompute soft weights: softplus(norm_diff - min_diff)
        # Falls back to uniform weights (1.0) if norm_diffs not in file or uniform_weight is set
        if self.wd_seka_uniform_weight:
            soft_weights = torch.ones(P_diff_stack.shape[0], P_diff_stack.shape[1])
        elif 'norm_diffs' in obj:
            norm_diffs = obj['norm_diffs'].to(torch.float32)
            min_diff = obj.get('min_diff', 0.0)
            soft_weights = torch.nn.functional.softplus(norm_diffs - min_diff)
        else:
            soft_weights = torch.ones(P_diff_stack.shape[0], P_diff_stack.shape[1])

        # steering mask
        m_dev = (steer_mask_tensor if steer_mask_tensor is None
                 else steer_mask_tensor.unsqueeze(0) if steer_mask_tensor.dim() == 1
                 else steer_mask_tensor).to(first_dev) if steer_mask_tensor is not None else None

        root = self.model.module if hasattr(self.model, "module") else self.model

        for i, L in enumerate(sel_layers):
            attn = root.model.layers[L].self_attn if "gemma3" not in root.__class__.__name__.lower() else root.language_model.model.layers[L].self_attn
            mod = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj

            layer_device = next(mod.parameters()).device
            m_local = m_dev.to(layer_device) if m_dev is not None else None
            Pd_layer = P_diff_stack[i].to(layer_device)     # (H, d, d)
            w_layer = soft_weights[i].to(layer_device)      # (H,)

            def _hook(_, __, k_in,
                      m=m_local, Pd=Pd_layer, w=w_layer, g=gain):
                need_transpose = False
                if k_in.dim() == 4:
                    B, T, H, D = k_in.shape
                    k_view = k_in
                elif k_in.dim() == 3:
                    B, T, D_all = k_in.shape
                    H = Pd.shape[0]
                    D = Pd.shape[1]
                    k_view = k_in.view(B, T, H, D)
                else:
                    return k_in

                if m is None or m.sum() == 0:
                    return k_in

                if m.shape != (B, T):
                    if m.shape == (B, H):
                        k_view = k_view.transpose(1, 2)
                        need_transpose = True
                    else:
                        return k_in

                k_feat = phi(k_view, feature_function)
                k_sel = k_feat[m].to(Pd.dtype)              # (N_sel, H, D)
                proj = torch.einsum('n h d, h d k -> n h k', k_sel, Pd)
                # Apply soft per-head weight: w shape (H,) -> broadcast over (N_sel, H, D)
                delta = g * w.unsqueeze(0).unsqueeze(-1) * proj
                k_feat[m] += delta.to(k_feat.dtype)
                k_out = phi_inv(k_feat, feature_function)

                if need_transpose:
                    k_out = k_out.transpose(1, 2)
                if k_in.dim() == 4:
                    return k_out
                else:
                    return k_out.contiguous().view(B, T, H * D)

            self._hooks.append(mod.register_forward_hook(_hook))

        if not silence:
            print(f"WD-SEKA hooks attached on layers {sel_layers}")

    def attach_kv_seka(self, steer_mask_tensor=None, silence=False):
        """KV-SEKA: Attach differential projection on both Key and Value."""
        self.remove_projection()

        kv_pt = self.kv_seka_pt
        gk = self.kv_seka_gain_k
        gv = self.kv_seka_gain_v
        layers_spec = self.layers
        feature_function = self.feature_function

        n_layers = len(self.model.model.layers) if "gemma3" not in self.model.__class__.__name__.lower() else len(self.model.language_model.model.layers)
        first_dev = self.device

        obj = torch.load(kv_pt, map_location=first_dev)
        Pk_stack = obj['k_proj'].to(torch.float32)
        Pv_stack = obj['v_proj'].to(torch.float32)
        k_nds = obj['k_norm_diffs'].to(torch.float32)
        v_nds = obj['v_norm_diffs'].to(torch.float32)
        min_diff = obj.get('min_diff', 0.0)
        sel_layers = _parse_layers(layers_spec, n_layers)

        k_weights = torch.nn.functional.softplus(k_nds - min_diff)
        v_weights = torch.nn.functional.softplus(v_nds - min_diff)

        m_dev = (steer_mask_tensor if steer_mask_tensor is None
                 else steer_mask_tensor.unsqueeze(0) if steer_mask_tensor.dim() == 1
                 else steer_mask_tensor).to(first_dev) if steer_mask_tensor is not None else None

        root = self.model.module if hasattr(self.model, "module") else self.model

        for i, L in enumerate(sel_layers):
            attn = root.model.layers[L].self_attn if "gemma3" not in root.__class__.__name__.lower() else root.language_model.model.layers[L].self_attn
            k_mod = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
            v_mod = attn.v_proj

            layer_device = next(k_mod.parameters()).device
            m_local = m_dev.to(layer_device) if m_dev is not None else None
            Pk_l = Pk_stack[i].to(layer_device)
            Pv_l = Pv_stack[i].to(layer_device)
            wk_l = k_weights[i].to(layer_device)
            wv_l = v_weights[i].to(layer_device)

            # --- Key hook ---
            def _k_hook(_, __, k_in, m=m_local, Pk=Pk_l, w=wk_l, g=gk):
                need_transpose = False
                if k_in.dim() == 4:
                    B, T, H, D = k_in.shape
                    k_view = k_in
                elif k_in.dim() == 3:
                    B, T, D_all = k_in.shape
                    H, D = Pk.shape[0], Pk.shape[1]
                    k_view = k_in.view(B, T, H, D)
                else:
                    return k_in
                if m is None or m.sum() == 0:
                    return k_in
                if m.shape != (B, T):
                    if m.shape == (B, H):
                        k_view = k_view.transpose(1, 2)
                        need_transpose = True
                    else:
                        return k_in
                k_feat = phi(k_view, feature_function)
                k_sel = k_feat[m].to(Pk.dtype)
                proj = torch.einsum('n h d, h d k -> n h k', k_sel, Pk)
                delta = g * w.unsqueeze(0).unsqueeze(-1) * proj
                k_feat[m] += delta.to(k_feat.dtype)
                k_out = phi_inv(k_feat, feature_function)
                if need_transpose:
                    k_out = k_out.transpose(1, 2)
                return k_out if k_in.dim() == 4 else k_out.contiguous().view(B, T, H * D)

            self._hooks.append(k_mod.register_forward_hook(_k_hook))

            # --- Value hook ---
            def _v_hook(_, __, v_in, m=m_local, Pv=Pv_l, w=wv_l, g=gv):
                need_transpose = False
                if v_in.dim() == 4:
                    B, T, H, D = v_in.shape
                    v_view = v_in
                elif v_in.dim() == 3:
                    B, T, D_all = v_in.shape
                    H, D = Pv.shape[0], Pv.shape[1]
                    v_view = v_in.view(B, T, H, D)
                else:
                    return v_in
                if m is None or m.sum() == 0:
                    return v_in
                if m.shape != (B, T):
                    if m.shape == (B, H):
                        v_view = v_view.transpose(1, 2)
                        need_transpose = True
                    else:
                        return v_in
                v_sel = v_view[m].to(Pv.dtype)
                proj = torch.einsum('n h d, h d k -> n h k', v_sel, Pv)
                delta = g * w.unsqueeze(0).unsqueeze(-1) * proj
                v_view[m] += delta.to(v_view.dtype)
                if need_transpose:
                    v_view = v_view.transpose(1, 2)
                return v_view if v_in.dim() == 4 else v_view.contiguous().view(B, T, H * D)

            self._hooks.append(v_mod.register_forward_hook(_v_hook))

        if not silence:
            print(f"KV-SEKA hooks attached on layers {sel_layers}")

    def remove_projection(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        pass
