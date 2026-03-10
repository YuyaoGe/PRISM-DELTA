import os
import argparse
import json
import logging
import datasets
import transformers
from pathlib import Path

from benchmarks.biasbios.preprocess import load_dataset
from benchmarks.biasbios.evaluate import biasbios_prediction_evaluation
from benchmarks.utils.pasta_utils import setup_logger

from src.model import SEKALLM, AdaptiveSEKALLM
from pastalib.pasta import PASTA, read_head_config

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    """Run the evaluation for instruction following tasks."""
    datasets.disable_caching()
    
    # Initialize the model and tokenizer 
    pasta = None
    if args.kv_seka:
        model = SEKALLM(
            args.model,
            pos_pt=args.kv_seka_proj,
            marker_start=args.marker_start,
            marker_end=args.marker_end,
            layers=args.layers,
            torch_dtype="auto",
            device="auto",
            attn_implementation="sdpa",
            kv_seka_pt=args.kv_seka_proj,
            kv_seka_gain_k=args.kv_seka_gain_k,
            kv_seka_gain_v=args.kv_seka_gain_v,
        )
        tokenizer = model.tok
        if not args.add_marker:
            logger.warning("KV-SEKA requires markers, setting add_marker to True.")
            args.add_marker = True
    elif args.wd_seka:
        model = SEKALLM(
            args.model,
            pos_pt=args.wd_seka_proj,
            marker_start=args.marker_start,
            marker_end=args.marker_end,
            layers=args.layers,
            torch_dtype="auto",
            device="auto",
            attn_implementation="sdpa",
            wd_seka_pt=args.wd_seka_proj,
            wd_seka_gain=args.wd_seka_gain,
            wd_seka_uniform_weight=args.uniform_weight,
        )
        tokenizer = model.tok
        if not args.add_marker:
            logger.warning("WD-SEKA requires markers, setting add_marker to True.")
            args.add_marker = True
    elif args.seka:
        if "_tanh" in args.pos:
            feature_fn = "tanh"
        elif "_elu" in args.pos:
            feature_fn = "elu"
        elif "_squared" in args.pos:
            feature_fn = "squared-exponential"
        else:
            feature_fn = None

        model = SEKALLM(
            args.model,
            pos_pt=args.pos,
            neg_pt=args.neg,
            marker_start=args.marker_start,
            marker_end=args.marker_end,
            layers=args.layers,
            amplify_pos=args.amplify_pos,
            amplify_neg=args.amplify_neg,
            feature_function=feature_fn,
            torch_dtype="auto",
            device="auto",
            attn_implementation="sdpa",
        )
        tokenizer = model.tok

        # Force add_marker flag to be True
        if not args.add_marker:
            logger.warning("SEKA LLM requires markers, setting add_marker to True.")
            args.add_marker = True
    elif args.adaptive_seka:
        if args.adaptive_expert_path is None:
            raise ValueError("Adaptive SEKA requires an adaptive expert path.")

        expert_path = json.load(open(args.adaptive_expert_path, "r"))
        
        model = AdaptiveSEKALLM(
            args.model,
            expert_paths=expert_path,
            marker_start=args.marker_start,
            marker_end=args.marker_end,
            layers=args.layers,
            top_k_singular=args.top_k_singular,
            combination_method=args.combination_method,
            amplify_factor=args.adaptive_amplify_factor,
            device="auto",
        )
            
        tokenizer = model.tok
        
        # Force add_marker flag to be True
        if not args.add_marker:
            logger.warning("Adaptive SEKA LLM requires markers, setting add_marker to True.")
            args.add_marker = True
    elif args.anchor:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            padding_side="left"
        )
        args.add_marker = True
        args.marker_start = "<anchor>"
        args.marker_end = "</anchor>"
        logger.info(f"Anchor steering requires markers, setting markers to {args.marker_start} and {args.marker_end}.")
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if args.pasta:
            head_config = read_head_config(args.head_config)
            pasta = PASTA(
                model, 
                tokenizer,
                head_config=head_config, 
                alpha=args.pasta_alpha, 
                scale_position=args.scale_position,
            )
    
    # Set up the evaluation data 
    dataset = load_dataset(args.data_path, args.attribute_no_entity, args.example_subset)

    result_output_dir = Path(args.output_dir)
    result_output_dir.mkdir(exist_ok=True, parents=True) 
    result_output_file = result_output_dir / "result.json"
    
    if not os.path.exists(result_output_file) or args.overwrite_output_dir:
        logger.info("begin evaluation")
        
        if args.add_marker and args.marker_end is None:
            args.marker_end = args.marker_start
        
        results = biasbios_prediction_evaluation(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            desc="BiasBios Evaluation",
            add_marker=args.add_marker,
            marker_start=args.marker_start,
            marker_end=args.marker_end,
            seka=args.seka or args.adaptive_seka or args.wd_seka or args.kv_seka,
            pasta=pasta,
            anchor=args.anchor,
            anchor_strength=args.anchor_strength,
        )
        logging.info(
            f"Evaluation complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )
        # Readout the results
        with result_output_file.open("w") as f:
            json.dump(results.to_dict(), f, indent=4)
    else:
        logger.info(
            f"existing results found at {result_output_file}; skipping"
        )

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(
        description="Evaluation model generation on BiasBios dataset."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="model name or path",
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to the dataset"
    )
    parser.add_argument(
        "--attribute-no-entity",
        action="store_true",
        default=False,
        help="set context = attribute",
    )
    parser.add_argument(
        "--example_subset", type=str, default=None, help="run on a subset of data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="unique name for the experiment",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true", help="")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max generation length.")

    parser.add_argument("--chat", action="store_true", default=False, help="Apply chat template")
    parser.add_argument("--add_marker", action="store_true", default=False, help="Apply marked prompting")
    parser.add_argument('--marker_start', default='**',
                help='highlight start marker (e.g. 👉 )')
    parser.add_argument('--marker_end', default=None,
                help='highlight end marker; defaults to same as start')

    parser.add_argument("--seka", action="store_true", default=False, help="Use SEKA model")
    parser.add_argument('--pos', type=str, default=None,
                    help='positive (relevant) projector .pt')
    parser.add_argument('--neg', type=str, default=None,
                help='optional negative (irrelevant) projector .pt')
    parser.add_argument('--amplify_pos', default=1.5, type=float)
    parser.add_argument('--amplify_neg', default=0.5, type=float)
    parser.add_argument('--layers', default='last10',
                help="'all' / 'last4' / '0,4,19' …")
    
    parser.add_argument("--adaptive-seka", action="store_true", default=False, help="Use adaptive SEKA model")
    parser.add_argument("--adaptive-expert-path", type=str, default=None, help="Path to adaptive expert file")
    parser.add_argument("--adaptive_amplify_factor", type=float, default=1.0, help="Amplification factor for adaptive SEKA")
    parser.add_argument("--top_k_singular", type=int, default=5, help="Top k singular values for adaptive SEKA")
    parser.add_argument("--combination_method", type=str, default="weighted_top_k", choices=["weighted_top_k", "all_weighted", "top_k_uniform"], help="Combination method for adaptive SEKA")
    
    parser.add_argument("--pasta", action="store_true", default=False, help="Use PASTA model")
    parser.add_argument("--head_config", type=str, default=None, help="PASTA head config for steering")
    parser.add_argument("--pasta_alpha", type=float, default=None, help="Scaling coefficient")
    parser.add_argument("--scale_position", type=str, default=None, help="Steer the selected section or others")

    parser.add_argument("--anchor", action="store_true", default=False, help="Use anchor steering")
    parser.add_argument("--anchor_strength", type=float, default=1.6, help="Anchor strength for steering")

    parser.add_argument("--wd-seka", action="store_true", default=False, help="Use WD-SEKA (weighted differential)")
    parser.add_argument("--wd-seka-proj", type=str, default=None, help="Path to WD-SEKA differential projection .pt")
    parser.add_argument("--wd-seka-gain", type=float, default=1.0, help="WD-SEKA steering gain")
    parser.add_argument("--uniform-weight", action="store_true", default=False, help="Use uniform weight w=1 instead of softplus (for ablation)")

    parser.add_argument("--kv-seka", action="store_true", default=False, help="Use KV-SEKA (Key+Value joint steering)")
    parser.add_argument("--kv-seka-proj", type=str, default=None, help="Path to KV-SEKA projection .pt")
    parser.add_argument("--kv-seka-gain-k", type=float, default=0.4, help="KV-SEKA Key steering gain")
    parser.add_argument("--kv-seka-gain-v", type=float, default=0.2, help="KV-SEKA Value steering gain")

    args = parser.parse_args()

    assert sum([args.seka, args.adaptive_seka, args.wd_seka, args.kv_seka]) <= 1, "Cannot use multiple SEKA variants at the same time."

    main(args)
