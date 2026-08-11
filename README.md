# [COLM 2026] PRISM-Δ: Differential Subspace Steering for Prompt Highlighting in Large Language Models

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2603.10705&color=red&logo=arxiv)](https://arxiv.org/abs/2603.10705)
[![Paper (PDF)](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/pdf/2603.10705)
[![Hugging Face Papers](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2603.10705)
[![Projections](https://img.shields.io/badge/📦%20Projections-Prism__Delta-orange)](https://huggingface.co/YuyaoGe/Prism_Delta)
[![Stars](https://img.shields.io/github/stars/YuyaoGe/PRISM-DELTA?style=social)]()

**Official implementation of the COLM 2026 paper "Prism-Δ: Differential Subspace Steering for Prompt Highlighting in Large Language Models"** by *Yuyao Ge, Shenghua Liu, Yiwei Wang, Baolong Bi, Lingrui Mei, Jiayu Yao, Tianyu Liu, Jiafeng Guo,* and *Xueqi Cheng*.

PRISM-Δ is a **training-free** method that makes a language model prioritize the spans a user has highlighted. It decomposes the difference between positive and negative cross-covariance matrices so that structure shared by relevant and irrelevant contexts cancels out, weights every attention head by a continuous softplus score instead of a hard threshold, and edits Key — optionally also Value — vectors in place. No fine-tuning, no extra forward passes, compatible with FlashAttention.

## 📣 News
- **[Aug 2026]** Precomputed projections released on [Hugging Face](https://huggingface.co/YuyaoGe/Prism_Delta).
- **[Jul 2026]** Our paper has been accepted to **COLM 2026**! 🎉
- **[Mar 2026]** Paper and code released ([arXiv:2603.10705](https://arxiv.org/abs/2603.10705)).

## Contents
- [Key ideas](#key-ideas)
- [Repo layout](#repo-layout)
- [Quick start](#quick-start)
  - [1. Install](#1-install)
  - [2. Get models, data and projections](#2-get-models-data-and-projections)
  - [3. Run](#3-run)
- [Other benchmarks](#other-benchmarks)
- [Recommended hyperparameters](#recommended-hyperparameters)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Key ideas

<p align="center">
  <img src="assets/overview.png" alt="PRISM-Delta overview" width="90%">
</p>

- **Differential cross-covariance.** Existing Key-editing methods read a steering direction off relevant contexts alone, so the direction carries whatever structure relevant and irrelevant contexts have in common. Taking the SVD of the *difference* $\Omega_\Delta = \Omega^+ - \Omega^-$ cancels that shared part and leaves only what discriminates.
- **Softplus head weighting.** Each head gets a continuous importance weight from its discriminability score, so a weak-but-useful head contributes at reduced strength rather than being switched off by a threshold.
- **Dual-channel steering.** The same construction extends from the Key (routing) channel to the Value (content) channel, giving PRISM-ΔV, which recovers fluency that Key-only steering costs.

## Repo layout
```
src/
  model/
    prism_llm.py                    # inference-time steering (Key / Key+Value)
    adaptive_prism_llm.py           # query-adaptive multi-expert variant
    projection_builder_base.py      # cross-covariance extraction + differential SVD
  custom_builders/
    synthetic_qa_builder.py         # build projections from contrastive triplets
benchmarks/
  eval_bias_gen.py                  # BiasBios
  eval_fact_gen.py                  # CounterFact
  eval_biasbios_instruction.py      # Pronoun Change
pastalib/                           # PASTA baseline + head profiling configs
data/synthetic/pair_qa_new.jsonl    # contrastive data for projection building
```

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"
```

### 2. Get models, data and projections

**Models** — any of [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base), [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base), [Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base), [gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt), [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt).

**Data** — [SEKA-datasets](https://huggingface.co/datasets/waylonli/SEKA-datasets) covers BiasBios, CounterFact (`pasta_bench`) and Pronoun Change. Extract it under a local directory such as `./datasets/`.

**Projections** — download the precomputed ones and skip building:
```bash
git clone https://huggingface.co/YuyaoGe/Prism_Delta ./projections
```
or build them yourself (3–8 minutes per model on one GPU):
```bash
# PRISM-Δ (Key-only)
python src/custom_builders/synthetic_qa_builder.py \
  --model <path-to-model> --data data/synthetic/pair_qa_new.jsonl \
  --output_dir ./projections/biasbios/prism \
  --max_samples 200 --min_diff 0.08 --top_pct 0.998 --diff-only

# PRISM-ΔV (Key + Value)
python src/custom_builders/synthetic_qa_builder.py \
  --model <path-to-model> --data data/synthetic/pair_qa_new.jsonl \
  --output_dir ./projections/biasbios/prism-kv \
  --max_samples 200 --min_diff 0.08 --top_pct 0.998 --kv-diff-only
```

### 3. Run
```bash
export PYTHONPATH=$(pwd)

# Vanilla (no steering)
python benchmarks/eval_bias_gen.py \
  --model <path-to-model> --data_path <path-to-biasbios.json> \
  --output_dir ./results/vanilla \
  --overwrite_output_dir --batch_size 256 --max_new_tokens 64

# PRISM-Δ
python benchmarks/eval_bias_gen.py \
  --model <path-to-model> --data_path <path-to-biasbios.json> \
  --output_dir ./results/prism-k \
  --overwrite_output_dir --batch_size 256 --max_new_tokens 64 \
  --wd-seka --wd-seka-proj <path-to-diff_proj.pt> \
  --wd-seka-gain 0.40 --layers all

# PRISM-ΔV
python benchmarks/eval_bias_gen.py \
  --model <path-to-model> --data_path <path-to-biasbios.json> \
  --output_dir ./results/prism-kv \
  --overwrite_output_dir --batch_size 256 --max_new_tokens 64 \
  --kv-seka --kv-seka-proj <path-to-kv_diff_proj.pt> \
  --kv-seka-gain-k 0.40 --kv-seka-gain-v 0.10 --layers all
```

## Other benchmarks

**CounterFact** (test split `5000:10000`):
```bash
python benchmarks/eval_fact_gen.py \
  --model <path-to-model> --data_path <path-to-pasta_bench> \
  --output_dir ./results/counterfact/prism-k \
  --overwrite_output_dir --batch_size 128 --max_new_tokens 32 \
  --example_subset 5000:10000 \
  --wd-seka --wd-seka-proj <path-to-diff_proj.pt> \
  --wd-seka-gain 2.50 --layers all
```

**Pronoun Change** (test split `5000:10000`):
```bash
python benchmarks/eval_biasbios_instruction.py \
  --model <path-to-model> --data_path <path-to-biasbios.json> \
  --output_dir ./results/pronoun/prism-k \
  --overwrite_output_dir --batch_size 64 --max_new_tokens 128 \
  --task pronchange --example_subset 5000:10000 \
  --wd-seka --wd-seka-proj <path-to-diff_proj.pt> \
  --wd-seka-gain 0.05 --layers all
```

## Recommended hyperparameters

`gamma` and `delta_min` are build-time settings; `g_K` is applied at inference. Per-benchmark values for every configuration are listed on the [projections page](https://huggingface.co/YuyaoGe/Prism_Delta) and in the paper's appendix.

| Model | gamma | delta_min | g_K | Batch Size |
|-------|-------|-----------|-----|------------|
| [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | 0.998 | 0.08 | 0.40 | 256 |
| [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | 0.998 | 0.08 | 0.40 | 128 |
| [Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base) | 0.998 | 0.08 | 0.40 | 64 |
| [gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt) | 0.850 | 0.08 | 0.50 | 256 |
| [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt) | 0.990 | 0.04 | 0.40 | 64 |

## Citation
```bibtex
@inproceedings{ge2026prism,
  title={Prism-$\Delta$: Differential Subspace Steering for Prompt Highlighting in Large Language Models},
  author={Ge, Yuyao and Liu, Shenghua and Wang, Yiwei and Bi, Baolong and Mei, Lingrui and Yao, Jiayu and Liu, Tianyu and Guo, Jiafeng and Cheng, Xueqi},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

## Acknowledgements
This work builds on [**SEKA**](https://github.com/waylonli/SEKA), [**SEA-LLM**](https://github.com/yfqiu-nlp/sea-llm), [**PASTA**](https://github.com/QingruZhang/PASTA), and [**Selective Prompt Anchoring**](https://github.com/magic-YuanTian/Selective-Prompt-Anchoring).
