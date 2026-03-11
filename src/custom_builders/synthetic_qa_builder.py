"""Build PRISM projections from synthetic contrastive QA pairs."""
from src.model import ProjectionBuilderBase

import json
import random
import argparse

# default random seed (can be overridden via --seed)
random.seed(42)


class SynthQABuilder(ProjectionBuilderBase):
    """Build differential projections from synthetic contrastive QA data."""

    def __init__(self, seed=42, **kwargs):
        """Initialise builder with a fixed random seed for reproducibility."""
        super().__init__(**kwargs)
        self.seed = seed
        random.seed(seed)

    def iter_examples(self):
        """Yield shuffled examples up to max_samples."""
        all_examples = []
        with open(self.data_path) as f:
            for line in f:
                all_examples.append(json.loads(line))

        rng = random.Random(self.seed)
        rng.shuffle(all_examples)

        for i, ex in enumerate(all_examples):
            if i >= self.max_samples:
                break
            yield ex

    def get_triplets(self, ex: dict) -> list[tuple[str, str, str, str]]:
        """Return (context, relevant_q, answer, irrelevant_q) tuples."""
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'], ex['question_2']),
            (ex['context_2'], ex['question_2'], ex['answer_2'], ex['question_1'])
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build projections from synthetic QA data")
    parser.add_argument('--model', required=True, help='Model path or HF identifier')
    parser.add_argument('--data', required=True, help='Path to synthetic QA JSON file')
    parser.add_argument('--layers', default='all', help='Layers to use for projection')
    parser.add_argument('--top_pct', type=float, default=0.9,
                        help='Percentage of variance to retain in SVD')
    parser.add_argument('--feature', type=str, default=None,
                        help='Feature function to apply (tanh, elu, squared-exponential)')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to process')
    parser.add_argument('--min_diff', type=float, default=2.0,
                        help='Minimum norm difference threshold for applying projection')
    parser.add_argument('--chat', action='store_true',
                        help='Apply chat template to prompts')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for projections')

    # NEW: Output format options
    parser.add_argument('--save-svd', action='store_true',
                        help='Save SVD components (U matrices and singular values)')
    parser.add_argument('--save-traditional', action='store_true', default=True,
                        help='Save traditional projection matrices (default: True)')
    parser.add_argument('--svd-only', action='store_true',
                        help='Save only SVD components (equivalent to --save-svd --no-save-traditional)')
    parser.add_argument('--save-differential', action='store_true',
                        help='PRISM-K: Save differential cross-covariance projection with norm_diff weights')
    parser.add_argument('--diff-only', action='store_true',
                        help='PRISM-K: Save only differential projection')
    parser.add_argument('--save-kv-differential', action='store_true',
                        help='PRISM-KV: Save Key+Value differential projections')
    parser.add_argument('--kv-diff-only', action='store_true',
                        help='PRISM-KV: Save only KV differential projection')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data shuffling (for seed sensitivity experiments)')

    args = parser.parse_args()

    # Handle svd-only flag
    if args.svd_only:
        args.save_svd = True
        args.save_traditional = False

    # Handle diff-only flag
    if args.diff_only:
        args.save_differential = True
        args.save_traditional = False

    # Handle kv-diff-only flag
    if args.kv_diff_only:
        args.save_kv_differential = True
        args.save_traditional = False

    # Ensure at least one output format is selected
    if not args.save_svd and not args.save_traditional:
        print("Warning: No output format selected. Defaulting to traditional projections.")
        args.save_traditional = True

    # Log the configuration
    print("Building synthetic QA projections with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Layers: {args.layers}")
    print(f"  Feature function: {args.feature}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output formats:")
    print(f"    - Traditional projections: {args.save_traditional}")
    print(f"    - SVD components: {args.save_svd}")
    print(f"    - Differential (PRISM-K): {args.save_differential}")

    builder = SynthQABuilder(
        model_path=args.model,
        data_path=args.data,
        layers=args.layers,
        top_pct=args.top_pct,
        feature=args.feature,
        max_samples=args.max_samples,
        min_diff=args.min_diff,
        chat=args.chat,
        save_svd=args.save_svd,
        save_traditional=args.save_traditional,
        save_differential=getattr(args, 'save_differential', False),
        save_kv_differential=getattr(args, 'save_kv_differential', False),
        seed=args.seed,
    )

    builder.run(args.output_dir)

    print(f"\n🎉 Synthetic QA projection building complete!")
    print(f"   Output directory: {args.output_dir}")

    if args.save_traditional:
        print(f"   Traditional files: *_pos_proj.pt, *_neg_proj.pt")
    if args.save_svd:
        print(f"   SVD files: *_pos_svd.pt, *_neg_svd.pt")
    if args.save_differential:
        print(f"   Differential files: *_diff_proj.pt")
    if args.save_kv_differential:
        print(f"   KV Differential files: *_kv_diff_proj.pt")

    # norm_diff_14b = torch.load(os.path.join(args.output_dir, 'norm_diffs_Qwen3-14B-Base.pt'), weights_only=False)
    # SynthQABuilder.plot_norm_heatmap(norm_diff_14b, "Qwen3-14B-Base", range(40), args.output_dir)
