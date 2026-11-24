"""Run comparable baseline and DCA pretraining experiments locally.

Examples:
  python -m scripts.run_dca_experiment --depth=12 --num-iterations=100
  python -m scripts.run_dca_experiment --depth=12 --dca-k 0 2 4 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline/DCA pretraining comparisons locally")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--dca-k", type=int, nargs="+", default=[0, 4],
                        help="DCA stack sizes; 0 is the baseline")
    parser.add_argument("--grn-lrs", type=float, nargs="+", default=[0.005],
                        help="one GRN learning rate to use for each DCA k")
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--target-param-data-ratio", type=float, default=12)
    parser.add_argument("--device-batch-size", type=int, default=-1)
    parser.add_argument("--total-batch-size", type=int, default=-1)
    parser.add_argument("--window-pattern", type=str, default="SSSL")
    parser.add_argument("--run-prefix", type=str, default="dca-exp")
    parser.add_argument("--model-tag-prefix", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without starting training")
    return parser.parse_args()


def positive_k_values(args: argparse.Namespace) -> list[int]:
    values = sorted(set(args.dca_k))
    if any(k < 0 for k in values):
        raise SystemExit("--dca-k values must be non-negative")
    if len(args.grn_lrs) not in (1, len(values)):
        raise SystemExit("--grn-lrs must contain one value or one value per --dca-k value")
    return values


def grn_lr_for(args: argparse.Namespace, k_values: list[int], k: int) -> float:
    if len(args.grn_lrs) == 1:
        return args.grn_lrs[0]
    return args.grn_lrs[k_values.index(k)]


def common_training_args(args: argparse.Namespace, k: int, grn_lr: float) -> list[str]:
    result = [
        f"--depth={args.depth}",
        f"--dca-k={k}",
        f"--grn-lr={grn_lr}",
        f"--window-pattern={args.window_pattern}",
        f"--target-param-data-ratio={args.target_param_data_ratio}",
        f"--num-iterations={args.num_iterations}",
    ]
    if args.device_batch_size != -1:
        result.append(f"--device-batch-size={args.device_batch_size}")
    if args.total_batch_size != -1:
        result.append(f"--total-batch-size={args.total_batch_size}")
    return result


def experiment_name(args: argparse.Namespace, k: int) -> str:
    kind = "baseline" if k == 0 else f"dca{k}"
    return f"{args.run_prefix}-d{args.depth}-{kind}"


def model_tag(args: argparse.Namespace, k: int) -> str:
    prefix = args.model_tag_prefix or f"{args.run_prefix}-d{args.depth}"
    kind = "baseline" if k == 0 else f"dca{k}"
    return f"{prefix}-{kind}"


def print_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)


def run_local(args: argparse.Namespace, k_values: list[int]) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    for k in k_values:
        grn_lr = grn_lr_for(args, k_values, k)
        name = experiment_name(args, k)
        tag = model_tag(args, k)
        command = [
            sys.executable,
            "-m",
            "scripts.base_train",
        ] + common_training_args(args, k, grn_lr) + [
            f"--run={name}",
            f"--model-tag={tag}",
        ]
        print_command(command)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=root, env=env, check=True)


def main() -> None:
    args = parse_args()
    k_values = positive_k_values(args)
    run_local(args, k_values)


if __name__ == "__main__":
    main()
