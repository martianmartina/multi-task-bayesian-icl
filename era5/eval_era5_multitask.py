#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import pytorch_lightning as pl

from src.data.multi_task_era5 import MultiTaskERA5DataModule
from src.models.multi_task_era5_implicit import MultiTaskERA5ImplicitLearner
from src.models.multi_task_era5_prior_permutation import (
    MultiTaskERA5PriorPermutationInvariantLearner,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained ERA5 multitask checkpoint on the test split."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to .ckpt file OR run output dir containing checkpoints/ "
            "Previous argument to args.output_dir."
        ),
    )
    p.add_argument("--test_grib_path", type=str, default=None, help="Overrides grib_path from summary.json.")
    p.add_argument("--nic", type=int, default=None, choices=[0, 1, 2], help="Overrides nic from summary.json.")
    p.add_argument("--num_test_samples", type=int, default=2000)
    p.add_argument(
        "--split_strategy",
        type=str,
        default=None,
        choices=["iid", "ood"],
        help="Overrides split_strategy from summary.json. Defaults to 'ood' if unavailable.",
    )
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="Datamodule seed for test sample generation.")
    p.add_argument("--time_start", type=str, default=None, help="Optional inclusive time filter start.")
    p.add_argument("--time_end", type=str, default=None, help="Optional inclusive time filter end.")
    p.add_argument(
        "--stats_grib_path",
        type=str,
        default=None,
        help=(
            "GRIB used for standardization stats. Defaults to the checkpoint "
            "summary grib_path, even when --test_grib_path overrides sample data."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mt_icl", "prior_permutation"],
        help="Overrides model type from summary.json.",
    )

    # Trainer options
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--strategy", type=str, default="auto")
    p.add_argument("--precision", type=str, default="16-mixed")

    p.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Path to save evaluation json. Default: <run_dir>/test_eval_summary.json",
    )
    return p.parse_args()


def _load_summary_if_exists(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        return json.load(f)


def _resolve_checkpoint_path(checkpoint_arg: str) -> tuple[Path, Optional[Path]]:
    path = Path(checkpoint_arg).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    ckpt_dir = path / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in: {path}")

    exact_best = ckpt_dir / "best.ckpt"
    if exact_best.exists():
        return exact_best, path

    raise FileNotFoundError(f"No best.ckpt files found in: {ckpt_dir}")


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    ckpt_path, run_dir = _resolve_checkpoint_path(args.checkpoint)
    summary = _load_summary_if_exists(run_dir) if run_dir is not None else {}

    test_grib_path = args.test_grib_path or summary.get("grib_path")
    stats_grib_path = args.stats_grib_path or summary.get("grib_path") or test_grib_path
    nic = args.nic if args.nic is not None else summary.get("nic")
    split_strategy = args.split_strategy or summary.get("split_strategy", "ood")
    model_name = args.model or summary.get("model", "mt_icl")

    if test_grib_path is None:
        raise ValueError("test_grib_path is required. Pass --test_grib_path or provide a run dir with summary.json.")
    if nic is None:
        raise ValueError("nic is required. Pass --nic or provide a run dir with summary.json.")

    dm = MultiTaskERA5DataModule(
        grib_path=test_grib_path,
        num_in_context_datasets=int(nic),
        num_train_samples=1,
        num_val_samples=1,
        num_test_samples=args.num_test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        add_task_ids=True,
        split_strategy=split_strategy,
        seed=args.seed,
        time_start=args.time_start,
        time_end=args.time_end,
        stats_grib_path=stats_grib_path,
    )
    dm.setup("test")

    strategy = args.strategy
    if args.devices == 1 and strategy == "ddp":
        strategy = "auto"

    model_cls = (
        MultiTaskERA5PriorPermutationInvariantLearner
        if model_name == "prior_permutation"
        else MultiTaskERA5ImplicitLearner
    )
    model = model_cls.load_from_checkpoint(str(ckpt_path))
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        deterministic=False,
        enable_progress_bar=True,
    )

    metrics = trainer.test(model, datamodule=dm, verbose=False)[0]

    result = {
        "checkpoint": str(ckpt_path),
        "grib_path": test_grib_path,
        "stats_grib_path": stats_grib_path,
        "nic": int(nic),
        "num_test_samples": args.num_test_samples,
        "split_strategy": split_strategy,
        "model": model_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "time_start": args.time_start,
        "time_end": args.time_end,
        "metrics": metrics,
    }

    if args.eval_output is not None:
        out_path = Path(args.eval_output).expanduser().resolve()
    elif run_dir is not None:
        out_path = run_dir / "test_eval_summary.json"
    else:
        out_path = Path.cwd() / "test_eval_summary.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Test Evaluation ===")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()