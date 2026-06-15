#!/usr/bin/env python
import os
import json
import math
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.data.multi_task_era5 import MultiTaskERA5DataModule
from src.models.multi_task_era5_implicit import MultiTaskERA5ImplicitLearner
from src.models.multi_task_era5_prior_permutation import (
    MultiTaskERA5PriorPermutationInvariantLearner,
)


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--grib_path", type=str, required=True)
    p.add_argument("--nic", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_train_samples", type=int, default=16000)
    p.add_argument("--num_val_samples", type=int, default=2000)
    p.add_argument("--num_test_samples", type=int, default=2000)
    p.add_argument(
        "--split_strategy",
        type=str,
        default="ood",
        choices=["iid", "ood"],
        help=(
            "'iid': train/val/test sample from all 2019 with different seeds. "
            "'ood': train before 2019-06-17, val 2019-06-17 through 2019-06-30, "
            "test from 2019-07-01 onward."
        ),
    )
    p.add_argument("--seed", type=int, default=0)

    # Training schedule
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    p.add_argument("--iterations_per_epoch", type=int, default=None,
                   help="If set, overrides num_train_samples = batch_size * iterations_per_epoch")
    p.add_argument("--save_predictions", action="store_true")

    # Model
    p.add_argument(
        "--model",
        type=str,
        default="mt_icl",
        choices=["mt_icl", "prior_permutation"],
    )
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--input_noise", type=float, default=0.0,
                   help="Standard deviation of Gaussian noise added to inputs for regularization.")
    p.add_argument("--rope", action="store_true")
    p.add_argument("--no_rope", dest="rope", action="store_false")
    p.set_defaults(rope=True)

    # Trainer / DDP
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--strategy", type=str, default="auto",
                   help="Use 'ddp' for multi-GPU distributed training")
    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])
    p.add_argument("--torch_compile", action="store_true",
                   help="Compile model with torch.compile for better throughput.")
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=20)

    # Logging
    p.add_argument("--logger", type=str, default="csv", choices=["csv", "wandb"])
    p.add_argument("--wandb_project", type=str, default="era5-multitask-icl")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # Output
    p.add_argument("--output_dir", type=str, required=True)

    return p.parse_args()

def build_logger(args, output_dir: Path):
    if args.logger == "csv":
        return CSVLogger(save_dir=str(output_dir), name="logs")

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"era5_nic{args.nic}_seed{args.seed}"

    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        save_dir=str(output_dir),
        mode=args.wandb_mode,
        log_model=False,
    )
    return logger

def build_datamodule(args):
    num_train_samples = args.num_train_samples
    if args.iterations_per_epoch is not None:
        num_train_samples = args.batch_size * args.iterations_per_epoch

    dm = MultiTaskERA5DataModule(
        grib_path=args.grib_path,
        num_in_context_datasets=args.nic,
        num_train_samples=num_train_samples,
        num_val_samples=args.num_val_samples,
        num_test_samples=args.num_test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        add_task_ids=True,
        split_strategy=args.split_strategy,
        seed=args.seed,
    )
    return dm, num_train_samples


def build_model(args, max_steps):
    model_cls = (
        MultiTaskERA5PriorPermutationInvariantLearner
        if args.model == "prior_permutation"
        else MultiTaskERA5ImplicitLearner
    )
    model_kwargs = {
        "x_dim": 4,
        "y_dim": 1,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_steps": max_steps,
        "rope": args.rope,
        "use_seg_ids": True,
        "input_noise": args.input_noise,
    }
    if args.model == "mt_icl":
        model_kwargs["use_task_ids"] = True

    model = model_cls(**model_kwargs)
    return model


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False # a bit slower but better reproducibility?
        torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm, actual_num_train_samples = build_datamodule(args)
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * args.max_epochs

    model = build_model(args, max_steps=max_steps)
    if args.torch_compile:
        model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best",
        monitor="val_nll",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_nll",
        mode="min",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        strict=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = build_logger(args, output_dir)

    # Lightning strategy choice
    strategy = args.strategy
    if args.devices == 1 and strategy == "ddp":
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        benchmark=False,
        enable_progress_bar=True,
    )
    if args.logger == "wandb" and hasattr(logger, "experiment"):
        logger.experiment.config.update(
            {
                "nic": args.nic,
                "seed": args.seed,
                "grib_path": args.grib_path,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "num_train_samples": actual_num_train_samples,
                "num_val_samples": args.num_val_samples,
                "num_test_samples": args.num_test_samples,
                "split_strategy": args.split_strategy,
                "max_epochs": args.max_epochs,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "iterations_per_epoch": args.iterations_per_epoch,
                "model": args.model,
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "rope": args.rope,
                "devices": args.devices,
                "strategy": strategy,
                "precision": args.precision,
            },
            allow_val_change=True,
        )
    trainer.fit(model, datamodule=dm)

    # Load best ckpt if available
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        model_cls = (
            MultiTaskERA5PriorPermutationInvariantLearner
            if args.model == "prior_permutation"
            else MultiTaskERA5ImplicitLearner
        )
        best_model = model_cls.load_from_checkpoint(best_ckpt)
    else:
        best_model = model

    # Evaluate the best validation checkpoint on the held-out test split by default.
    dm.setup("test")
    eval_metrics = trainer.test(best_model, datamodule=dm, verbose=False)[0]
    eval_split = "test"

    summary = {
        "nic": args.nic,
        "seed": args.seed,
        "grib_path": args.grib_path,
        "actual_num_train_samples": actual_num_train_samples,
        "num_val_samples": args.num_val_samples,
        "num_test_samples": args.num_test_samples,
        "split_strategy": args.split_strategy,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "model": args.model,
        "devices": args.devices,
        "strategy": strategy,
        "eval_split": eval_split,
        "best_checkpoint": best_ckpt,
        "metrics": eval_metrics,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    if args.logger == "wandb":
        try:
            import wandb
            if wandb.run is not None:
                wandb.summary.update(summary)
                wandb.finish()
        except Exception as e:
            print(f"W&B finalization warning: {e}")


if __name__ == "__main__":
    main()
