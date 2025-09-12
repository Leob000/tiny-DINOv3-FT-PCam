import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.pcam_hf import PCamH5HF
from src.models.backbone_dinov3 import DinoV3Backbone
from src.models.classifier import DinoV3PCam
from src.utils.metrics import eval_binary_scores
from src.utils.seed import set_seed


# Optional FLOPs (safe fallback if ptflops not installed or fails)
def try_flops(model, img_size=224, device="cpu"):
    try:
        from ptflops import get_model_complexity_info

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x)

        wrap = Wrap(model).to(device)
        macs, _ = get_model_complexity_info(
            wrap,
            (3, img_size, img_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        assert isinstance(macs, (int, float)), f"macs type is {type(macs)}"
        return float(macs * 2 / 1e9)  # GFLOPs (MACs->FLOPs x2)
    except Exception:
        return None


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_loss(model, loader, device, crit, max_batches=0):
    """Fast eval: average cross-entropy loss on GPU. No concatenation/CPU metrics."""
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader, start=1):
            time_batch_start = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(pixel_values=x)
            loss = crit(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            print(
                f"Eval batch {bi} | time: {time.time() - time_batch_start:.3f}s",
                end="\r",
            )
            if max_batches and bi >= max_batches:
                break
    return total_loss / max(1, total_n)


def evaluate(model, loader, device, max_batches=0):
    model.eval()
    probs, labels = [], []
    with torch.inference_mode():
        for bi, (x, y) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            logits = model(pixel_values=x)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
            if max_batches and bi >= max_batches:
                break
    p = np.concatenate(probs) if probs else np.array([0.5], dtype=np.float32)
    y = np.concatenate(labels) if labels else np.array([0], dtype=np.int64)
    return eval_binary_scores(p, y)


def time_latency(model, loader, device, warmup=20, iters=100, max_batches=0):
    model.eval()

    def _next(it, loader):
        try:
            return next(it)
        except StopIteration:
            return next(iter(loader))

    # warmup
    it = iter(loader)
    for _ in range(warmup):
        x, _ = _next(it, loader)
        x = x.to(device)
        _ = model(pixel_values=x)
        if device == "cuda":
            torch.cuda.synchronize()
        if device == "mps":
            torch.mps.synchronize()
        if max_batches:
            break
    # timed
    it = iter(loader)
    nimg, steps = 0, 0
    t0 = time.time()
    while steps < iters:
        x, _ = _next(it, loader)
        x = x.to(device)
        _ = model(pixel_values=x)
        if device == "cuda":
            torch.cuda.synchronize()
        if device == "mps":
            torch.mps.synchronize()
        nimg += x.size(0)
        steps += 1
        if max_batches:
            break
    dt = time.time() - t0
    # avoid div-by-zero
    if steps == 0 or nimg == 0:
        return float("nan"), float("nan")
    return (dt / nimg) * 1000.0, nimg / dt


def main():
    import wandb

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_log_every_steps",
        type=int,
        default=50,
        help="Log train loss & LR every X global steps (0 disables).",
    )
    ap.add_argument(
        "--val_eval_frac",
        type=float,
        default=0.25,
        help="Fraction of an epoch between MID-epoch FULL validation evals (0 disables).",
    )
    ap.add_argument(
        "--val_mid_epoch",
        action="store_true",
        help="Enable MID-epoch FULL validation (loss-only unless --val_heavy_mid).",
    )
    ap.add_argument(
        "--val_epoch_end",
        action="store_true",
        help="Enable END-of-epoch FULL validation (loss-only unless --val_heavy_end).",
    )
    ap.add_argument(
        "--val_heavy_mid",
        action="store_true",
        help="At MID-epoch evals, also compute AUROC/AUPRC/ECE/etc.",
    )
    ap.add_argument(
        "--val_heavy_end",
        action="store_true",
        help="At END-epoch evals, also compute AUROC/AUPRC/ECE/etc.",
    )
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    ap.add_argument("--wandb_project", type=str, default="dinov3-pcam-compress")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument(
        "--wandb_mode", type=str, default=None, choices=[None, "online", "offline"]
    )
    ap.add_argument("--method", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="src/data/pcam")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--model_id", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    ap.add_argument("--results_csv", type=str, default="results.csv")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Limit number of train batches per epoch (0=all).",
    )
    ap.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Limit number of val/test batches (0=all).",
    )
    ap.add_argument(
        "--skip_bench", action="store_true", help="Skip FLOPs and latency benchmarks."
    )
    ap.add_argument("--lat_warmup", type=int, default=20, help="Latency warmup iters.")
    ap.add_argument("--lat_iters", type=int, default=100, help="Latency timed iters.")
    args = ap.parse_args()
    if args.wandb:
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode

        run_name = args.wandb_run_name or f"linear_probe-res{args.resolution}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")
        wandb.config.update(
            {
                "method": args.method,
                "device": get_device(),
                "model_id": args.model_id,
                "train_log_every_steps": args.train_log_every_steps,
                "val_eval_frac": args.val_eval_frac,
                "val_mid_epoch": args.val_mid_epoch,
                "val_epoch_end": args.val_epoch_end,
                "val_heavy_mid": args.val_heavy_mid,
                "val_heavy_end": args.val_heavy_end,
            }
        )

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Datasets
    def h5p(split, kind):
        return os.path.join(
            args.data_dir, f"camelyonpatch_level_2_split_{split}_{kind}.h5"
        )

    train_ds = PCamH5HF(
        h5p("train", "x"),
        h5p("train", "y"),
        model_id=args.model_id,
        image_size=args.resolution,
    )
    val_ds = PCamH5HF(
        h5p("valid", "x"),
        h5p("valid", "y"),
        model_id=args.model_id,
        image_size=args.resolution,
    )
    test_ds = PCamH5HF(
        h5p("test", "x"),
        h5p("test", "y"),
        model_id=args.model_id,
        image_size=args.resolution,
    )

    tr = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    steps_per_epoch = len(tr)
    # Respect debug mode if you cap train batches
    effective_steps = min(steps_per_epoch, args.max_train_batches or steps_per_epoch)

    # Train logging schedule: every X global steps
    train_log_every = max(0, int(args.train_log_every_steps))

    # Validation schedule: every fraction of the epoch (mid-epoch)
    val_every_steps = 0
    if args.val_mid_epoch and args.val_eval_frac and args.val_eval_frac > 0:
        val_every_steps = max(1, int(round(effective_steps * args.val_eval_frac)))

    # FULL loaders for heavy metrics at the very end
    va_full = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,  #  keep workers alive across epochs
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    te_full = DataLoader(
        test_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # Model: backbone + linear head

    backbone = DinoV3Backbone(model_id=args.model_id, dtype=torch.float32)
    model = DinoV3PCam(backbone).to(device)

    # Freeze backbone (linear probe)
    for p in model.backbone.parameters():
        p.requires_grad = False

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        global_step = (epoch - 1) * steps_per_epoch  # absolute step across epochs
        since_log_loss_sum = 0.0
        since_log_count = 0

        for bi, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(pixel_values=x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # accumulate for train window
            bs = x.size(0)
            since_log_loss_sum += loss.item() * bs
            since_log_count += bs
            global_step += 1

            # (A) TRAIN logging: every X global steps
            if train_log_every > 0 and (global_step % train_log_every) == 0:
                mean_window_loss = since_log_loss_sum / max(1, since_log_count)
                if args.wandb:
                    wandb.log(
                        {
                            "global_step": global_step,
                            "train/loss_window": float(mean_window_loss),
                            "lr": opt.param_groups[0]["lr"],
                        }
                    )
                since_log_loss_sum = 0.0
                since_log_count = 0

            # (B) MID-epoch FULL validation at fraction boundaries (different schedule)
            mid_boundary = (
                args.val_mid_epoch
                and val_every_steps > 0
                and (((bi % val_every_steps) == 0) or (bi == effective_steps))
            )
            if mid_boundary:
                # Full val set: loss
                t0 = time.time()
                val_loss_mid = evaluate_loss(
                    model, va_full, device, crit, max_batches=0
                )
                log_dict = {
                    "global_step": global_step,
                    "val/loss_full": float(val_loss_mid),
                    "val/_scope": "mid_epoch_full",
                }

                # Optional heavy metrics on full val
                if args.val_heavy_mid:
                    mid_metrics = evaluate(model, va_full, device, max_batches=0)
                    log_dict.update(
                        {
                            "val/AUROC": mid_metrics["AUROC"],
                            "val/AUPRC": mid_metrics["AUPRC"],
                            "val/NLL": mid_metrics["NLL"],
                            "val/Brier": mid_metrics["Brier"],
                            "val/ECE": mid_metrics["ECE"],
                            "val/Acc@0.5": mid_metrics["Acc@0.5"],
                            "val/Sens@95%Spec": mid_metrics["Sens@95%Spec"],
                        }
                    )
                if args.wandb:
                    wandb.log(log_dict)
                print(
                    f"\n[mid-val] full loss={val_loss_mid:.4f} ({time.time() - t0:.2f}s)"
                )
        # END-OF-EPOCH validation (optional)
        val_loss_epoch_end = None
        if args.val_epoch_end:
            t0 = time.time()
            val_loss_epoch_end = evaluate_loss(
                model, va_full, device, crit, max_batches=0
            )
            log_dict = {
                "global_step": global_step,  # same x-axis
                "epoch": epoch,
                "train/loss_epoch": running_loss
                / max(1, len(tr.dataset)),  # rough scalar #type:ignore
                "val/loss_full": float(val_loss_epoch_end),
                "val/_scope": "epoch_end_full",
                "lr": opt.param_groups[0]["lr"],
            }

            if args.val_heavy_end:
                end_metrics = evaluate(model, va_full, device, max_batches=0)
                log_dict.update(
                    {
                        "val/AUROC": end_metrics["AUROC"],
                        "val/AUPRC": end_metrics["AUPRC"],
                        "val/NLL": end_metrics["NLL"],
                        "val/Brier": end_metrics["Brier"],
                        "val/ECE": end_metrics["ECE"],
                        "val/Acc@0.5": end_metrics["Acc@0.5"],
                        "val/Sens@95%Spec": end_metrics["Sens@95%Spec"],
                    }
                )

            if args.wandb:
                wandb.log(log_dict)
            print(
                f"[epoch-end val] full loss={val_loss_epoch_end:.4f} ({time.time() - t0:.2f}s)"
            )

            # Track best by epoch-end full val loss (if enabled)
            if best_state is None or val_loss_epoch_end < best_state["val_loss_full"]:
                best_state = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss_full": float(val_loss_epoch_end),
                    "args": vars(args),
                }

    print("Final evaluation (heavy metrics) on full val & test...")
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Full heavy evals (AUROC/AUPRC/etc.). Use full sets; set max_batches=0 explicitly.
    val_metrics_full = evaluate(model, va_full, device, max_batches=0)
    test_metrics_full = evaluate(model, te_full, device, max_batches=0)
    print("Final evaluation done.")

    # Systems metrics
    print("Params: computing")
    params = count_params(model)
    print("Params: done")
    flops = None
    lat_ms, thr = float("nan"), float("nan")
    if not args.skip_bench:
        print("FLOPs & Latency: computing")
        flops = try_flops(model, img_size=args.resolution, device=device)
        lat_ms, thr = time_latency(
            model,
            te_full,
            device=device,
            warmup=args.lat_warmup,
            iters=args.lat_iters,
            max_batches=args.max_eval_batches,
        )
        print("FLOPs & Latency: done")
        if args.wandb:
            # also log to the timeline (with the final global_step for nice x-axis)
            wandb.log(
                {
                    "global_step": global_step,
                    "systems/flops_g": (flops if flops is not None else None),
                    "systems/latency_ms": lat_ms,
                    "systems/throughput_img_s": thr,
                }
            )
            # and store in run summary
            assert wandb.run is not None
            wandb.run.summary.update(
                {
                    "systems/flops_g": (flops if flops is not None else None),
                    "systems/latency_ms": lat_ms,
                    "systems/throughput_img_s": thr,
                }
            )

    # Log row to results.csv
    row = {
        "method": args.method,
        "resolution": args.resolution,
        "params": params,
        "flops_g": flops if flops is not None else "",
        "AUROC": test_metrics_full["AUROC"],
        "AUPRC": test_metrics_full["AUPRC"],
        "Acc": test_metrics_full["Acc@0.5"],
        "NLL": test_metrics_full["NLL"],
        "Brier": test_metrics_full["Brier"],
        "ECE": test_metrics_full["ECE"],
        "Sens@95Spec": test_metrics_full["Sens@95%Spec"],
        "latency_ms": lat_ms,
        "throughput_img_s": thr,
    }
    if args.wandb:
        wandb.log(
            {
                # final full VAL metrics
                "final/val/AUROC": val_metrics_full["AUROC"],
                "final/val/AUPRC": val_metrics_full["AUPRC"],
                "final/val/NLL": val_metrics_full["NLL"],
                "final/val/Brier": val_metrics_full["Brier"],
                "final/val/ECE": val_metrics_full["ECE"],
                "final/val/Acc@0.5": val_metrics_full["Acc@0.5"],
                "final/val/Sens@95%Spec": val_metrics_full["Sens@95%Spec"],
                # final full TEST metrics
                "test/AUROC": test_metrics_full["AUROC"],
                "test/AUPRC": test_metrics_full["AUPRC"],
                "test/NLL": test_metrics_full["NLL"],
                "test/Brier": test_metrics_full["Brier"],
                "test/ECE": test_metrics_full["ECE"],
                "test/Acc@0.5": test_metrics_full["Acc@0.5"],
                "test/Sens@95%Spec": test_metrics_full["Sens@95%Spec"],
            }
        )

        best_val_loss = None
        best_epoch = None
        if best_state is not None:
            best_val_loss = best_state.get(
                "val_loss_full", best_state.get("val_loss_epoch")
            )
            best_epoch = best_state.get("epoch")

        wandb.run.summary["best_epoch"] = best_epoch  # type:ignore
        wandb.run.summary["best_val_loss"] = best_val_loss  # type:ignore

        # Log results.csv as an artifact so every run bundles the table
        try:
            art = wandb.Artifact("results_table", type="results")
            art.add_file(args.results_csv)
            wandb.run.log_artifact(art)  # type:ignore
        except Exception:
            pass

        wandb.finish()
    header = list(row.keys())
    exists = os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

    print("\n== Final VAL metrics ==")
    for k, v in val_metrics_full.items():
        print(f"{k}: {v:.4f}")

    print("\n== Final TEST metrics ==")
    for k, v in test_metrics_full.items():
        print(f"{k}: {v:.4f}")
    print(f"Params: {params:,}")
    print(f"FLOPs (G): {flops if flops is not None else 'N/A'}")
    print(f"Latency (ms/img): {lat_ms:.2f} | Throughput (img/s): {thr:.2f}")


if __name__ == "__main__":
    main()
