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


def evaluate_loss_streaming(model, loader, it, device, crit, k):
    """
    Consume next k batches from (persistent) iterator `it` of `loader`.
    Returns (avg_loss, updated_iterator).
    """
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for _ in range(k):
            if it is None:
                it = iter(loader)
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(pixel_values=x)
            loss = crit(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
    return (total_loss / max(1, total_n)), it


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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_frac",
        type=float,
        default=0.25,
        help="Fraction of an epoch between mid-epoch logs/evals. 0 disables.",
    )
    ap.add_argument(
        "--mid_eval_batches",
        type=int,
        default=4,
        help="How many validation batches to use for each mid-epoch mini-eval.",
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
        import wandb

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
    )
    steps_per_epoch = len(tr)
    # Respect debug mode if you cap train batches
    effective_steps = min(steps_per_epoch, args.max_train_batches or steps_per_epoch)

    # Convert fraction → step interval (at least 1 if enabled)
    log_every_steps = 0
    if args.eval_frac and args.eval_frac > 0:
        log_every_steps = max(1, int(round(effective_steps * args.eval_frac)))

    # FAST loaders for mid-epoch loss-only evals (no worker spawn cost)
    va_fast = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,  # no worker processes
        pin_memory=(device == "cuda"),
    )

    # Keep a persistent iterator across mini-evals
    va_fast_iter = None

    # FULL loaders for heavy metrics at the very end
    va_full = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,  #  keep workers alive across epochs
        pin_memory=(device == "cuda"),
    )
    te_full = DataLoader(
        test_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=(device == "cuda"),
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

            # accumulate loss for the current logging window
            this_loss = loss.item()
            bs = x.size(0)
            since_log_loss_sum += this_loss * bs
            since_log_count += bs
            global_step += 1

            # run at every fraction of the epoch (and at the end of the effective epoch)
            hit_boundary = (log_every_steps > 0) and (
                (bi % log_every_steps) == 0 or bi == effective_steps
            )
            if hit_boundary:
                # 1) mean train loss over the window
                mean_window_loss = since_log_loss_sum / max(1, since_log_count)

                # 2) fast mini validation: **loss only**, streaming K batches
                time_val_eval = time.time()
                val_loss_window, va_fast_iter = evaluate_loss_streaming(
                    model, va_fast, va_fast_iter, device, crit, k=args.mid_eval_batches
                )
                print(
                    f"\n[val] loss={val_loss_window:.4f} | time: {time.time() - time_val_eval:.2f}s"
                )

                # 3) single W&B log step (no heavy metrics here)
                if args.wandb:
                    import wandb

                    wandb.log(
                        {
                            "global_step": global_step,
                            "train/loss_window": float(mean_window_loss),
                            "val/loss_window": float(val_loss_window),
                            "lr": opt.param_groups[0]["lr"],
                            "val/_scope": "mid_epoch",
                        }
                    )

                # 4) reset window accumulators
                since_log_loss_sum = 0.0
                since_log_count = 0

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())
            if args.max_train_batches and bi >= args.max_train_batches:
                break
        # train_loss = running_loss / max(
        #     1, (args.batch_size * min(len(tr), args.max_train_batches or len(tr)))
        # )

        # Epoch-end fast validation (loss only)
        # quick epoch-end check: loss-only on a few batches (or set k to cover all if you like)
        k_epoch = (
            args.max_eval_batches
            if args.max_eval_batches > 0
            else args.mid_eval_batches
        )
        val_loss_epoch, va_fast_iter = evaluate_loss_streaming(
            model, va_fast, va_fast_iter, device, crit, k=k_epoch
        )
        print(f"[val] loss={val_loss_epoch:.4f}")

        if args.wandb:
            import wandb

            wandb.log(
                {
                    "global_step": global_step,  # last step this epoch
                    "epoch": epoch,
                    "train/loss_epoch": running_loss
                    / max(1, len(tr.dataset)),  # rough scalar #type:ignore
                    "val/loss_epoch": float(val_loss_epoch),
                    "lr": opt.param_groups[0]["lr"],
                    "val/_scope": "epoch_end",
                }
            )

        # Track best by val loss
        if best_state is None or val_loss_epoch < best_state["val_loss_epoch"]:
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss_epoch": float(val_loss_epoch),
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
        import wandb

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
        # optional summaries
        wandb.run.summary["best_epoch"] = best_state["epoch"] if best_state else None  # type: ignore
        wandb.run.summary["best_val_loss"] = (  # type:ignore
            best_state["val_loss_epoch"] if best_state else None
        )

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
