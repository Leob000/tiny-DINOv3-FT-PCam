import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.pcam_hf import PCamH5HF
from src.models.backbone_dinov3 import DinoV3Backbone, DinoV3PCam
from src.models.lora import inject_lora, LoRALinear
from src.utils.metrics import eval_binary_scores
from src.utils.seed import set_seed
from typing import Dict


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


def build_lr_log(opt, prefix: str, global_step: int):
    log = {"global_step": global_step}
    for i, g in enumerate(opt.param_groups):
        tag = g.get("name", f"group{i}")
        log[f"{prefix}/lr_{tag}"] = g["lr"]
    return log


def state_dict_cpu(m: nn.Module) -> Dict[str, torch.Tensor]:
    # safe for MPS/CUDA: store on CPU
    return {k: v.detach().to("cpu") for k, v in m.state_dict().items()}


def lora_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect only LoRA A/B tensors from LoRALinear modules, by qualified name."""
    out: Dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear) and mod.r > 0:
            out[f"{name}.A"] = mod.A.detach().cpu()
            out[f"{name}.B"] = mod.B.detach().cpu()
    return out


def save_checkpoint_full(path: str, model: nn.Module, args, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "type": "full",
        "state_dict": state_dict_cpu(model),
        "args": vars(args),
        "extra": extra or {},
    }
    torch.save(payload, path)


def main():
    import wandb

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        type=str,
        default="head_only",
        choices=["head_only", "fullft", "lora"],
        help="Training method: freeze backbone (head_only), full finetune, or LoRA.",
    )
    ap.add_argument(
        "--lr_head", type=float, default=None, help="Overrides LR for cls head."
    )
    ap.add_argument(
        "--lr_backbone",
        type=float,
        default=None,
        help="Overrides LR for backbone (fullft).",
    )
    ap.add_argument(
        "--lr_lora", type=float, default=None, help="Overrides LR for LoRA params."
    )

    # LoRA hyperparams
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated substrings of Linear module names to wrap with LoRA.",
    )
    ap.add_argument(
        "--lora_include_mlp",
        action="store_true",
        help="Also apply LoRA to MLP 'up_proj,down_proj'.",
    )

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
    ap.add_argument(
        "--wandb_mode", type=str, default=None, choices=[None, "online", "offline"]
    )
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
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--save_last", action="store_true")
    ap.add_argument("--save_best", action="store_true")
    args = ap.parse_args()
    EVAL_K = max(0, int(args.max_eval_batches))

    args.lr_head = args.lr if args.lr_head is None else args.lr_head
    args.lr_backbone = args.lr if args.lr_backbone is None else args.lr_backbone
    args.lr_lora = args.lr if args.lr_lora is None else args.lr_lora

    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.method}"
    run_dir = os.path.join(args.save_dir, run_name)
    if args.wandb:
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode

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
                "lr_head": args.lr_head,
                "lr_backbone": args.lr_backbone,
                "lr_lora": args.lr_lora,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_targets": args.lora_targets,
                "lora_include_mlp": args.lora_include_mlp,
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
    # Respect debug mode if train batches capped
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
        drop_last=False,
    )
    te_full = DataLoader(
        test_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # Model: backbone + linear head

    backbone = DinoV3Backbone(model_id=args.model_id, dtype=torch.float32)
    model = DinoV3PCam(backbone).to(device)

    # Decide what to train
    param_groups = []
    trainable_names = []
    lora_param_count = 0
    if args.method == "head_only":
        # freeze backbone
        for p in model.backbone.parameters():
            p.requires_grad = False
        param_groups.append(
            {"params": model.head.parameters(), "lr": args.lr_head, "name": "head"}
        )
        trainable_names.append("head")

    elif args.method == "fullft":
        # train everything; optionally give head its own LR
        for p in model.parameters():
            p.requires_grad = True
        # two groups so head can learn faster if desired
        head_params = list(model.head.parameters())
        bb_params = [
            p for n, p in model.named_parameters() if not n.startswith("head.")
        ]
        param_groups.append(
            {"params": bb_params, "lr": args.lr_backbone, "name": "backbone"}
        )
        param_groups.append({"params": head_params, "lr": args.lr_head, "name": "head"})
        trainable_names.extend(["backbone", "head"])

    elif args.method == "lora":
        # freeze backbone weights
        for p in model.backbone.parameters():
            p.requires_grad = False

        # choose targets
        target_keys = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
        if args.lora_include_mlp:
            target_keys += ["up_proj", "down_proj"]

        # inject LoRA into the DINOv3 module inside the wrapper
        dino_core = model.backbone.model
        lora_params = inject_lora(
            dino_core,
            target_keys=target_keys,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        model.to(device)

        # train LoRA params + head
        param_groups.append(
            {
                "params": lora_params,
                "lr": args.lr_lora,
                "weight_decay": 0.0,
                "name": "lora",
            }
        )
        param_groups.append(
            {"params": model.head.parameters(), "lr": args.lr_head, "name": "head"}
        )
        trainable_names.extend(["lora", "head"])

        lora_param_count = sum(p.numel() for p in lora_params)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Systems stats: params total + trainable (grouped here near param selection)
    params_total = count_params(model)

    def count_trainable(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    trainable_count = count_trainable(model)

    print(
        f"Params (total): {params_total:,} | "
        f"Trainable: {trainable_count:,} ({'+'.join(trainable_names)})"
    )
    global_step = 0
    if args.wandb:
        # Put them in both the timeline (step 0) and the run summary
        wandb.log(
            {
                "global_step": global_step,
                "systems/params_total": params_total,
                "systems/params_trainable_count": trainable_count,
                "systems/params_trainable_groups": trainable_names,
                **(
                    {"systems/params_lora": lora_param_count}
                    if lora_param_count
                    else {}
                ),
            }
        )
        assert wandb.run is not None
        wandb.run.summary["systems/params_total"] = params_total
        wandb.run.summary["systems/params_trainable_count"] = trainable_count
        wandb.run.summary["systems/params_trainable_groups"] = trainable_names
        if lora_param_count:
            wandb.run.summary["systems/params_lora"] = lora_param_count

    # build optimizer with param groups
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    crit = nn.CrossEntropyLoss()
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        train_samples_seen = 0
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
            running_loss += loss.item() * bs
            train_samples_seen += bs
            global_step += 1

            # Train logging: every X global steps
            if train_log_every > 0 and (global_step % train_log_every) == 0:
                mean_window_loss = since_log_loss_sum / max(1, since_log_count)
                if args.wandb:
                    log = {
                        "global_step": global_step,
                        "train/loss_window": float(mean_window_loss),
                    }
                    log.update(
                        build_lr_log(opt, prefix="train", global_step=global_step)
                    )
                    wandb.log(log)
                since_log_loss_sum = 0.0
                since_log_count = 0

            # MID-epoch FULL validation at fraction boundaries (different schedule)
            mid_boundary = (
                args.val_mid_epoch
                and val_every_steps > 0
                and (((bi % val_every_steps) == 0) or (bi == effective_steps))
            )
            if mid_boundary:
                # Full val set: loss
                t0 = time.time()
                was_training = model.training
                val_loss_mid = evaluate_loss(
                    model, va_full, device, crit, max_batches=EVAL_K
                )
                log_dict = {
                    "global_step": global_step,
                    "val/loss_full": float(val_loss_mid),
                    "val/_scope": "mid_epoch_full",
                }

                # Optional heavy metrics on full val
                if args.val_heavy_mid:
                    mid_metrics = evaluate(model, va_full, device, max_batches=EVAL_K)
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
                model.train(was_training)

            if args.max_train_batches and bi >= args.max_train_batches:
                break

        # END-OF-EPOCH validation
        val_loss_epoch_end = None
        if args.val_epoch_end:
            t0 = time.time()
            was_training = model.training
            val_loss_epoch_end = evaluate_loss(
                model, va_full, device, crit, max_batches=EVAL_K
            )
            log_dict = {
                "global_step": global_step,  # same x-axis
                "epoch": epoch,
                "train/loss_epoch": (running_loss / max(1, train_samples_seen)),
                "val/loss_full": float(val_loss_epoch_end),
                "val/_scope": "epoch_end_full",
            }

            if args.val_heavy_end:
                end_metrics = evaluate(model, va_full, device, max_batches=EVAL_K)
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
            log_dict.update(
                build_lr_log(opt, prefix="epoch_end", global_step=global_step)
            )
            if args.wandb:
                wandb.log(log_dict)
            print(
                f"[epoch-end val] full loss={val_loss_epoch_end:.4f} ({time.time() - t0:.2f}s)"
            )
            model.train(was_training)

            # Track best by epoch-end full val loss (if enabled)
            if best_state is None or val_loss_epoch_end < best_state["val_loss_full"]:
                best_state = {
                    "state_dict": state_dict_cpu(model),
                    "epoch": epoch,
                    "val_loss_full": float(val_loss_epoch_end),
                }
                if args.save_best:
                    save_checkpoint_full(
                        os.path.join(run_dir, "best_full.pt"),
                        model,
                        args,
                        extra={
                            "val_loss_full": float(val_loss_epoch_end),
                            "epoch": epoch,
                        },
                    )

    print("Final evaluation (heavy metrics) on full val & test...")
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"], strict=True)

    # Full heavy evals (AUROC/AUPRC/etc.). Use full sets; set max_batches=0 explicitly.
    val_metrics_full = evaluate(model, va_full, device, max_batches=EVAL_K)
    test_metrics_full = evaluate(model, te_full, device, max_batches=EVAL_K)
    print("Final evaluation done.")

    # Systems metrics
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
            max_batches=EVAL_K,
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
        wandb.run.summary["best_epoch"] = best_epoch  # type: ignore
        wandb.run.summary["best_val_loss"] = best_val_loss  # type: ignore
        wandb.finish()

    print("\n== Final VAL metrics ==")
    for k, v in val_metrics_full.items():
        print(f"{k}: {v:.4f}")

    print("\n== Final TEST metrics ==")
    for k, v in test_metrics_full.items():
        print(f"{k}: {v:.4f}")
    print(f"FLOPs (G): {flops if flops is not None else 'N/A'}")
    print(f"Latency (ms/img): {lat_ms:.2f} | Throughput (img/s): {thr:.2f}")

    # Save last checkpoint if requested
    if args.save_last:
        save_checkpoint_full(
            os.path.join(run_dir, "last_full.pt"),
            model,
            args,
            extra={
                "final_val": val_metrics_full,
                "final_test": test_metrics_full,
                "epoch": args.epochs,
            },
        )


if __name__ == "__main__":
    main()
