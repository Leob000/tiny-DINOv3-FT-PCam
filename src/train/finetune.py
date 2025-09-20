import argparse
import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.backbone_dinov3 import DinoV3Backbone, DinoV3PCam
from src.models.lora import LoRALinear, inject_lora
from src.utils.eval_utils import (
    count_params,
    try_flops,
    get_device,
    evaluate,
    evaluate_loss,
    time_latency,
)
from src.utils.seed import set_seed
from src.utils.data_utils import build_eval_loaders, build_train_loader


def build_lr_log(opt, prefix: str, global_step: int):
    log = {"global_step": global_step}
    for i, g in enumerate(opt.param_groups):
        tag = g.get("name", f"group{i}")
        log[f"{prefix}/lr_{tag}"] = g["lr"]
    return log


def build_cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup to step `warmup_steps`, then cosine decay to step `total_steps`.
    Works per-step (call scheduler.step() once after each optimizer.step()).
    """
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    warmup_steps = min(warmup_steps, total_steps - 1)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def enable_and_collect_norms_bias(
    backbone_core: nn.Module, mode: str
) -> list[nn.Parameter]:
    """
    backbone_core: the HF DINOv3 module (e.g., model.backbone.model)
    mode: 'none' | 'norms' | 'bias' | 'both'
    Returns a de-duplicated list of Parameters that should be optimized.
    """
    if mode == "none":
        return []

    params = []

    if mode in ("norms", "both"):
        for m in backbone_core.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    p.requires_grad = True
                    params.append(p)

    if mode in ("bias", "both"):
        for name, p in backbone_core.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
                params.append(p)

    uniq = list({id(p): p for p in params}.values())
    return uniq


def state_dict_cpu(m: nn.Module) -> Dict[str, torch.Tensor]:
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


def save_checkpoint_head(path: str, model: nn.Module, args, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # grab LayerNorm/bias overrides if they were trained
    backbone_overrides = collect_backbone_overrides(
        model.backbone.model,  # type:ignore
        args.train_norms_bias,
    )

    payload = {
        "type": "head_only",
        "head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},  # type:ignore
        "backbone_model_id": args.model_id,
        "image_size": args.resolution,
        "args": vars(args),
        "extra": extra or {},
    }
    # only store if non-empty
    if backbone_overrides:
        payload["backbone_overrides"] = backbone_overrides
        payload["train_norms_bias_mode"] = args.train_norms_bias
    torch.save(payload, path)


def save_checkpoint_lora(path: str, model: nn.Module, args, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # grab LayerNorm/bias overrides if they were trained
    backbone_overrides = collect_backbone_overrides(
        model.backbone.model,  # type:ignore
        args.train_norms_bias,
    )

    payload = {
        "type": "lora",
        "adapters": lora_adapter_state_dict(model),  # only A/B
        "head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},  # type:ignore
        "backbone_model_id": args.model_id,
        "image_size": args.resolution,
        "lora_hparams": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": args.lora_targets,
            "include_mlp": args.lora_include_mlp,
        },
        "args": vars(args),
        "extra": extra or {},
    }
    # only store if non-empty
    if backbone_overrides:
        payload["backbone_overrides"] = backbone_overrides
        payload["train_norms_bias_mode"] = args.train_norms_bias
    torch.save(payload, path)


def collect_backbone_overrides(
    backbone_core: nn.Module, mode: str
) -> Dict[str, torch.Tensor]:
    """
    Return a {param_name: tensor} dict for LayerNorm (weights/biases) and/or any *.bias
    inside the backbone, matching what was enabled by `train_norms_bias`.
    Tensors are detached to CPU so checkpoints stay lightweight.
    """
    if mode == "none":
        return {}

    overrides: Dict[str, torch.Tensor] = {}

    if mode in ("norms", "both"):
        for mod_name, mod in backbone_core.named_modules():
            if isinstance(mod, nn.LayerNorm):
                for p_name, p in mod.named_parameters(recurse=False):
                    full = f"{mod_name}.{p_name}" if mod_name else p_name
                    overrides[full] = p.detach().cpu().clone()

    if mode in ("bias", "both"):
        for name, p in backbone_core.named_parameters():
            if name.endswith(".bias"):
                overrides[name] = p.detach().cpu().clone()

    return overrides


def parse_args():
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
    ap.add_argument(
        "--train_norms_bias",
        type=str,
        default="none",
        choices=["none", "norms", "bias", "both"],
        help="If not 'none', also train LayerNorm params ('norms'), biases ('bias'), or both ('both') in the backbone for head_only/LoRA.",
    )
    ap.add_argument(
        "--lr_norms_bias",
        type=float,
        default=None,
        help="LR for norms/bias param group. Defaults to lr_head (head_only) or lr_lora (LoRA).",
    )
    ap.add_argument(
        "--warmup_steps", type=int, default=200, help="Linear warmup steps."
    )
    ap.add_argument(
        "--grad_clip", type=float, default=1.0, help="Grad clip max-norm; 0 disables."
    )
    ap.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="CrossEntropy label smoothing.",
    )
    ap.add_argument(
        "--val_epoch0",
        action="store_true",
        help="Evaluate on the full validation set before any training (epoch=0).",
    )
    ap.add_argument(
        "--val_heavy_zero",
        action="store_true",
        help="At epoch-0 eval, also compute AUROC/AUPRC/ECE/etc.",
    )
    ap.add_argument(
        "--aug_histology",
        action="store_true",
        help="Enable histology-friendly training augmentations (flips/rot90/color jitter).",
    )
    ap.add_argument(
        "--tta_eval",
        action="store_true",
        help="Enable test-time augmentation (flips + 90° rotations) for val/test heavy evals.",
    )
    ap.add_argument(
        "--select_metric",
        type=str,
        default="auroc",
        choices=["val_loss", "auroc", "sens95", "nll", "brier", "ece", "acc"],
        help="Metric to select the best epoch/checkpoint. 'auroc' = maximize AUROC; 'val_loss'/'nll'/'brier'/'ece' = minimize; 'sens95'/'acc' = maximize.",
    )

    return ap.parse_args()


def _is_better(new, best, metric):
    maximize = {"auroc", "sens95", "acc"}
    minimize = {"val_loss", "nll", "brier", "ece"}
    if metric in maximize:
        return (best is None) or (new > best)
    if metric in minimize:
        return (best is None) or (new < best)
    raise ValueError(f"Unknown select_metric: {metric}")


class LinearTrainer:
    def __init__(self, args):
        self.args = args
        self.run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{self.args.method}"
        self.run_dir = os.path.join(self.args.save_dir, self.run_name)
        self.eval_batches = max(0, int(self.args.max_eval_batches))

        set_seed(self.args.seed)
        self.device = get_device()
        print(f"Device: {self.device}")

        self.wandb = None
        self.wandb_run = None
        self._init_wandb()

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self._build_dataloaders()
        self.steps_per_epoch = len(self.train_loader)
        self.effective_steps = min(
            self.steps_per_epoch,
            self.args.max_train_batches or self.steps_per_epoch,
        )
        self.train_log_every = max(0, int(self.args.train_log_every_steps))
        self.val_every_steps = self._compute_val_schedule()

        (
            self.model,
            self.optimizer,
            self.scheduler,
            meta,
        ) = self._prepare_model_and_optimizer()
        self.trainable_names = meta["trainable_names"]
        self.lora_param_count = meta["lora_param_count"]
        self.params_total = meta["params_total"]
        self.trainable_count = meta["trainable_count"]

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        self.global_step = 0
        self.best_state = None

    def train(self):
        self._log_system_stats()
        self._maybe_epoch_zero_eval()

        for epoch in range(1, self.args.epochs + 1):
            running_loss, train_samples_seen = self._train_epoch(epoch)
            if self.args.val_epoch_end:
                mean_loss = running_loss / max(1, train_samples_seen)
                self._handle_epoch_end_eval(epoch, mean_loss)

        val_metrics, test_metrics, flops, lat_ms, thr = self._final_evaluation()
        self._print_final_metrics(val_metrics, test_metrics, flops, lat_ms, thr)
        self._save_last_checkpoint(val_metrics, test_metrics)
        self._finalize_wandb()

    def _init_wandb(self):
        if not self.args.wandb:
            return
        import wandb

        if self.args.wandb_mode:
            os.environ["WANDB_MODE"] = self.args.wandb_mode

        run = wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=self.run_name,
            config=vars(self.args),  # type:ignore
        )
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")
        wandb.config.update(
            {
                "method": self.args.method,
                "run_name": self.run_name,
                "device": self.device,
                "model_id": self.args.model_id,
                "train_log_every_steps": self.args.train_log_every_steps,
                "val_eval_frac": self.args.val_eval_frac,
                "val_mid_epoch": self.args.val_mid_epoch,
                "val_epoch_end": self.args.val_epoch_end,
                "val_heavy_mid": self.args.val_heavy_mid,
                "val_heavy_end": self.args.val_heavy_end,
                "lr_head": self.args.lr_head,
                "lr_backbone": self.args.lr_backbone,
                "lr_lora": self.args.lr_lora,
                "lora_r": self.args.lora_r,
                "lora_alpha": self.args.lora_alpha,
                "lora_dropout": self.args.lora_dropout,
                "lora_targets": self.args.lora_targets,
                "lora_include_mlp": self.args.lora_include_mlp,
                "train_norms_bias": self.args.train_norms_bias,
                "lr_norms_bias": self.args.lr_norms_bias,
            }
        )
        self.wandb = wandb
        self.wandb_run = run

    def _build_dataloaders(self):
        train_loader = build_train_loader(
            data_dir=self.args.data_dir,
            model_id=self.args.model_id,
            image_size=self.args.resolution,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            device=self.device,
            aug_histology=self.args.aug_histology,
        )
        val_loader, test_loader = build_eval_loaders(
            data_dir=self.args.data_dir,
            model_id=self.args.model_id,
            image_size=self.args.resolution,
            batch_size=self.args.val_batch_size,
            num_workers=self.args.num_workers,
            device=self.device,
        )
        return train_loader, val_loader, test_loader

    def _compute_val_schedule(self) -> int:
        if not self.args.val_mid_epoch:
            return 0
        if not self.args.val_eval_frac or self.args.val_eval_frac <= 0:
            return 0
        return max(1, int(round(self.effective_steps * self.args.val_eval_frac)))

    def _prepare_model_and_optimizer(self):
        backbone = DinoV3Backbone(model_id=self.args.model_id, dtype=torch.float32)
        model = DinoV3PCam(backbone).to(self.device)

        param_groups = []
        trainable_names = []
        lora_param_count = 0

        if self.args.method == "head_only":
            for p in model.backbone.parameters():
                p.requires_grad = False
            param_groups.append(
                {
                    "params": model.head.parameters(),
                    "lr": self.args.lr_head,
                    "name": "head",
                }
            )
            trainable_names.append("head")
            nb_params = enable_and_collect_norms_bias(
                model.backbone.model, self.args.train_norms_bias
            )
            if nb_params:
                lr_nb = (
                    self.args.lr_norms_bias
                    if self.args.lr_norms_bias is not None
                    else self.args.lr_head
                )
                param_groups.append(
                    {
                        "params": nb_params,
                        "lr": lr_nb,
                        "weight_decay": 0.0,
                        "name": "norms_bias",
                    }
                )
                trainable_names.append("norms_bias")

        elif self.args.method == "fullft":
            for p in model.parameters():
                p.requires_grad = True
            head_params = list(model.head.parameters())
            bb_params = [
                p for n, p in model.named_parameters() if not n.startswith("head.")
            ]
            param_groups.append(
                {"params": bb_params, "lr": self.args.lr_backbone, "name": "backbone"}
            )
            param_groups.append(
                {"params": head_params, "lr": self.args.lr_head, "name": "head"}
            )
            trainable_names.extend(["backbone", "head"])

        elif self.args.method == "lora":
            for p in model.backbone.parameters():
                p.requires_grad = False

            target_keys = [
                s.strip() for s in self.args.lora_targets.split(",") if s.strip()
            ]
            if self.args.lora_include_mlp:
                target_keys += ["up_proj", "down_proj"]

            dino_core = model.backbone.model
            lora_params = inject_lora(
                dino_core,
                target_keys=target_keys,
                r=self.args.lora_r,
                alpha=self.args.lora_alpha,
                dropout=self.args.lora_dropout,
            )
            model.to(self.device)

            nb_params = enable_and_collect_norms_bias(
                model.backbone.model, self.args.train_norms_bias
            )
            if nb_params:
                lr_nb = (
                    self.args.lr_norms_bias
                    if self.args.lr_norms_bias is not None
                    else self.args.lr_lora
                )
                param_groups.append(
                    {
                        "params": nb_params,
                        "lr": lr_nb,
                        "weight_decay": 0.0,
                        "name": "norms_bias",
                    }
                )
                trainable_names.append("norms_bias")

            param_groups.append(
                {
                    "params": lora_params,
                    "lr": self.args.lr_lora,
                    "weight_decay": 0.0,
                    "name": "lora",
                }
            )
            param_groups.append(
                {
                    "params": model.head.parameters(),
                    "lr": self.args.lr_head,
                    "name": "head",
                }
            )
            trainable_names.extend(["lora", "head"])
            lora_param_count = sum(p.numel() for p in lora_params)
        else:
            raise ValueError(f"Unknown method: {self.args.method}")

        params_total = count_params(model)
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        total_steps = max(1, self.args.epochs * max(1, self.effective_steps))
        scheduler = build_cosine_with_warmup(
            optimizer, self.args.warmup_steps, total_steps
        )

        meta = {
            "trainable_names": trainable_names,
            "lora_param_count": lora_param_count,
            "params_total": params_total,
            "trainable_count": trainable_count,
        }
        return model, optimizer, scheduler, meta

    def _log_system_stats(self):
        log = {
            "global_step": self.global_step,
            "systems/params_total": self.params_total,
            "systems/params_trainable_count": self.trainable_count,
            "systems/params_trainable_groups": self.trainable_names,
        }
        if self.lora_param_count:
            log["systems/params_lora"] = self.lora_param_count
        self._wandb_log(log)
        summary = {
            "systems/params_total": self.params_total,
            "systems/params_trainable_count": self.trainable_count,
            "systems/params_trainable_groups": self.trainable_names,
        }
        if self.lora_param_count:
            summary["systems/params_lora"] = self.lora_param_count
        self._wandb_summary_update(summary)

    def _maybe_epoch_zero_eval(self):
        if not self.args.val_epoch0:
            return
        t0 = time.time()
        val_loss, metrics = self._evaluate_val_full(heavy=self.args.val_heavy_zero)
        dt = time.time() - t0
        log_dict = {
            "global_step": self.global_step,
            "epoch": 0,
            "val/loss_full": float(val_loss),
            "val/_scope": "epoch0_full",
        }
        if self.args.val_heavy_zero and metrics:
            log_dict.update(self._format_val_metrics(metrics))
        self._wandb_log(log_dict)
        print(f"[epoch-0 val] full loss={val_loss:.4f} ({dt:.2f}s)")

    def _train_epoch(self, epoch: int) -> Tuple[float, int]:
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        running_loss = 0.0
        train_samples_seen = 0
        since_log_loss_sum = 0.0
        since_log_count = 0

        for batch_idx, batch in enumerate(pbar, start=1):
            loss_val, batch_size = self._train_step(batch)
            loss_sum = loss_val * batch_size
            running_loss += loss_sum
            train_samples_seen += batch_size
            since_log_loss_sum += loss_sum
            since_log_count += batch_size

            if (
                self.train_log_every > 0
                and (self.global_step % self.train_log_every) == 0
            ):
                mean_window_loss = since_log_loss_sum / max(1, since_log_count)
                log = {
                    "global_step": self.global_step,
                    "train/loss_window": float(mean_window_loss),
                }
                log.update(
                    build_lr_log(
                        self.optimizer, prefix="train", global_step=self.global_step
                    )
                )
                self._wandb_log(log)
                since_log_loss_sum = 0.0
                since_log_count = 0

            if self._should_run_mid_eval(batch_idx):
                self._run_mid_epoch_eval()

            if self.args.max_train_batches and batch_idx >= self.args.max_train_batches:
                break

        return running_loss, train_samples_seen

    def _train_step(self, batch):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        logits = self.model(pixel_values=x)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.args.grad_clip and self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        bs = x.size(0)
        self.global_step += 1
        return loss.item(), bs

    def _should_run_mid_eval(self, batch_idx: int) -> bool:
        if not self.args.val_mid_epoch or self.val_every_steps <= 0:
            return False
        return (
            batch_idx % self.val_every_steps
        ) == 0 or batch_idx == self.effective_steps

    def _run_mid_epoch_eval(self):
        t0 = time.time()
        val_loss, metrics = self._evaluate_val_full(heavy=self.args.val_heavy_mid)
        dt = time.time() - t0
        log_dict = {
            "global_step": self.global_step,
            "val/loss_full": float(val_loss),
            "val/_scope": "mid_epoch_full",
        }
        if self.args.val_heavy_mid and metrics:
            log_dict.update(self._format_val_metrics(metrics))
        self._wandb_log(log_dict)
        print(f"\n[mid-val] full loss={val_loss:.4f} ({dt:.2f}s)")

    def _handle_epoch_end_eval(self, epoch: int, train_loss_epoch: float):
        t0 = time.time()
        need_heavy = self.args.val_heavy_end or (self.args.select_metric != "val_loss")
        val_loss, metrics = self._evaluate_val_full(heavy=need_heavy)
        dt = time.time() - t0
        log_dict = {
            "global_step": self.global_step,
            "epoch": epoch,
            "train/loss_epoch": float(train_loss_epoch),
            "val/loss_full": float(val_loss),
            "val/_scope": "epoch_end_full",
        }
        if self.args.val_heavy_end and metrics:
            log_dict.update(self._format_val_metrics(metrics))

        sel_value = self._select_metric_value(val_loss, metrics)
        tie_value = float(metrics["NLL"]) if metrics is not None else float("inf")
        log_dict["select/metric"] = self.args.select_metric
        log_dict["select/value"] = float(sel_value)
        log_dict["select/tie_nll"] = tie_value
        log_dict.update(
            build_lr_log(
                self.optimizer, prefix="epoch_end", global_step=self.global_step
            )
        )
        self._wandb_log(log_dict)
        print(f"[epoch-end val] full loss={val_loss:.4f} ({dt:.2f}s)")

        self._maybe_update_best(epoch, val_loss, sel_value, tie_value)
        if self.wandb_run is not None:
            self.wandb_run.summary["select_metric"] = (
                self.best_state.get("select_metric") if self.best_state else None
            )
            self.wandb_run.summary["select_value"] = (
                self.best_state.get("select_value") if self.best_state else None
            )

    def _select_metric_value(
        self, val_loss: float, metrics: Optional[Dict[str, float]]
    ):
        metric = self.args.select_metric
        if metric == "val_loss":
            return float(val_loss)
        if not metrics:
            raise ValueError("Heavy metrics required for selection")
        key_map = {
            "auroc": "AUROC",
            "sens95": "Sens@95%Spec",
            "nll": "NLL",
            "brier": "Brier",
            "ece": "ECE",
            "acc": "Acc@0.5",
        }
        key = key_map.get(metric)
        if key is None:
            raise ValueError(f"Unknown select_metric: {metric}")
        return float(metrics[key])

    def _maybe_update_best(
        self, epoch: int, val_loss: float, sel_value: float, tie_value: float
    ):
        current_best = (
            None if self.best_state is None else self.best_state["select_value"]
        )
        if (
            self.best_state is None
            or _is_better(sel_value, current_best, self.args.select_metric)
            or (
                sel_value == self.best_state["select_value"]
                and tie_value < self.best_state.get("tie_nll", float("inf"))
            )
        ):
            self.best_state = {
                "state_dict": state_dict_cpu(self.model),
                "epoch": epoch,
                "val_loss_full": float(val_loss),
                "select_metric": self.args.select_metric,
                "select_value": float(sel_value),
                "tie_nll": float(tie_value),
            }
            if self.args.save_best:
                extra = {
                    "val_loss_full": float(val_loss),
                    "epoch": epoch,
                    "select_metric": self.args.select_metric,
                    "select_value": float(sel_value),
                    "tie_nll": float(tie_value),
                }
                self._save_checkpoint("best", extra)

    def _evaluate_val_full(self, heavy: bool):
        was_training = self.model.training
        val_loss = evaluate_loss(
            self.model,
            self.val_loader,
            self.device,
            self.criterion,
            max_batches=self.eval_batches,
        )
        metrics = None
        if heavy:
            metrics = evaluate(
                self.model,
                self.val_loader,
                self.device,
                max_batches=self.eval_batches,
            )
        self.model.train(was_training)
        return float(val_loss), metrics

    def _evaluate_metrics(self, loader, use_tta: bool = False):
        was_training = self.model.training
        metrics = evaluate(
            self.model,
            loader,
            self.device,
            max_batches=self.eval_batches,
            use_tta=use_tta,
        )
        self.model.train(was_training)
        return metrics

    def _format_val_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        return {
            "val/AUROC": metrics["AUROC"],
            "val/AUPRC": metrics["AUPRC"],
            "val/NLL": metrics["NLL"],
            "val/Brier": metrics["Brier"],
            "val/ECE": metrics["ECE"],
            "val/Acc@0.5": metrics["Acc@0.5"],
            "val/Sens@95%Spec": metrics["Sens@95%Spec"],
        }

    def _final_evaluation(self):
        print("Final evaluation (heavy metrics) on full val & test...")
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["state_dict"], strict=True)

        val_metrics_full = self._evaluate_metrics(
            self.val_loader, use_tta=self.args.tta_eval
        )
        test_metrics_full = self._evaluate_metrics(
            self.test_loader, use_tta=self.args.tta_eval
        )
        print("Final evaluation done.")

        flops = None
        lat_ms, thr = float("nan"), float("nan")
        if not self.args.skip_bench:
            print("FLOPs & Latency: computing")
            flops = try_flops(
                self.model, img_size=self.args.resolution, device=self.device
            )
            lat_ms, thr = time_latency(
                self.model,
                self.test_loader,
                device=self.device,
                warmup=self.args.lat_warmup,
                iters=self.args.lat_iters,
                max_batches=self.eval_batches,
            )
            print("FLOPs & Latency: done")
            self._wandb_log(
                {
                    "global_step": self.global_step,
                    "systems/flops_g": flops,
                    "systems/latency_ms": lat_ms,
                    "systems/throughput_img_s": thr,
                }
            )
            self._wandb_summary_update(
                {
                    "systems/flops_g": flops,
                    "systems/latency_ms": lat_ms,
                    "systems/throughput_img_s": thr,
                }
            )

        final_log = {
            "final/val/AUROC": val_metrics_full["AUROC"],
            "final/val/AUPRC": val_metrics_full["AUPRC"],
            "final/val/NLL": val_metrics_full["NLL"],
            "final/val/Brier": val_metrics_full["Brier"],
            "final/val/ECE": val_metrics_full["ECE"],
            "final/val/Acc@0.5": val_metrics_full["Acc@0.5"],
            "final/val/Sens@95%Spec": val_metrics_full["Sens@95%Spec"],
            "test/AUROC": test_metrics_full["AUROC"],
            "test/AUPRC": test_metrics_full["AUPRC"],
            "test/NLL": test_metrics_full["NLL"],
            "test/Brier": test_metrics_full["Brier"],
            "test/ECE": test_metrics_full["ECE"],
            "test/Acc@0.5": test_metrics_full["Acc@0.5"],
            "test/Sens@95%Spec": test_metrics_full["Sens@95%Spec"],
        }
        final_log["global_step"] = self.global_step
        self._wandb_log(final_log)  # type:ignore

        return val_metrics_full, test_metrics_full, flops, lat_ms, thr

    def _print_final_metrics(
        self,
        val_metrics_full: Dict[str, float],
        test_metrics_full: Dict[str, float],
        flops: Optional[float],
        lat_ms: float,
        thr: float,
    ):
        print("\n== Final VAL metrics ==")
        for k, v in val_metrics_full.items():
            print(f"{k}: {v:.4f}")

        print("\n== Final TEST metrics ==")
        for k, v in test_metrics_full.items():
            print(f"{k}: {v:.4f}")
        print(f"FLOPs (G): {flops if flops is not None else 'N/A'}")
        print(f"Latency (ms/img): {lat_ms:.2f} | Throughput (img/s): {thr:.2f}")

    def _save_last_checkpoint(self, val_metrics, test_metrics):
        if not self.args.save_last:
            return
        extra = {
            "final_val": val_metrics,
            "final_test": test_metrics,
            "epoch": self.args.epochs,
        }
        self._save_checkpoint("last", extra)

    def _save_checkpoint(self, prefix: str, extra):
        filename = self._checkpoint_filename(prefix)
        path = os.path.join(self.run_dir, filename)
        if self.args.method == "fullft":
            save_checkpoint_full(path, self.model, self.args, extra=extra)
        elif self.args.method == "lora":
            save_checkpoint_lora(path, self.model, self.args, extra=extra)
        elif self.args.method == "head_only":
            save_checkpoint_head(path, self.model, self.args, extra=extra)
        else:
            raise ValueError(f"Unknown method: {self.args.method}")

    def _checkpoint_filename(self, prefix: str) -> str:
        if self.args.method == "fullft":
            return f"{prefix}_full.pt"
        if self.args.method == "lora":
            return f"{prefix}_lora.pt"
        if self.args.method == "head_only":
            return f"{prefix}_head.pt"
        raise ValueError(f"Unknown method: {self.args.method}")

    def _finalize_wandb(self):
        if self.wandb_run is None or self.wandb is None:
            return
        best_val_loss = None
        best_epoch = None
        if self.best_state is not None:
            best_val_loss = self.best_state.get("val_loss_full")
            best_epoch = self.best_state.get("epoch")
        self.wandb_run.summary["best_epoch"] = best_epoch
        self.wandb_run.summary["best_val_loss"] = best_val_loss
        self.wandb.finish()

    def _wandb_log(self, payload: Dict[str, object]):
        if self.wandb is not None:
            self.wandb.log(payload)

    def _wandb_summary_update(self, payload: Dict[str, object]):
        if self.wandb_run is not None:
            self.wandb_run.summary.update(payload)


def main():
    args = parse_args()
    args.lr_head = args.lr if args.lr_head is None else args.lr_head
    args.lr_backbone = args.lr if args.lr_backbone is None else args.lr_backbone
    args.lr_lora = args.lr if args.lr_lora is None else args.lr_lora

    trainer = LinearTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
