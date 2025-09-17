import argparse
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from src.models.backbone_dinov3 import DinoV3Backbone, DinoV3PCam
from src.models.lora import LoRALinear, inject_lora
from src.train.data_utils import build_eval_loaders
from src.utils.eval_utils import evaluate, get_device, evaluate_loss

DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_IMAGE_SIZE = 224
CHECKPOINT_ROOT = "checkpoints/saved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a checkpoint and evaluate it on PCam validation/test splits."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (relative paths are resolved under checkpoints/saved).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="src/data/pcam",
        help="Directory containing PCam .h5 files.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional override for HuggingFace backbone model id.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Optional override for image resolution used at training/eval time.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=256,
        help="Batch size for validation/test loaders.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--tta_eval",
        action="store_true",
        help="Enable test-time augmentation during heavy metrics evaluation.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Optional limit on validation/test batches (0 = full dataset).",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        default="none",
        help="Placeholder pruning method name; currently not implemented.",
    )
    parser.add_argument(
        "--prune_amount",
        type=float,
        default=0.0,
        help="Placeholder pruning amount in [0,1]; currently not implemented.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for this evaluation run.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dinov3-pcam-compress",
        help="Weights & Biases project to log under.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity (team/user).",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=[None, "online", "offline"],
        help="Override WANDB_MODE environment variable.",
    )
    return parser.parse_args()


def resolve_checkpoint(path: str) -> str:
    if os.path.isabs(path):
        return path
    candidate = os.path.join(CHECKPOINT_ROOT, path)
    return candidate if os.path.exists(candidate) else path


def infer_checkpoint_type(path: str) -> str:
    name = os.path.basename(path).lower()
    if "lora" in name:
        return "lora"
    if "full" in name:
        return "fullft"
    if "head" in name:
        return "head_only"
    raise ValueError(
        f"Unable to infer checkpoint type from filename '{name}'. Expected 'head', 'lora', or 'full'."
    )


def extract_training_metadata(
    payload: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[str, int]:
    ckpt_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    ckpt_model_id = None
    ckpt_resolution = None
    if isinstance(ckpt_args, dict):
        ckpt_model_id = ckpt_args.get("model_id")
        ckpt_resolution = ckpt_args.get("resolution")
    backbone_model_id = (
        payload.get("backbone_model_id") if isinstance(payload, dict) else None
    )
    image_size = payload.get("image_size") if isinstance(payload, dict) else None

    model_id = args.model_id or backbone_model_id or ckpt_model_id or DEFAULT_MODEL_ID
    resolution = (
        args.resolution
        if args.resolution is not None
        else image_size
        if image_size is not None
        else ckpt_resolution
        if ckpt_resolution is not None
        else DEFAULT_IMAGE_SIZE
    )
    return model_id, int(resolution)  # type:ignore


def _apply_backbone_overrides(
    module: nn.Module, overrides: Dict[str, torch.Tensor]
) -> int:
    if not overrides:
        return 0
    sd = module.state_dict()
    matched = 0
    missing = []
    mismatched = []
    for k, v in overrides.items():
        if k in sd:
            if sd[k].shape == v.shape:
                sd[k] = v
                matched += 1
            else:
                mismatched.append((k, tuple(sd[k].shape), tuple(v.shape)))
        else:
            missing.append(k)
    module.load_state_dict(sd, strict=False)
    if missing:
        print(
            f"[warn] backbone_overrides: {len(missing)} keys not found (first 5): {missing[:5]}"
        )
    if mismatched:
        print(
            f"[warn] backbone_overrides: {len(mismatched)} shape mismatches (first 3): {mismatched[:3]}"
        )
    return matched


def load_model(
    checkpoint_path: str,
    ckpt_type: str,
    model_id: str,
    device: str,
    payload: Optional[Dict[str, object]] = None,
) -> Tuple[DinoV3PCam, Dict[str, object]]:
    if payload is None:
        payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload is not None

    backbone = DinoV3Backbone(model_id=model_id, dtype=torch.float32)
    model = DinoV3PCam(backbone)

    if ckpt_type == "fullft":
        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise KeyError("Full checkpoint missing 'state_dict'.")
        model.load_state_dict(state_dict)  # type:ignore

    elif ckpt_type == "head_only":
        head_state = payload.get("head")
        if head_state is None:
            raise KeyError("Head-only checkpoint missing 'head'.")
        model.head.load_state_dict(head_state)  # type:ignore

    elif ckpt_type == "lora":
        adapters = payload.get("adapters")
        head_state = payload.get("head")
        hparams = payload.get("lora_hparams", {})
        if adapters is None or head_state is None:
            raise KeyError("LoRA checkpoint missing adapters or head state.")
        targets = hparams.get("targets", "")  # type:ignore
        if isinstance(targets, str):
            target_keys = [s.strip() for s in targets.split(",") if s.strip()]
        else:
            target_keys = list(targets)
        if hparams.get("include_mlp"):  # type:ignore
            for extra in ("up_proj", "down_proj"):
                if extra not in target_keys:
                    target_keys.append(extra)
        inject_lora(
            model.backbone.model,
            target_keys=target_keys,
            r=hparams.get("r", 0),  # type:ignore
            alpha=hparams.get("alpha", 1),  # type:ignore
            dropout=hparams.get("dropout", 0.0),  # type:ignore
        )
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear) and module.r > 0:
                a_key = f"{name}.A"
                b_key = f"{name}.B"
                if a_key in adapters:  # type:ignore
                    module.A.data.copy_(adapters[a_key])  # type:ignore
                if b_key in adapters:  # type:ignore
                    module.B.data.copy_(adapters[b_key])  # type:ignore
        model.head.load_state_dict(head_state)  # type:ignore

    else:
        raise ValueError(f"Unsupported checkpoint type: {ckpt_type}")

    # warn if run trained norms/bias but no overrides are present
    ck_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    tn_mode = (
        ck_args.get("train_norms_bias", "none") if isinstance(ck_args, dict) else "none"
    )
    overrides = payload.get("backbone_overrides", {})

    if tn_mode != "none" and not overrides:
        print(
            "[warn] This checkpoint trained backbone LayerNorms/bias "
            f"('{tn_mode}') but contains no 'backbone_overrides'. "
            "Metrics may be lower than at training time."
        )

    # apply overrides if present
    if isinstance(overrides, dict) and overrides:
        n_applied = _apply_backbone_overrides(model.backbone.model, overrides)  # type: ignore
        print(f"Applied {n_applied} backbone override tensors (LayerNorm/bias).")

    model.eval()
    model.to(device)
    return model, payload


def maybe_apply_pruning(model: nn.Module, method: str, amount: float) -> None:
    method = method.lower()
    if method in {"none", ""}:
        return
    print(
        f"[placeholder] Requested pruning method '{method}' with amount {amount}. No pruning applied."
    )


def init_wandb(
    args: argparse.Namespace, metadata: Dict[str, object]
) -> Tuple[Optional[Any], Optional[Any]]:
    if not args.wandb:
        return None, None

    import wandb

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    tags = ["eval"]
    ckpt_tag = metadata.get("ckpt_type")
    if isinstance(ckpt_tag, str):
        tags.append(f"ckpt:{ckpt_tag}")

    config = {**vars(args), **metadata}
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="eval",
        config=config,
        tags=tags,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("test/*", step_metric="global_step")
    return wandb, run


def log_wandb_metrics(
    wandb_module: Optional[Any],
    wandb_run: Optional[Any],
    log_payload: Dict[str, float],
    summary_payload: Optional[Dict[str, float]] = None,
) -> None:
    if wandb_module is None:
        return
    wandb_module.log(log_payload)
    if wandb_run is not None and summary_payload:
        wandb_run.summary.update(summary_payload)


def finalize_wandb(wandb_module: Optional[Any]) -> None:
    if wandb_module is not None:
        wandb_module.finish()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    ckpt_type = infer_checkpoint_type(checkpoint_path)
    device = get_device()

    raw_payload = torch.load(checkpoint_path, map_location="cpu")
    model_id, image_size = extract_training_metadata(raw_payload, args)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Detected type: {ckpt_type}")
    print(f"Model ID: {model_id}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")

    wandb_metadata: Dict[str, object] = {
        "checkpoint_path": checkpoint_path,
        "ckpt_type": ckpt_type,
        "resolved_model_id": model_id,
        "resolved_image_size": image_size,
        "device": device,
    }
    wandb_module, wandb_run = init_wandb(args, wandb_metadata)

    try:
        val_loader, test_loader = build_eval_loaders(
            data_dir=args.data_dir,
            model_id=model_id,
            image_size=image_size,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        model, _ = load_model(
            checkpoint_path=checkpoint_path,
            ckpt_type=ckpt_type,
            model_id=model_id,
            device=device,
            payload=raw_payload,
        )

        maybe_apply_pruning(model, args.prune_method, args.prune_amount)

        criterion = nn.CrossEntropyLoss()
        eval_limit = max(0, int(args.max_eval_batches))

        print("Running validation loss evaluation...")
        val_loss = evaluate_loss(
            model, val_loader, device, criterion, max_batches=eval_limit
        )
        print(f"Validation loss: {val_loss:.4f}")

        print("Running test loss evaluation...")
        test_loss = evaluate_loss(
            model, test_loader, device, criterion, max_batches=eval_limit
        )
        print(f"Test loss: {test_loss:.4f}")

        print("Running full metric evaluation (validation)...")
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            max_batches=eval_limit,
            use_tta=args.tta_eval,
        )
        print("Validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        print("Running full metric evaluation (test)...")
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            max_batches=eval_limit,
            use_tta=args.tta_eval,
        )
        print("Test metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")

        wandb_log: Dict[str, float] = {
            "global_step": 0.0,
            "val/loss": float(val_loss),
            "test/loss": float(test_loss),
        }
        for key, value in val_metrics.items():
            wandb_log[f"val/{key}"] = float(value)
        for key, value in test_metrics.items():
            wandb_log[f"test/{key}"] = float(value)

        summary_payload = {k: v for k, v in wandb_log.items() if k != "global_step"}
        summary_payload.update(
            {
                "val/max_eval_batches": float(eval_limit),
                "test/max_eval_batches": float(eval_limit),
                "eval/tta_enabled": float(args.tta_eval),
            }
        )
        log_wandb_metrics(wandb_module, wandb_run, wandb_log, summary_payload)
    finally:
        finalize_wandb(wandb_module)


if __name__ == "__main__":
    main()
