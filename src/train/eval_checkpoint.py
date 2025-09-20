import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from src.models.backbone_dinov3 import DinoV3Backbone, DinoV3PCam
from src.models.lora import LoRALinear, inject_lora
from src.train.data_utils import build_eval_loaders
from src.utils.eval_utils import evaluate, get_device, evaluate_loss

DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_IMAGE_SIZE = 224
CHECKPOINT_ROOT = "checkpoints/saved"

KNOWN_SVD_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
]


def _parse_prune_targets(raw: str) -> List[str]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return []
    lowered = [t.lower() for t in tokens]
    if any(t in {"all", "*"} for t in lowered):
        return list(KNOWN_SVD_TARGETS)

    unknown = sorted({t for t in tokens if t not in KNOWN_SVD_TARGETS})
    if unknown:
        joined = ", ".join(unknown)
        print(
            f"[warn] Unrecognized prune targets requested ({joined}); they will be used as substring filters."
        )
    return tokens


def _resolve_parent_module(
    root: nn.Module, qualified_name: str
) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _count_parameters(module: nn.Module) -> int:
    """Return the total number of parameters in a module."""
    return sum(param.numel() for param in module.parameters())


@torch.no_grad()
def _prune_attention_heads(
    module: nn.Module, energy_threshold: float
) -> Tuple[int, int, int, List[Tuple[str, int, int]]]:
    """Prune low-energy attention heads based on the output projection.

    For every submodule that exposes ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj``
    linears along with a ``num_heads`` attribute (e.g. ``DINOv3ViTAttention``), heads
    are ranked by the L2 energy of the corresponding ``o_proj`` slices. The minimum
    number of heads whose cumulative squared energy exceeds ``energy_threshold`` is
    kept, with the remaining heads removed across all projections.

    Returns the tuple ``(blocks_pruned, original_params, new_params, details)`` where
    ``details`` lists ``(module_name, previous_heads, kept_heads)`` for the pruned
    attention blocks.
    """

    changed = 0
    orig_params = 0
    new_params = 0
    info: List[Tuple[str, int, int]] = []

    thr = min(max(float(energy_threshold), 1e-6), 1.0)

    for name, parent in module.named_modules():
        has_projections = all(
            hasattr(parent, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        if not (has_projections and hasattr(parent, "num_heads")):
            continue

        q, k, v, o = parent.q_proj, parent.k_proj, parent.v_proj, parent.o_proj
        if not all(isinstance(x, nn.Linear) for x in (q, k, v, o)):
            continue
        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)
        assert isinstance(o, nn.Linear)

        d_model = q.out_features
        num_heads = int(getattr(parent, "num_heads"))
        if num_heads <= 1:
            continue

        head_dim = d_model // num_heads
        if head_dim * num_heads != d_model:
            continue

        # Rank heads by the energy (squared norm) of the corresponding o_proj slice.
        wo = o.weight.detach()  # (d_model, d_model)
        head_scores = (
            wo[:, : num_heads * head_dim]
            .reshape(d_model, num_heads, head_dim)
            .norm(dim=(0, 2))
        )  # (num_heads,)

        energy = head_scores.square()
        total_energy = float(torch.sum(energy))
        if total_energy <= 0.0:
            continue

        order = torch.argsort(head_scores, descending=True)
        cum = torch.cumsum(energy[order], dim=0) / total_energy
        keep_heads = int(torch.searchsorted(cum, thr).item()) + 1
        keep_heads = max(1, min(keep_heads, num_heads))

        if keep_heads >= num_heads:
            continue

        keep = torch.sort(order[:keep_heads]).values.tolist()

        def _expand_indices(indices: List[int]) -> List[int]:
            expanded: List[int] = []
            for h in indices:
                start = h * head_dim
                expanded.extend(range(start, start + head_dim))
            return expanded

        rows = _expand_indices(keep)  # selection for q/k/v output rows
        cols = _expand_indices(keep)  # selection for o input columns

        device = q.weight.device
        dtype = q.weight.dtype

        new_q = nn.Linear(
            q.in_features,
            keep_heads * head_dim,
            bias=q.bias is not None,
            device=device,
            dtype=dtype,
        )
        new_k = nn.Linear(
            k.in_features,
            keep_heads * head_dim,
            bias=k.bias is not None,
            device=device,
            dtype=dtype,
        )
        new_v = nn.Linear(
            v.in_features,
            keep_heads * head_dim,
            bias=v.bias is not None,
            device=device,
            dtype=dtype,
        )
        new_o = nn.Linear(
            keep_heads * head_dim,
            o.out_features,
            bias=o.bias is not None,
            device=device,
            dtype=dtype,
        )

        new_q.weight.copy_(q.weight.detach()[rows, :])
        if new_q.bias is not None and q.bias is not None:
            new_q.bias.copy_(q.bias.detach()[rows])

        new_k.weight.copy_(k.weight.detach()[rows, :])
        if new_k.bias is not None and k.bias is not None:
            new_k.bias.copy_(k.bias.detach()[rows])

        new_v.weight.copy_(v.weight.detach()[rows, :])
        if new_v.bias is not None and v.bias is not None:
            new_v.bias.copy_(v.bias.detach()[rows])

        new_o.weight.copy_(o.weight.detach()[:, cols])
        if new_o.bias is not None and o.bias is not None:
            new_o.bias.copy_(o.bias.detach())

        new_q.weight.requires_grad = q.weight.requires_grad
        if new_q.bias is not None and q.bias is not None:
            new_q.bias.requires_grad = q.bias.requires_grad
        new_k.weight.requires_grad = k.weight.requires_grad
        if new_k.bias is not None and k.bias is not None:
            new_k.bias.requires_grad = k.bias.requires_grad
        new_v.weight.requires_grad = v.weight.requires_grad
        if new_v.bias is not None and v.bias is not None:
            new_v.bias.requires_grad = v.bias.requires_grad
        new_o.weight.requires_grad = o.weight.requires_grad
        if new_o.bias is not None and o.bias is not None:
            new_o.bias.requires_grad = o.bias.requires_grad

        parent.q_proj = new_q
        parent.k_proj = new_k
        parent.v_proj = new_v
        parent.o_proj = new_o
        setattr(parent, "num_heads", int(keep_heads))
        if hasattr(parent, "num_attention_heads"):
            setattr(parent, "num_attention_heads", int(keep_heads))
        if hasattr(parent, "head_dim"):
            setattr(parent, "head_dim", int(head_dim))

        changed += 1
        block_orig = sum(_count_parameters(layer) for layer in (q, k, v, o))
        block_new = sum(
            _count_parameters(layer) for layer in (new_q, new_k, new_v, new_o)
        )
        orig_params += block_orig
        new_params += block_new
        info.append((name, num_heads, keep_heads))

    return changed, orig_params, new_params, info


@torch.no_grad()
def _prune_mlp_neurons(
    module: nn.Module, energy_threshold: float
) -> Tuple[int, int, int, List[Tuple[str, int, int]]]:
    changed = 0
    orig = 0
    new = 0
    details: List[Tuple[str, int, int]] = []
    for name, parent in module.named_modules():
        if not (hasattr(parent, "up_proj") and hasattr(parent, "down_proj")):
            continue
        up, down = parent.up_proj, parent.down_proj
        if not (isinstance(up, nn.Linear) and isinstance(down, nn.Linear)):
            continue
        if up.out_features != down.in_features:
            continue

        Wup = up.weight.detach()  # (r, d)
        Wdown = down.weight.detach()  # (d, r)
        r = up.out_features
        if r <= 1:
            continue

        up_norm = torch.linalg.norm(Wup, dim=1)  # (r,)
        down_norm = torch.linalg.norm(Wdown, dim=0)  # (r,)
        scores = up_norm * down_norm
        energy = scores**2
        keep_order = torch.argsort(scores, descending=True)
        cum = torch.cumsum(energy[keep_order], dim=0) / torch.sum(energy)
        keep_r = (
            int(
                torch.searchsorted(
                    cum, min(max(float(energy_threshold), 1e-6), 1.0)
                ).item()
            )
            + 1
        )
        keep_r = max(1, min(keep_r, r))
        if keep_r >= r:
            continue

        idx = torch.sort(keep_order[:keep_r]).values
        dev, dt = up.weight.device, up.weight.dtype

        # Build new layers on the same device/dtype as the originals before copying
        new_up = nn.Linear(up.in_features, keep_r, bias=(up.bias is not None)).to(
            device=dev, dtype=dt
        )
        new_up.weight.copy_(Wup[idx, :])
        if up.bias is not None:
            new_up.bias.copy_(up.bias.detach()[idx])

        new_down = nn.Linear(
            keep_r, down.out_features, bias=(down.bias is not None)
        ).to(device=dev, dtype=dt)
        new_down.weight.copy_(Wdown[:, idx])
        if down.bias is not None:
            new_down.bias.copy_(down.bias.detach())

        # Match the original gradient settings
        new_up.weight.requires_grad = up.weight.requires_grad
        if new_up.bias is not None and up.bias is not None:
            new_up.bias.requires_grad = up.bias.requires_grad
        new_down.weight.requires_grad = down.weight.requires_grad
        if new_down.bias is not None and down.bias is not None:
            new_down.bias.requires_grad = down.bias.requires_grad

        parent.up_proj = new_up
        parent.down_proj = new_down

        changed += 1
        o = (
            Wup.numel()
            + Wdown.numel()
            + (up.bias.numel() if up.bias is not None else 0)
            + (down.bias.numel() if down.bias is not None else 0)
        )
        n = (
            new_up.weight.numel()
            + new_down.weight.numel()
            + (new_up.bias.numel() if new_up.bias is not None else 0)
            + (new_down.bias.numel() if new_down.bias is not None else 0)
        )
        orig += o
        new += n
        details.append((name, r, keep_r))
    return changed, orig, new, details


@torch.no_grad()
def merge_lora_adapters(module: nn.Module) -> int:
    to_replace: List[Tuple[str, LoRALinear]] = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, LoRALinear):
            submodule.merge()
            to_replace.append((name, submodule))

    for name, lora_mod in to_replace:
        parent, leaf = _resolve_parent_module(module, name)
        setattr(parent, leaf, lora_mod.base)
    if to_replace:
        print(f"Merged {len(to_replace)} LoRA adapters into their base linear layers.")
    return len(to_replace)


@torch.no_grad()
def _truncated_svd_linear(
    linear: nn.Linear, energy_threshold: float
) -> Optional[Tuple[nn.Module, int, int, int]]:
    weight = linear.weight.detach()
    out_features, in_features = weight.shape
    min_dim = min(out_features, in_features)
    if min_dim == 0:
        return None

    svd_dtype = torch.float32
    weight_cpu = weight.to(dtype=svd_dtype, device="cpu")
    try:
        U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
    except RuntimeError as exc:  # pragma: no cover - protective guard
        print(f"[warn] SVD failed for layer ({exc}); skipping compression.")
        return None

    squared = S**2
    total_energy = torch.sum(squared)
    if total_energy.item() == 0:
        return None

    cumulative = torch.cumsum(squared, dim=0) / total_energy
    clamped = min(max(float(energy_threshold), 1e-6), 1.0)
    keep_rank = int(torch.searchsorted(cumulative, clamped).item()) + 1
    keep_rank = max(1, min(keep_rank, min_dim))
    if keep_rank >= min_dim:
        return None

    bias_params = linear.bias.numel() if linear.bias is not None else 0
    original_params = weight.numel() + bias_params
    new_params_estimate = keep_rank * (in_features + out_features) + bias_params
    if new_params_estimate >= original_params:
        return None

    first = nn.Linear(in_features, keep_rank, bias=False)
    second = nn.Linear(keep_rank, out_features, bias=linear.bias is not None)

    first.weight.copy_(Vh[:keep_rank, :])
    second.weight.copy_(U[:, :keep_rank] * S[:keep_rank])
    if linear.bias is not None:
        second.bias.copy_(linear.bias.detach())

    device = linear.weight.device
    dtype = linear.weight.dtype
    first = first.to(device=device, dtype=dtype)
    second = second.to(device=device, dtype=dtype)
    for param in first.parameters():
        param.requires_grad = False
    for param in second.parameters():
        param.requires_grad = False

    new_module = nn.Sequential(first, second)
    new_params = sum(p.numel() for p in new_module.parameters())
    return new_module, keep_rank, original_params, new_params


@torch.no_grad()
def _apply_truncated_svd_to_model(
    module: nn.Module, target_keys: List[str], energy_threshold: float
) -> Tuple[int, int, int, List[Tuple[str, int, int, int]]]:
    if not target_keys:
        return 0, 0, 0, []

    candidates: List[Tuple[str, nn.Linear]] = []
    for name, submodule in module.named_modules():
        if not name:
            continue
        if isinstance(submodule, nn.Linear) and any(key in name for key in target_keys):
            candidates.append((name, submodule))

    replaced: List[Tuple[str, int, int, int]] = []
    original_params = 0
    compressed_params = 0

    for name, linear in candidates:
        result = _truncated_svd_linear(linear, energy_threshold)
        if result is None:
            continue
        new_module, rank, old_params, new_params = result
        parent, leaf = _resolve_parent_module(module, name)
        setattr(parent, leaf, new_module)
        replaced.append((name, linear.in_features, linear.out_features, rank))
        original_params += old_params
        compressed_params += new_params

    return len(replaced), original_params, compressed_params, replaced


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
        help=(
            "Pruning method to apply (supported: none, truncated_svd, attention_heads, "
            "mlp_neurons)."
        ),
    )
    parser.add_argument(
        "--prune_amount",
        type=float,
        default=0.95,
        help=(
            "Energy threshold for energy-based pruning (fraction of squared norm to keep)."
        ),
    )
    parser.add_argument(
        "--prune_targets",
        type=str,
        default=",".join(KNOWN_SVD_TARGETS),
        help=(
            "Comma-separated substrings for linear layer names to compress. Known "
            "options: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj. Use 'all' or '*' to "
            "select all known targets. Only used with truncated SVD pruning."
        ),
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
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="eval_run",
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


def apply_pruning(
    model: nn.Module, method: str, energy_threshold: float, target_spec: str
) -> None:
    method = method.strip().lower()
    if method in {"none", ""}:
        return

    threshold = float(energy_threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError("Energy threshold must be in (0, 1].")

    if method in {"truncated_svd", "svd", "tsvd"}:
        targets = _parse_prune_targets(target_spec)
        if not targets:
            print("[warn] No prune targets resolved; skipping pruning.")
            return

        replaced, original_params, compressed_params, details = (
            _apply_truncated_svd_to_model(model, targets, threshold)
        )

        if replaced == 0:
            print(
                "[info] Truncated SVD did not modify any layers (check target substrings or threshold)."
            )
            return

        reduction = 1.0 - (compressed_params / max(1, original_params))
        print(
            f"Applied truncated SVD to {replaced} layers; params {original_params} → {compressed_params} ({reduction:.1%} reduction)."
        )
        for name, in_feats, out_feats, rank in details[:5]:
            print(f"  - {name}: ({out_feats}×{in_feats}) → rank {rank} factorization")
        if len(details) > 5:
            print(f"    … {len(details) - 5} additional layers compressed")
        return

    if method in {"attention_heads", "attn_heads", "attention", "head_prune"}:
        changed, original_params, compressed_params, details = _prune_attention_heads(
            model, threshold
        )
        if changed == 0:
            print(
                "[info] Attention head pruning did not modify any blocks (try lowering the threshold)."
            )
            return

        reduction = 1.0 - (compressed_params / max(1, original_params))
        print(
            f"Pruned attention heads in {changed} blocks; params {original_params} → {compressed_params} ({reduction:.1%} reduction)."
        )
        for name, prev_heads, kept_heads in details[:5]:
            print(f"  - {name}: heads {prev_heads} → {kept_heads}")
        if len(details) > 5:
            print(f"    … {len(details) - 5} additional attention blocks pruned")
        return

    if method in {"mlp", "mlp_neurons", "mlp-prune", "mlp-neurons"}:
        changed, original_params, compressed_params, details = _prune_mlp_neurons(
            model, threshold
        )
        if changed == 0:
            print(
                "[info] MLP neuron pruning did not modify any layers (try lowering the threshold)."
            )
            return

        reduction = 1.0 - (compressed_params / max(1, original_params))
        print(
            f"Applied MLP neuron pruning to {changed} blocks; params {original_params} → {compressed_params} ({reduction:.1%} reduction)."
        )
        for name, prev_width, kept_width in details[:5]:
            print(f"  - {name}: width {prev_width} → {kept_width}")
        if len(details) > 5:
            print(f"    … {len(details) - 5} additional MLP blocks pruned")
        return

    raise ValueError(f"Unsupported pruning method '{method}'.")


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
        name=args.wandb_run_name,
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
        "prune_method": args.prune_method,
        "prune_amount": args.prune_amount,
        "prune_targets": args.prune_targets,
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

        merge_lora_adapters(model)
        params_before = _count_parameters(model)
        print(f"Model parameters before pruning: {params_before:,}")

        apply_pruning(
            model,
            args.prune_method,
            args.prune_amount,
            args.prune_targets,
        )

        params_after = _count_parameters(model)
        params_diff = params_before - params_after
        params_percent_diff = (params_before - params_after) / max(1, params_before)
        print(f"Model parameters after pruning: {params_after:,}")
        print(f"Parameter difference (before - after): {params_diff:+,}")
        print(f"Parameter reduction: {params_percent_diff:.1%}")

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

        wandb_log.update(
            {
                "model/params_before": float(params_before),
                "model/params_after": float(params_after),
                "model/params_diff": float(params_diff),
                "model/params_pct_reduction": float(params_percent_diff),
            }
        )

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
