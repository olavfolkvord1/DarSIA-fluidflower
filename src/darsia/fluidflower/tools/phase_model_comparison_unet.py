"""Bounded, baseline-aware U-Net candidate for phase-model comparisons.

Inputs are fixed-scale RGB in [0, 1], current image followed by its registered
pre-injection image. Targets are *direct pixel labels* for CO2 presence and gas
within CO2. A target of -1 is unknown. Regional presence or majority annotations
must not be converted into dense targets by this backend.

PyTorch is optional: importing this module requires only NumPy. Validation, when
provided as (inputs, targets, weights), is reported once after fitting; it never
selects epochs, hyperparameters, or checkpoints.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


FORMAT_VERSION = "phase-comparison-small-unet-v1"
INPUT_CHANNELS = (
    "current_red", "current_green", "current_blue",
    "preinjection_red", "preinjection_green", "preinjection_blue",
)
HEAD_NAMES = ("co2_presence", "gas_within_co2")
PROFILES = {
    "smoke": {"epochs": 1, "max_steps": 8},
    "pilot": {"epochs": 15, "max_steps": 4096},
}
MAX_BATCH_PIXELS = 262_144


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "The U-Net comparison backend requires optional dependency PyTorch. "
            "Install a PyTorch build appropriate for the intended CPU/CUDA device."
        ) from exc


def _device(torch: Any, device: str) -> Any:
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported for this comparison.")
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch runtime has no available CUDA device.")
        if result.index is not None and result.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device}")
    return result


@contextmanager
def _runtime(torch: Any, device: Any, seed: int | None, num_threads: int):
    """Bound CPU workers and restore process settings/RNG after each operation."""
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    old_threads = torch.get_num_threads()
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cuda_ids = []
    if device.type == "cuda":
        cuda_ids = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        torch.set_num_threads(num_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=cuda_ids):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if cuda_ids:
                    with torch.cuda.device(cuda_ids[0]):
                        torch.cuda.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32


def _build_model(torch: Any, base_channels: int = 8) -> Any:
    if base_channels not in {8, 16}:
        raise ValueError("base_channels must be 8 or 16 for this bounded candidate.")
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, incoming: int, outgoing: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1),
                nn.GroupNorm(4, outgoing),
                nn.SiLU(),
                nn.Conv2d(outgoing, outgoing, 3, padding=1),
                nn.GroupNorm(4, outgoing),
                nn.SiLU(),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_channels = base_channels
            self.encoder1 = Block(6, base_channels)
            self.encoder2 = Block(base_channels, 2 * base_channels)
            self.bottleneck = Block(2 * base_channels, 4 * base_channels)
            self.decoder2 = Block(6 * base_channels, 2 * base_channels)
            self.decoder1 = Block(3 * base_channels, base_channels)
            self.heads = nn.Conv2d(base_channels, 2, 1)

        def forward(self, value: Any) -> Any:
            height, width = value.shape[-2:]
            pad_h, pad_w = (-height) % 4, (-width) % 4
            value = torch.nn.functional.pad(value, (0, pad_w, 0, pad_h), mode="replicate")
            skip1 = self.encoder1(value)
            skip2 = self.encoder2(torch.nn.functional.max_pool2d(skip1, 2))
            value = self.bottleneck(torch.nn.functional.max_pool2d(skip2, 2))
            value = torch.nn.functional.interpolate(value, size=skip2.shape[-2:], mode="nearest")
            value = self.decoder2(torch.cat((value, skip2), dim=1))
            value = torch.nn.functional.interpolate(value, size=skip1.shape[-2:], mode="nearest")
            value = self.decoder1(torch.cat((value, skip1), dim=1))
            return self.heads(value)[..., :height, :width]

    return SmallUNet()


def _inputs(inputs: Any) -> np.ndarray:
    array = np.asarray(inputs, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 6 or min(array.shape[-2:]) < 4:
        raise ValueError("inputs must have shape (N, 6, H, W), with H and W at least 4.")
    if not np.isfinite(array).all() or (array.size and (array.min() < 0 or array.max() > 1)):
        raise ValueError("inputs must be finite RGB values in [0, 1]; no learned normalization is applied.")
    if int(array.shape[-2]) * int(array.shape[-1]) > MAX_BATCH_PIXELS:
        raise ValueError(f"A patch exceeds the bounded {MAX_BATCH_PIXELS}-pixel resource limit.")
    return array


def _training_arrays(inputs: Any, targets: Any, weights: Any):
    x = _inputs(inputs)
    y = np.asarray(targets, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    expected = (len(x), 2, *x.shape[-2:])
    if y.shape != expected or w.shape != expected:
        raise ValueError(f"targets and weights must both have shape {expected}.")
    if not np.isfinite(y).all() or not np.isin(y, [-1.0, 0.0, 1.0]).all():
        raise ValueError("targets must contain only -1 (unknown), 0, or 1 direct pixel labels.")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative.")
    return x, y, w


def _batch_size(array: np.ndarray, requested: int) -> int:
    if requested < 1:
        raise ValueError("batch_size must be positive.")
    return min(int(requested), max(1, MAX_BATCH_PIXELS // math.prod(array.shape[-2:])))


def _masked_bce_with_logits(
    logits: Any, targets: Any, weights: Any, *, normalization_weight: float | None = None,
) -> Any:
    """Weighted direct-pixel loss with an optional fixed training-population normalizer.

    By default this is the weighted mean over this array. Mini-batch training
    instead supplies the expected batch mass from the complete training set, so
    batches do not each erase their relative run/region weighting.
    """
    torch = _torch()
    valid = (targets >= 0) & (weights > 0)
    effective = torch.where(valid, weights, torch.zeros_like(weights))
    if normalization_weight is not None:
        if not math.isfinite(normalization_weight) or normalization_weight <= 0:
            raise ValueError("normalization_weight must be positive and finite.")
        denominator = normalization_weight
    else:
        denominator = effective.sum()
    if normalization_weight is None and denominator.item() == 0:
        return logits.sum() * 0.0
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    return (losses * effective).sum() / denominator


def _evaluate_loss(torch: Any, model: Any, arrays: Any, device: Any, batch_size: int) -> dict:
    x, y, w = arrays
    numerator, denominator = 0.0, 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = start + batch_size
            tx = torch.as_tensor(np.ascontiguousarray(x[start:stop]), device=device)
            ty = torch.as_tensor(np.ascontiguousarray(y[start:stop]), device=device)
            tw = torch.as_tensor(np.ascontiguousarray(w[start:stop]), device=device)
            mass = float(torch.where(ty >= 0, tw, torch.zeros_like(tw)).sum().item())
            if mass:
                loss = _masked_bce_with_logits(model(tx), ty, tw)
                numerator += float(loss.item()) * mass
                denominator += mass
    return {"weighted_bce": numerator / denominator if denominator else None,
            "direct_weight_sum": denominator, "patches": len(x)}


def fit_unet(
    inputs: Any, targets: Any, weights: Any, *, seed: int = 20260911,
    device: str = "cpu", profile: str = "smoke", validation: Any = None,
    epochs: int | None = None, max_steps: int | None = None,
    batch_size: int = 2, learning_rate: float = 1e-3, base_channels: int = 8,
    num_threads: int = 4,
) -> tuple[Any, dict]:
    """Fit a fixed-budget candidate on direct pixel labels, never regional guesses."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {tuple(PROFILES)}.")
    epochs = PROFILES[profile]["epochs"] if epochs is None else int(epochs)
    max_steps = PROFILES[profile]["max_steps"] if max_steps is None else int(max_steps)
    if epochs < 1 or max_steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, max_steps, and learning_rate must be positive.")
    x, y, w = _training_arrays(inputs, targets, weights)
    active = (y >= 0) & (w > 0)
    train_indices = np.flatnonzero(active.reshape(len(x), -1).any(axis=1)) if len(x) else np.array([], dtype=int)
    if not len(train_indices):
        raise ValueError("No positive-weight direct pixel labels are available; regional boxes cannot replace them.")
    total_training_weight = float(np.where(active, w, 0).sum(dtype=np.float64))
    validation_arrays = None
    if validation is not None:
        if not isinstance(validation, (tuple, list)) or len(validation) != 3:
            raise ValueError("validation must be (inputs, targets, weights).")
        validation_arrays = _training_arrays(*validation)
    effective_batch = _batch_size(x, batch_size)
    torch = _torch()
    resolved_device = _device(torch, device)
    model = None
    optimizer = None
    steps = 0
    completed_epochs = 0
    losses = []
    rng = np.random.default_rng(seed)
    try:
        with _runtime(torch, resolved_device, seed, num_threads):
            model = _build_model(torch, base_channels).to(resolved_device)
            initial_parameters = [parameter.detach().cpu().clone() for parameter in model.parameters()]
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            for _ in range(epochs):
                order = rng.permutation(train_indices)
                model.train()
                completed_this_epoch = True
                for start in range(0, len(order), effective_batch):
                    if steps >= max_steps:
                        completed_this_epoch = False
                        break
                    indices = order[start:start + effective_batch]
                    tx = torch.as_tensor(np.ascontiguousarray(x[indices]), device=resolved_device)
                    ty = torch.as_tensor(np.ascontiguousarray(y[indices]), device=resolved_device)
                    tw = torch.as_tensor(np.ascontiguousarray(w[indices]), device=resolved_device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _masked_bce_with_logits(
                        model(tx), ty, tw,
                        normalization_weight=total_training_weight * len(indices) / len(train_indices),
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("U-Net training produced a non-finite masked loss.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    losses.append(float(loss.detach().cpu().item()))
                    del tx, ty, tw, loss
                if completed_this_epoch:
                    completed_epochs += 1
                if steps >= max_steps:
                    break
            model.eval()
            update_l2 = math.sqrt(sum(float(torch.sum((after.detach().cpu() - before) ** 2).item())
                                      for before, after in zip(initial_parameters, model.parameters())))
            metadata = {
                "format_version": FORMAT_VERSION,
                "input_channels": list(INPUT_CHANNELS), "head_names": list(HEAD_NAMES),
                "normalization": "fixed RGB [0,1]; no fitted statistics",
                "architecture": {"name": "small_unet_3level", "base_channels": base_channels},
                "seed": int(seed), "device": str(resolved_device), "torch_version": str(torch.__version__),
                "profile": profile, "epochs_requested": epochs, "epochs_completed": completed_epochs,
                "max_steps": max_steps, "steps": steps, "batch_size": effective_batch,
                "learning_rate": learning_rate, "num_threads": num_threads,
                "train_patches": len(x), "labelled_train_patches": len(train_indices),
                "direct_pixels_by_head": {head: int(active[:, index].sum()) for index, head in enumerate(HEAD_NAMES)},
                "direct_weight_sum": total_training_weight,
                "loss": "global positive-weight direct-pixel mean binary cross entropy",
                "minibatch_normalization": "total training weight * actual batch patches / informative training patches",
                "minibatch_sampling": "uniform permutation of informative training patches each epoch",
                "loss_per_update": losses, "parameter_update_l2": update_l2,
                "stop_reason": "max_steps" if steps >= max_steps else "epochs_complete",
                "convergence_assessed": False,
                "regional_presence_or_majority_densified": False,
                "validation_used_for_selection": False,
                "determinism": "seeded NumPy/torch; cuDNN deterministic, benchmark disabled; hardware-dependent arithmetic",
                "precision": "float32; CUDA TF32 disabled during training and prediction",
            }
            if validation_arrays is not None:
                metadata["validation"] = _evaluate_loss(
                    torch, model, validation_arrays, resolved_device,
                    _batch_size(validation_arrays[0], batch_size),
                )
            model.comparison_metadata = metadata
        return model, metadata
    except Exception:
        if model is not None:
            model.cpu()
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()


def predict_unet(model: Any, inputs: Any, *, batch_size: int = 2, num_threads: int = 4) -> np.ndarray:
    """Return independent sigmoid heads, shape (N, 2, H, W), on host memory."""
    torch = _torch()
    x = _inputs(inputs)
    effective_batch = _batch_size(x, batch_size)
    device = next(model.parameters()).device
    output = np.empty((len(x), 2, *x.shape[-2:]), dtype=np.float32)
    was_training = model.training
    try:
        with _runtime(torch, device, None, num_threads), torch.inference_mode():
            model.eval()
            for start in range(0, len(x), effective_batch):
                stop = min(start + effective_batch, len(x))
                tensor = torch.as_tensor(np.ascontiguousarray(x[start:stop]), device=device)
                prediction = torch.sigmoid(model(tensor))
                output[start:stop] = prediction.cpu().numpy()
                del tensor, prediction
        return output
    finally:
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def save_unet(model: Any, path: str | Path, metadata: dict | None = None) -> Path:
    """Atomically persist tensor weights plus JSON-compatible contract/provenance."""
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_metadata = json.loads(json.dumps(metadata if metadata is not None else model.comparison_metadata))
    payload = {
        "format_version": FORMAT_VERSION, "input_channels": list(INPUT_CHANNELS),
        "head_names": list(HEAD_NAMES), "base_channels": model.base_channels,
        "metadata": safe_metadata,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_unet(path: str | Path, *, device: str = "cpu") -> tuple[Any, dict]:
    """Load only safe tensor/primitives payloads, enforcing feature and head order."""
    torch = _torch()
    resolved_device = _device(torch, device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported U-Net comparison artifact format.")
    if payload.get("input_channels") != list(INPUT_CHANNELS) or payload.get("head_names") != list(HEAD_NAMES):
        raise ValueError("U-Net artifact input channel or output head contract does not match.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("U-Net artifact metadata must be a dictionary.")
    model = _build_model(torch, int(payload["base_channels"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolved_device).eval()
    model.comparison_metadata = metadata
    return model, metadata
