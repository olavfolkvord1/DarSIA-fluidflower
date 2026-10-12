"""Opt-in learned aqueous completion, using the variant-B optical context.

The optical/physical parent owns history. Completion only adds aqueous pixels to
this frame's final map; it never seeds the parent or changes an existing label.
PyTorch is imported only when building or evaluating the learned stage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

VERSION = "aqueous-context-completion-v1"
PROFILE_FILE = "phase_aqueous_context_completion.json"
PROFILE_KEY = "aqueous_context_completion"
WORKING_SPACING_M = .001
PATCH = 128
STRIDE = 96


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_profile(root, inherited):
    """Load a portable sidecar without modifying the inherited profile file."""
    root = Path(root).resolve()
    path = root / PROFILE_FILE
    if not path.is_file():
        return inherited
    if inherited is None or PROFILE_KEY in inherited:
        raise ValueError("Aqueous completion requires one unwrapped inherited physics profile")
    specification = json.loads(path.read_text(encoding="utf-8"))
    if set(specification) != {"format", "checkpoint", "checkpoint_sha256"}:
        raise ValueError("Unknown aqueous completion profile fields")
    if specification["format"] != VERSION:
        raise ValueError("Unsupported aqueous completion format")
    checkpoint = (root / specification["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
        raise ValueError("Aqueous completion checkpoint must exist inside the model bundle")
    if _sha(checkpoint) != specification["checkpoint_sha256"]:
        raise ValueError("Aqueous completion checkpoint hash mismatch")
    return {**inherited, PROFILE_KEY: {**specification, "_checkpoint_path": str(checkpoint)}}


def split_profile(profile):
    specification = profile[PROFILE_KEY]
    if specification.get("format") != VERSION:
        raise ValueError("Unsupported aqueous completion format")
    return {k: v for k, v in profile.items() if k != PROFILE_KEY}, specification


def runtime_identity(inherited_identity, specification):
    # Absolute paths change when a verified bundle is copied to another machine.
    public = {k: v for k, v in specification.items() if not k.startswith("_")}
    payload = {"inherited_runtime": inherited_identity, "completion": public,
               "source_sha256": _sha(__file__)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_network(torch):
    """State-dict compatible with the pilot's reference_context variant."""
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, incoming, outgoing):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1), nn.GroupNorm(4, outgoing), nn.SiLU(),
                nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.GroupNorm(4, outgoing), nn.SiLU())

        def forward(self, value):
            return self.layers(value)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1, self.encoder2 = Block(12, 8), Block(8, 16)
            self.bottleneck, self.decoder2, self.decoder1 = Block(16, 32), Block(48, 16), Block(24, 8)
            self.heads = nn.Conv2d(8, 1, 1)
            nn.init.zeros_(self.heads.weight)
            nn.init.zeros_(self.heads.bias)

        def forward(self, value):
            height, width = value.shape[-2:]
            value = nn.functional.pad(value, (0, (-width) % 4, 0, (-height) % 4), mode="replicate")
            first = self.encoder1(value)
            second = self.encoder2(nn.functional.max_pool2d(first, 2))
            value = self.bottleneck(nn.functional.max_pool2d(second, 2))
            value = nn.functional.interpolate(value, size=second.shape[-2:], mode="nearest")
            value = self.decoder2(torch.cat((value, second), dim=1))
            value = nn.functional.interpolate(value, size=first.shape[-2:], mode="nearest")
            return self.heads(self.decoder1(torch.cat((value, first), dim=1)))[..., :height, :width]

    return Network()


def prepared_inputs(rgb, raw, co2, gas, domain):
    """Shared prepared-array contract; RGB has current then baseline channels."""
    rgb = np.asarray(rgb)
    domain = np.asarray(domain, np.float32)
    if rgb.shape != (*domain.shape, 6) or rgb.dtype != np.uint8:
        raise ValueError("Prepared current/baseline RGB must be registered uint8 six-channel data")
    values = rgb.astype(np.float32).transpose(2, 0, 1) / 255
    fields = [np.asarray(a, np.float32) for a in (raw, co2, gas)]
    if any(a.shape != domain.shape or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()
           for a in [domain, *fields]):
        raise ValueError("Prepared phase inputs must share a finite unit-interval grid")
    delta = np.stack([cv2.GaussianBlur(values[i] - values[i+3], (0, 0), 3,
                     borderType=cv2.BORDER_REFLECT_101) for i in range(3)])
    return np.concatenate([values, *(a[None] for a in fields), delta]), domain


def prepare_inputs(current_rgb, baseline_rgb, raw_co2, delivered_labels, domain, spacing_m):
    """Pool native inputs exactly as variant B, including pilot quantization."""
    domain = np.asarray(domain, bool)
    if baseline_rgb is None:
        raise ValueError("Aqueous completion requires the registered pre-injection baseline image")
    for image in (current_rgb, baseline_rgb):
        if np.asarray(image).shape != (*domain.shape, 3) or np.asarray(image).dtype != np.uint8:
            raise ValueError("Aqueous completion needs native registered uint8 current and baseline RGB")
    labels = np.asarray(delivered_labels)
    raw = np.asarray(raw_co2, np.float32)
    if labels.shape != domain.shape or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Delivered labels must use the registered water=0, aqueous=1, gas=2 grid")
    if raw.shape != domain.shape or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError("Raw CO2 probabilities must share the registered grid and lie in [0,1]")
    dy, dx = map(float, spacing_m)
    if not np.isfinite([dy, dx]).all() or min(dy, dx) <= 0:
        raise ValueError("Positive finite native pixel spacing is required")
    h, w = domain.shape
    target = (max(1, round(w * dx / WORKING_SPACING_M)), max(1, round(h * dy / WORKING_SPACING_M)))
    resize = lambda a: cv2.resize(a, target, interpolation=cv2.INTER_AREA)
    rgb = np.concatenate([resize(a) for a in (current_rgb, baseline_rgb)], axis=-1)
    # These float16 casts reproduce the frozen pilot NPZ representation, rather
    # than silently changing a threshold-sensitive input between fit and use.
    raw, co2, gas, dom = [resize(a).astype(np.float16) for a in
        (raw, (labels > 0).astype(np.float32), (labels == 2).astype(np.float32), domain.astype(np.float32))]
    return prepared_inputs(rgb, raw, co2, gas, dom)


def base_probability(x):
    return .2 + .6 * (x[:, 7] - x[:, 8]).clamp(0, 1)


def predict_probability(model, x):
    import torch
    return torch.sigmoid(torch.logit(base_probability(x)) + model(x)[:, 0])


def tile_locations(height, width):
    starts = lambda n: sorted(set(range(0, max(n-PATCH, 0)+1, STRIDE)) | {max(0, n-PATCH)})
    return [(y, x) for y in starts(height) for x in starts(width)]


def tile_window():
    ramp = np.maximum(np.hanning(PATCH).astype(np.float32), .05)
    return ramp[:, None] * ramp[None, :]


def infer_probability(model, x, *, device=None, batch_size=8):
    """Full-frame probability, using globally aligned 128/96 Hann patches."""
    import torch
    if device is None:
        device = next(model.parameters()).device
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    if x.ndim != 3 or x.shape[0] != 12 or min(x.shape[1:]) < 1 or batch_size < 1:
        raise ValueError("Expected nonempty 12-channel completion inputs and positive batch size")
    _, h, w = x.shape
    locations, window = tile_locations(h, w), tile_window()
    padded = torch.nn.functional.pad(x, (0, max(0, PATCH-w), 0, max(0, PATCH-h)), mode="replicate")
    numerator, mass = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(locations), batch_size):
            group = locations[start:start+batch_size]
            batch = torch.stack([padded[:, y:y+PATCH, z:z+PATCH] for y, z in group])
            values = predict_probability(model, batch).cpu().numpy()
            for (y, z), q in zip(group, values):
                hh, ww = min(PATCH, h-y), min(PATCH, w-z)
                numerator[y:y+hh, z:z+ww] += q[:hh, :ww] * window[:hh, :ww]
                mass[y:y+hh, z:z+ww] += window[:hh, :ww]
    if not np.all(mass > 0):
        raise ValueError("Incomplete aqueous completion frame inference")
    return numerator / mass


def delivered_labels(predictions, domain):
    co2 = np.asarray(predictions["co2_presence_probability"])
    gas = np.asarray(predictions["gas_probability_conditional_on_co2"])
    return np.where(domain & (co2 >= .5), np.where(gas >= .6, 2, 1), 0).astype(np.uint8)


def apply_probability(predictions, probability, domain):
    """Preserve the native parent fields everywhere except new aqueous pixels."""
    domain = np.asarray(domain, bool)
    q = np.asarray(probability, np.float32)
    if q.shape != domain.shape or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("Completion probabilities must lie in [0,1] on the native domain grid")
    co2 = np.asarray(predictions["co2_presence_probability"], np.float32)
    gas = np.asarray(predictions["gas_probability_conditional_on_co2"], np.float32)
    if co2.shape != domain.shape or gas.shape != domain.shape:
        raise ValueError("Completion and inherited probabilities must share the native grid")
    added = domain & (co2 < .5) & (q >= .5)
    new_co2, new_gas = co2.copy(), gas.copy()
    new_co2[added], new_gas[added] = q[added], 0.
    result = {**predictions, "co2_presence_probability": new_co2,
              "gas_probability_conditional_on_co2": new_gas}
    for key, fallback, updated in (
        ("aqueous_co2_probability", co2 * (1-gas), new_co2 * (1-new_gas)),
        ("free_co2_gas_probability", co2 * gas, new_co2 * new_gas),
        ("untouched_water_probability", 1-co2, 1-new_co2)):
        result[key] = np.array(predictions.get(key, fallback), copy=True)
        result[key][added] = updated[added]
    return result, added


class _Predictor:
    """Immutable fitted network; disposable reference histories share it."""
    def __init__(self, specification):
        import torch
        checkpoint = Path(specification["_checkpoint_path"])
        if _sha(checkpoint) != specification["checkpoint_sha256"]:
            raise ValueError("Aqueous completion checkpoint hash mismatch before inference")
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data.get("contextual") is not True:
            raise ValueError("Expected variant-B contextual aqueous completion checkpoint")
        self.model = build_network(torch)
        self.model.load_state_dict(data["state_dict"], strict=True)
        self.model.requires_grad_(False).eval()
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                self.model.to("cuda")
                self.device = "cuda"
            except RuntimeError:
                self.model.to("cpu")

    def __deepcopy__(self, memo):
        return self

    def __call__(self, inputs):
        import torch
        try:
            return infer_probability(self.model, inputs, device=self.device)
        except torch.cuda.OutOfMemoryError:
            self.model.to("cpu")
            self.device = "cpu"
            torch.cuda.empty_cache()
            return infer_probability(self.model, inputs, device="cpu")


class AqueousCompletionSequence:
    def __init__(self, inherited, specification):
        self.inherited, self.context, self.specification = inherited, inherited.context, specification
        self.calibration_baseline = None
        self._predictor = None
        self.last_diagnostics = {}

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "inherited"), name)

    def bind_reference(self, baseline_rgb):
        if self.calibration_baseline is not None:
            raise ValueError("Aqueous completion baseline is immutable")
        if baseline_rgb is None or np.asarray(baseline_rgb).shape != (*self.context.domain.shape, 3):
            raise ValueError("Aqueous completion requires the registered pre-injection baseline image")
        if np.asarray(baseline_rgb).dtype != np.uint8:
            raise ValueError("Aqueous completion baseline must be native corrected uint8 RGB")
        if hasattr(self.inherited, "bind_reference"):
            self.inherited.bind_reference(baseline_rgb)
        self.calibration_baseline = np.array(baseline_rgb, copy=True)

    def apply(self, predictions, elapsed_h):
        raise ValueError("Aqueous completion requires apply_with_reference and current corrected RGB")

    def apply_with_reference(self, predictions, elapsed_h, current_rgb):
        if self.calibration_baseline is None:
            raise ValueError("Bind the registered pre-injection baseline before aqueous completion")
        if hasattr(self.inherited, "apply_with_reference"):
            output, report = self.inherited.apply_with_reference(predictions, elapsed_h, current_rgb)
        else:
            output, report = self.inherited.apply(predictions, elapsed_h)
        negative = bool(report.get("pre_injection_negative_control", False))
        before = delivered_labels(output, self.context.domain)
        added = np.zeros_like(self.context.domain)
        grid_shape = None
        if not negative:
            spacing = (abs(float(self.context.y_m[1]-self.context.y_m[0])),
                       abs(float(self.context.x_m[1]-self.context.x_m[0])))
            inputs, _ = prepare_inputs(current_rgb, self.calibration_baseline,
                predictions["co2_presence_probability"], before, self.context.domain, spacing)
            if self._predictor is None:
                self._predictor = _Predictor(self.specification)
            coarse = self._predictor(inputs)
            grid_shape = list(coarse.shape)
            q = cv2.resize(coarse, self.context.domain.shape[::-1], interpolation=cv2.INTER_LINEAR)
            output, added = apply_probability(output, q, self.context.domain)
        self.last_diagnostics = {**self.inherited.last_diagnostics,
            "completion_before_co2": before > 0, "completion_before_gas": before == 2,
            "completion_added_aqueous": added}
        return output, {**report, PROFILE_KEY: {
            "format": VERSION, "checkpoint_sha256": self.specification["checkpoint_sha256"],
            "bypassed_pre_injection": negative, "added_aqueous_pixels": int(added.sum()),
            "removed_aqueous_pixels": 0, "gas_labels_unchanged": True,
            "working_grid_shape": grid_shape, "history_updates": False,
            "calibrated_probability": False}}

    def write_diagnostics(self, *args, **kwargs):
        # Reuse the standard writer directly: do not put completion outputs into
        # the inherited sequence's diagnostic/history state.
        from .phase_presence_context import PresenceSequence
        return PresenceSequence.write_diagnostics(self, *args, **kwargs)
