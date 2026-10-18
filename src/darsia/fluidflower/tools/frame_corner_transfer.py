"""Transfer marked rig-frame corners from a reference photograph to another
camera position by edge template matching.

A series may hold several camera positions (a bumped tripod between runs or
between cycles of one run, a different campaign). The user marks the inner
corners of the rig frame once, on a reference photograph; this tool finds the
same corners in the baseline of another camera position and reports, per
corner, the position, the normalised cross-correlation and whether the corner
was matched or only predicted from the other corners. The result is a
proposal for the user's visual check in the Geometry and facies tab; nothing
is adopted here.

Method: Sobel edge magnitude of both photographs; an "inward" template around
each reference corner (it extends into the frame, away from the photograph
border, because the inner corner structure lies inside); normalised
cross-correlation within a search window; a 4-point similarity fit; a second
pass with templates warped by that similarity around the predicted corners;
corners below ``ncc_min`` are replaced by the similarity prediction and
flagged. Validated on FF_RE (Claude, 18 Oct 2026): control runs with the same
camera position reproduce the reference corners within 1-2 px.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

__all__ = [
    "CornerMatch",
    "SimilarityFit",
    "TransferResult",
    "edge_magnitude",
    "fit_similarity",
    "transfer_corners",
]


@dataclass(frozen=True)
class CornerMatch:
    name: str
    row: float
    col: float
    ncc: float
    status: str  # "matched" or "predicted"
    reference_row: float
    reference_col: float

    @property
    def shift(self) -> tuple[float, float]:
        return (self.row - self.reference_row, self.col - self.reference_col)


@dataclass(frozen=True)
class SimilarityFit:
    scale: float
    rotation_deg: float
    translation_col: float
    translation_row: float
    residual_px: dict

    def apply(self, row: float, col: float) -> tuple[float, float]:
        th = math.radians(self.rotation_deg)
        x = self.scale * (math.cos(th) * col - math.sin(th) * row) + self.translation_col
        y = self.scale * (math.sin(th) * col + math.cos(th) * row) + self.translation_row
        return (y, x)


@dataclass(frozen=True)
class TransferResult:
    corners: dict
    fit: SimilarityFit

    def as_dict(self) -> dict:
        return {"corners": {k: asdict(v) for k, v in self.corners.items()}, "fit": asdict(self.fit)}


def edge_magnitude(gray, sigma: float = 1.5):
    """Sobel edge magnitude of a float32 grayscale image after a Gaussian blur."""
    g = cv2.GaussianBlur(np.asarray(gray, dtype=np.float32), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def fit_similarity(source_rc, target_rc) -> SimilarityFit:
    """Least-squares similarity (scale, rotation, translation) mapping source
    (row, col) points onto target points; residual per point."""
    names = list(source_rc)
    s = np.array([[source_rc[k][1], source_rc[k][0]] for k in names], float)  # x=col, y=row
    d = np.array([[target_rc[k][1], target_rc[k][0]] for k in names], float)
    ms, md = s.mean(0), d.mean(0)
    S, D = s - ms, d - md
    a = float((S[:, 0] * D[:, 0] + S[:, 1] * D[:, 1]).sum())
    b = float((S[:, 0] * D[:, 1] - S[:, 1] * D[:, 0]).sum())
    n = float((S**2).sum())
    scale = math.hypot(a, b) / n
    th = math.atan2(b, a)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * scale
    t = md - R @ ms
    pred = (R @ s.T).T + t
    res = {k: float(math.hypot(*(pred[i] - d[i]))) for i, k in enumerate(names)}
    return SimilarityFit(scale, math.degrees(th), float(t[0]), float(t[1]), res)


def _inward_extent(name_rc, centre_rc, inward: int, outward: int):
    """Row/col extents of the template around a corner: ``inward`` pixels
    towards the frame centre and ``outward`` pixels away from it."""
    r, c = name_rc
    cr, cc = centre_rc
    rows = (-outward, inward) if cr > r else (-inward, outward)
    cols = (-outward, inward) if cc > c else (-inward, outward)
    return rows, cols


def _match(edge, template, window) -> tuple[float, float, float, float]:
    """Best normalised cross-correlation of ``template`` inside ``window``
    (rs, re, cs, ce) of ``edge``; returns (row, col, ncc, second_peak) of the
    template's top-left corner with parabolic sub-pixel refinement."""
    rs, re_, cs, ce = window
    win = edge[rs:re_, cs:ce]
    if win.shape[0] < template.shape[0] or win.shape[1] < template.shape[1]:
        return (math.nan, math.nan, -1.0, -1.0)
    res = cv2.matchTemplate(win, template, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    c, r = loc

    def par(a, b, cc):
        d = a - 2 * b + cc
        return 0.0 if abs(d) < 1e-12 else 0.5 * (a - cc) / d

    dr = par(res[r - 1, c], res[r, c], res[r + 1, c]) if 0 < r < res.shape[0] - 1 else 0.0
    dc = par(res[r, c - 1], res[r, c], res[r, c + 1]) if 0 < c < res.shape[1] - 1 else 0.0
    rr, cc = np.ogrid[: res.shape[0], : res.shape[1]]
    res2 = res.copy()
    res2[(rr - r) ** 2 + (cc - c) ** 2 <= 15**2] = -1.0
    return (rs + r + dr, cs + c + dc, float(mx), float(res2.max()))


def transfer_corners(reference_edge, target_edge, corners: dict, *, inward: int = 230, outward: int = 30,
                     search: int = 90, refine_search: int = 28, ncc_min: float = 0.9) -> TransferResult:
    """Find the reference corners (name -> (row, col)) in the target edge image.

    ``reference_edge`` and ``target_edge`` are edge-magnitude images of the
    same size (see ``edge_magnitude``). Returns matched positions, the
    similarity fit and, for corners below ``ncc_min``, the fit's prediction
    with status "predicted".
    """
    ref = np.asarray(reference_edge, dtype=np.float32)
    tgt = np.asarray(target_edge, dtype=np.float32)
    if ref.shape != tgt.shape:
        raise ValueError("reference and target edge images must have the same shape")
    H, W = ref.shape
    centre = (float(np.mean([v[0] for v in corners.values()])), float(np.mean([v[1] for v in corners.values()])))

    def clamp_box(ri, ci, rows, cols):
        return (max(0, ri + rows[0]), min(H, ri + rows[1] + 1), max(0, ci + cols[0]), min(W, ci + cols[1] + 1))

    def pass_match(edge_src, predicted, radius):
        found = {}
        for name, (r0, c0) in corners.items():
            pr, pc = predicted[name]
            pri, pci = int(round(pr)), int(round(pc))
            rows, cols = _inward_extent((r0, c0), centre, inward, outward)
            tr0, tr1, tc0, tc1 = clamp_box(pri, pci, rows, cols)
            template = edge_src[tr0:tr1, tc0:tc1]
            off_r, off_c = pri - tr0, pci - tc0
            window = (max(0, tr0 - radius), min(H, tr1 + radius), max(0, tc0 - radius), min(W, tc1 + radius))
            r, c, ncc, second = _match(tgt, template, window)
            if math.isnan(r):
                found[name] = (pr, pc, -1.0)
                continue
            found[name] = (r + off_r + (pr - pri), c + off_c + (pc - pci), ncc)
        return found

    # pass 1: templates cut from the reference at the reference corners
    first = pass_match(ref, {k: v for k, v in corners.items()}, search)
    good = {k: (v[0], v[1]) for k, v in first.items() if v[2] >= ncc_min}
    use = good if len(good) >= 3 else {k: (v[0], v[1]) for k, v in first.items()}
    fit1 = fit_similarity({k: corners[k] for k in use}, use)
    # pass 2: templates cut from the reference warped by the first-pass similarity, around the predicted corners
    th = math.radians(fit1.rotation_deg)
    M = np.array([[fit1.scale * math.cos(th), -fit1.scale * math.sin(th), fit1.translation_col],
                  [fit1.scale * math.sin(th), fit1.scale * math.cos(th), fit1.translation_row]], np.float32)
    warped = cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    predicted = {k: fit1.apply(*corners[k]) for k in corners}
    second = pass_match(warped, predicted, refine_search)
    good2 = {k: (v[0], v[1]) for k, v in second.items() if v[2] >= ncc_min}
    use2 = good2 if len(good2) >= 3 else {k: (v[0], v[1]) for k, v in second.items()}
    fit2 = fit_similarity({k: corners[k] for k in use2}, use2)
    out = {}
    residual = {}
    for name, (r0, c0) in corners.items():
        r, c, ncc = second[name]
        pr, pc = fit2.apply(r0, c0)
        residual[name] = float(math.hypot(pr - r, pc - c))
        if ncc >= ncc_min:
            out[name] = CornerMatch(name, float(r), float(c), ncc, "matched", float(r0), float(c0))
        else:
            out[name] = CornerMatch(name, float(pr), float(pc), ncc, "predicted", float(r0), float(c0))
    fit = SimilarityFit(fit2.scale, fit2.rotation_deg, fit2.translation_col, fit2.translation_row, residual)
    return TransferResult(out, fit)
