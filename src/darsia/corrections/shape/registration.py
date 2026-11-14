"""Projective registration of camera images to a run grid.

A homography is fitted to point matches by RANSAC with a refinement. All arithmetic
runs in the floating precision of the given points (single precision at least). The
RANSAC sample sequence and refinement follow OpenCV's ``findHomography`` with
``RANSAC``, computed in that precision; its confidence, iteration limits and
stopping tolerance are the caller's settings.

"""

# The RANSAC sampling and refinement of the homography follow the OpenCV 4.13
# implementation.
# Copyright (C) 2000, Intel Corporation, all rights reserved.
# Copyright (C) 2013, OpenCV Foundation, all rights reserved.
# Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
# Copyright (C) 2009, Willow Garage Inc., all rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   * Redistribution's of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistribution's in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * The name of the copyright holders may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall the Intel Corporation or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
# Copyright (c) 2007, Miroslav Balda
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import itertools
from typing import Optional

import numpy as np
import scipy.linalg


def ransac_homography(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
    *,
    confidence: float,
    iterations: int,
    refinement_steps: int,
    refinement_tolerance: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit a homography to point matches by four-point RANSAC and refinement.

    The samples come from OpenCV's random number sequence; a sample is used if no
    three of its points are collinear and its orientations agree. Each model is a
    normalized direct linear fit, voted on by the forward error, and a better model
    lowers the number of samples to the count that reaches the confidence. The best
    model's inliers are fitted again and refined by damped Gauss-Newton steps on
    their forward errors. All of it runs in the points' floating precision (single
    precision at least).

    Parameters
    ----------
    source, target : array
        Matched points (x, y), one row per match.
    threshold : float
        Largest forward distance of an inlier.
    confidence : float
        Probability that some sample holds only inliers, which sets the number of
        samples (OpenCV: 0.995).
    iterations : int
        Largest number of samples (OpenCV: 2000).
    refinement_steps : int
        Largest number of refinement steps (OpenCV: 10).
    refinement_tolerance : float
        The refinement stops when the largest change of an entry of the homography,
        or the largest forward error, is below it (OpenCV: the single-precision
        machine epsilon).

    Returns
    -------
    array or None
        The 3 x 3 homography mapping the source onto the target points, normalized
        to a last entry of one; None if there are fewer than four matches or no
        sample is consistent.
    array or None
        Inlier flags of the matches (uint8, one column).

    """
    dtype = np.result_type(source.dtype, target.dtype, np.float32)
    source, target = np.asarray(source, dtype), np.asarray(target, dtype)
    count = len(source)
    if count < 4:
        return None, None
    if count == 4:
        return _normalized_fit(source, target), np.ones((4, 1), np.uint8)

    def inliers(h):
        """Matches within the threshold of their forward projection."""
        x, y = _project(h, source[:, 0], source[:, 1])
        return (x - target[:, 0]) ** 2 + (y - target[:, 1]) ** 2 <= dtype.type(
            threshold
        ) ** 2

    best = np.zeros(count, bool)
    state, limit, iteration = (1 << 64) - 1, iterations, 0
    while iteration < limit:
        sample = _viable_sample(source, target, state)
        if sample is None:
            break
        sample, state = sample
        mask = inliers(_normalized_fit(source[sample], target[sample]))
        size = np.count_nonzero(mask)
        if size >= 4 and size > np.count_nonzero(best):
            best = mask
            success = (dtype.type(size) / dtype.type(count)) ** 4
            if success == 1:
                break
            needed = np.log(dtype.type(1) - dtype.type(confidence)) / np.log1p(-success)
            limit = min(limit, round(float(needed)))
        iteration += 1
    if not best.any():
        return None, None
    h = _refine(
        _normalized_fit(source[best], target[best]),
        source[best],
        target[best],
        refinement_steps,
        refinement_tolerance,
    )
    return h, inliers(h).astype(np.uint8).reshape(-1, 1)


# ! ---- Auxiliary routines


def _project(matrix: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Map pixel centres through a homography, without homogeneous arrays."""
    den = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / den,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / den,
        )


def _viable_sample(source: np.ndarray, target: np.ndarray, state: int):
    """Next four-point sample of OpenCV's sequence with no collinear triple.

    Returns the sample and the generator state, or None after 10,000 rejections.

    """
    count = len(source)
    for _ in range(10000):
        sample = []
        while len(sample) < 4:
            state = (state & 0xFFFFFFFF) * 4164903690 + (state >> 32)
            index = (state & 0xFFFFFFFF) % count
            if index not in sample:
                sample.append(index)
        if _viable(source[sample], target[sample]):
            return sample, state
    return None


def _viable(source: np.ndarray, target: np.ndarray) -> bool:
    """No collinear triple, and the same orientation of all triples in both sets."""
    signs = []
    for a, b, c in itertools.combinations(range(4), 3):
        areas = []
        for points in (source, target):
            u, v = points[b] - points[a], points[c] - points[a]
            area = u[0] * v[1] - u[1] * v[0]
            if abs(area) <= np.finfo(points.dtype).eps * (
                np.abs(u).sum() + np.abs(v).sum()
            ):
                return False
            areas.append(area)
        signs.append(np.signbit(areas[0]) == np.signbit(areas[1]))
    return all(signs) or not any(signs)


def _normalized_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Direct linear fit with mean-absolute-deviation normalization of each axis."""
    dtype = source.dtype

    def normalize(points):
        """Centred and scaled points, with the centre and the scales."""
        center = points.mean(axis=0)
        span = np.abs(points - center).mean(axis=0)
        if np.any(span == 0):
            raise ValueError("Collinear registration points.")
        return (points - center) / span, center, span

    src, sc, ss = normalize(source)
    dst, dc, ds = normalize(target)
    xy1 = np.column_stack((src, np.ones(len(src), dtype)))
    rows = np.zeros((len(src), 2, 9), dtype)
    rows[:, 0, :3] = rows[:, 1, 3:6] = xy1
    rows[:, 0, 6:] = -dst[:, :1] * xy1
    rows[:, 1, 6:] = -dst[:, 1:] * xy1
    matrix = rows.reshape(-1, 9)
    _, vectors = scipy.linalg.eigh(
        matrix.T @ matrix, subset_by_index=(0, 0), check_finite=False
    )
    denormalize = np.diag(np.array([ds[0], ds[1], 1], dtype=dtype))
    denormalize[:2, 2] = dc
    normalize_src = np.diag(np.array([1 / ss[0], 1 / ss[1], 1], dtype=dtype))
    normalize_src[:2, 2] = -sc / ss
    transform = denormalize @ vectors[:, 0].reshape(3, 3) @ normalize_src
    return transform / transform[2, 2]


def _refine(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    steps: int,
    tolerance: float,
):
    """Damped Gauss-Newton steps on the forward errors of the inliers."""
    dtype = source.dtype
    eps = np.finfo(dtype).eps
    points = np.column_stack((source, np.ones(len(source), dtype)))

    def errors(h, jacobian=False):
        """Forward residuals, and optionally their derivative by the entries."""
        projected = points @ h.T
        den = projected[:, 2:]
        inverse = np.divide(
            dtype.type(1), den, out=np.zeros_like(den), where=np.abs(den) > eps
        )
        xy = projected[:, :2] * inverse
        residual = (xy - target).ravel()
        if not jacobian:
            return residual
        derivative = np.zeros((len(points), 2, 9), dtype)
        scaled = points * inverse
        derivative[:, 0, :3] = derivative[:, 1, 3:6] = scaled
        derivative[:, :, 6:] = -xy[:, :, None] * scaled[:, None, :]
        return residual, derivative.reshape(-1, 9)

    def inverse_normal(matrix):
        """Pseudo-inverse of a symmetric matrix from its eigenvalues."""
        values, vectors = scipy.linalg.eigh(matrix, check_finite=False)
        keep = np.abs(values) > eps * np.abs(values).sum()
        return (vectors[:, keep] / values[keep]) @ vectors[:, keep].T

    residual, jac = errors(transform, True)
    normal, gradient = jac.T @ jac, jac.T @ residual
    penalty = np.diag(np.diag(normal))
    damping, transition = dtype.type(1), dtype.type(0.75)
    cost = residual @ residual
    for _ in range(steps):
        change = inverse_normal(normal + damping * penalty) @ gradient
        proposed = transform - change.reshape(transform.shape)
        trial = errors(proposed)
        trial_cost = trial @ trial
        predicted = change @ (2 * gradient - normal @ change)
        ratio = (cost - trial_cost) / (
            predicted if abs(predicted) > eps else dtype.type(1)
        )
        if ratio > 0.75:
            damping *= dtype.type(0.5)
            if damping < transition:
                damping = dtype.type(0)
        elif ratio < 0.25:
            slope = change @ gradient
            factor = np.clip(
                (trial_cost - cost) / (slope if abs(slope) > eps else dtype.type(1))
                + 2,
                2,
                10,
            )
            if damping == 0:
                transition = dtype.type(1) / max(
                    eps, np.max(np.abs(np.diag(inverse_normal(normal))))
                )
                damping = transition
                factor *= dtype.type(0.5)
            damping *= factor
        if trial_cost < cost:
            transform, cost = proposed, trial_cost
            residual, jac = errors(transform, True)
            normal, gradient = jac.T @ jac, jac.T @ residual
        if max(abs(change)) < tolerance or max(abs(residual)) < tolerance:
            break
    return transform / transform[2, 2]
