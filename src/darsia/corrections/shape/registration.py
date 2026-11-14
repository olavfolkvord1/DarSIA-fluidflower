"""Projective registration of camera images to a run grid by matching local patches.

A reference image on the run grid (e.g. the setup's shape-corrected baseline) and a
camera image of the same rig are compared at a lattice of reference points. Each
point's patch, after a logarithm and a band-pass filter, is found in the camera
image by zero-mean normalized cross-correlation near its position under an
initial homography, with a sub-pixel peak. A homography from the run grid to the
camera image is then fitted to the matches by RANSAC with a refinement, checked for
folds, and verified at held-out zones whose patches are found near their predicted
positions.

All arithmetic runs in the floating precision of the given images (single precision
at least). The RANSAC sample sequence and refinement follow OpenCV's
``findHomography`` with ``RANSAC``, computed in that precision; its confidence,
iteration limits and stopping tolerance are settings.

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
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import scipy.linalg
import scipy.signal

import darsia


@dataclass(frozen=True)
class ProjectiveRegistrationSettings:
    """Settings of the projective registration; one configuration table.

    Positions and sizes are in pixels of the run grid, columns before rows.

    """

    fit_columns: tuple[int, int, int]
    """Start, stop and step of the columns of the reference points of the fit."""
    fit_rows: tuple[int, int, int]
    """Start, stop and step of the rows of the reference points of the fit."""
    template_half: int
    """Half width of the square reference patch of a fit point."""
    search_half: int
    """Largest displacement searched from a fit point's initial position."""
    min_correlation: float
    """Smallest correlation of a match of a fit point."""
    minimum_matches: int
    """Fewest matched fit points."""
    ransac_threshold: float
    """Largest distance of a RANSAC inlier from its predicted position."""
    ransac_confidence: float
    """Probability that some RANSAC sample holds only inliers, which sets the
    number of samples."""
    ransac_iterations: int
    """Largest number of RANSAC samples."""
    refinement_steps: int
    """Largest number of refinement steps of the RANSAC homography."""
    refinement_tolerance: float
    """The refinement stops when the largest change of an entry of the homography,
    or the largest forward error, is below this."""
    minimum_inlier_fraction: float
    """Smallest fraction of the matches that are inliers of the homography."""
    orientation_columns: list[int]
    """Columns of the points where the homography must not fold."""
    orientation_rows: list[int]
    """Rows of the points where the homography must not fold."""
    heldout_zones: list[list]
    """Check zones as [name, [first column, first row, end column, end row], step]."""
    heldout_template_half: int
    """Half width of the square reference patch of a check point."""
    heldout_search_half: int
    """Largest displacement searched from a check point's predicted position."""
    heldout_min_correlation: float
    """Smallest correlation of a match of a check point."""
    heldout_minimum_matches: int
    """Fewest matched check points in each zone."""
    sigma_small: float
    """Standard deviation of the Gaussian subtracted from, in the band-pass filter."""
    sigma_large: float
    """Standard deviation of the subtracted Gaussian, in the band-pass filter."""
    minimum_template_std: float
    """Reference patches with less spread of their filtered values are skipped."""
    log_offset: float
    """Offset added to the image values before the logarithm, in their units."""


@dataclass(frozen=True)
class ProjectiveRegistration:
    """Result of a projective registration."""

    homography: np.ndarray
    """Map from run-grid pixel centres (column, row, 1) to camera pixel centres."""
    matches: int
    """Matched fit points."""
    inlier_fraction: float
    """Fraction of the matched fit points that are inliers of the homography."""
    heldout: dict[str, np.ndarray]
    """Distances of the matched check points from their predicted positions, per
    zone."""


def projective_registration(
    reference: np.ndarray,
    image: np.ndarray,
    initial: np.ndarray,
    settings: ProjectiveRegistrationSettings,
) -> ProjectiveRegistration:
    """Register a camera image to the run grid of a reference image.

    Parameters
    ----------
    reference : array
        One channel of the reference image on the run grid, e.g. the green channel
        of the setup's shape-corrected baseline in camera units.
    image : array
        The same channel of the camera image, in the same units.
    initial : array
        Initial 3 x 3 homography from the run grid to the camera image, e.g. from
        the rig points (:func:`rig_points_homography`) for the setup's baseline, or
        the baseline's registration for a later frame.
    settings : ProjectiveRegistrationSettings
        Settings of the registration.

    Returns
    -------
    ProjectiveRegistration
        The homography from the run grid to the camera image and its quality.

    Raises
    ------
    ValueError
        If too few fit points or check points match, too few matches are inliers,
        or the homography folds the run grid.

    """
    dtype = np.result_type(reference.dtype, image.dtype, np.float32)
    a = _log_bandpass(reference, dtype, settings)
    b = _log_bandpass(image, dtype, settings)
    initial = np.asarray(initial, dtype)

    def guess(x, y):
        """Position of a reference point under the initial homography."""
        q = initial @ np.array([x, y, 1.0], dtype=dtype)
        return float(q[0] / q[2]), float(q[1] / q[2])

    points = (
        (x, y) for y in range(*settings.fit_rows) for x in range(*settings.fit_columns)
    )
    matches = _match(
        a,
        b,
        points,
        settings.template_half,
        settings.search_half,
        guess,
        settings.min_correlation,
        settings.minimum_template_std,
    )
    if len(matches) < settings.minimum_matches:
        raise ValueError(f"Only {len(matches)} fit points of the run grid match.")
    homography, inliers = ransac_homography(
        matches[:, :2],
        matches[:, 2:4],
        settings.ransac_threshold,
        confidence=settings.ransac_confidence,
        iterations=settings.ransac_iterations,
        refinement_steps=settings.refinement_steps,
        refinement_tolerance=settings.refinement_tolerance,
    )
    if (
        homography is None
        or inliers.mean(dtype=dtype) < settings.minimum_inlier_fraction
    ):
        raise ValueError("Too few matches are inliers of a homography.")
    if not np.all(np.isfinite(homography)):
        raise ValueError("The fitted homography is not finite.")

    # The local orientation at sampled points must not reverse or fold the grid.
    samples = np.array(
        [
            (x, y)
            for y in settings.orientation_rows
            for x in settings.orientation_columns
        ],
        dtype=dtype,
    )
    q = _project_points(samples, homography)
    qx = _project_points(samples + np.array([1, 0], dtype), homography)
    qy = _project_points(samples + np.array([0, 1], dtype), homography)
    jacobian = (qx[:, 0] - q[:, 0]) * (qy[:, 1] - q[:, 1]) - (qx[:, 1] - q[:, 1]) * (
        qy[:, 0] - q[:, 0]
    )
    # A finite homography can still send a sampled point to its horizon.
    if not np.isfinite(jacobian).all() or np.any(jacobian <= 0):
        raise ValueError("The fitted homography folds the run grid.")

    def predicted(x, y):
        """Position of a reference point under the fitted homography."""
        point = _project_points(np.array([[x, y]], dtype), homography)[0]
        return float(point[0]), float(point[1])

    heldout = {}
    for name, (x0, y0, x1, y1), step in settings.heldout_zones:
        checks = _match(
            a,
            b,
            ((x, y) for y in range(y0, y1, step) for x in range(x0, x1, step)),
            settings.heldout_template_half,
            settings.heldout_search_half,
            predicted,
            settings.heldout_min_correlation,
            settings.minimum_template_std,
        )
        if len(checks) < settings.heldout_minimum_matches:
            raise ValueError(f"Only {len(checks)} check points match in {name}.")
        prediction = _project_points(checks[:, :2].astype(dtype), homography)
        heldout[name] = np.linalg.norm(prediction - checks[:, 2:4], axis=1)
    return ProjectiveRegistration(
        homography, len(matches), float(inliers.mean(dtype=dtype)), heldout
    )


def rig_points_homography(
    corners: np.ndarray, shape: tuple[int, int], dtype: np.dtype | type
) -> np.ndarray:
    """Homography from the run grid to a camera image, given by four rig points.

    The run grid's corner pixel centres map onto the rig points, as in the crop of
    :class:`darsia.CurvatureCorrection`. The transform from the rig points to the
    corners is solved from the eight projective equations and inverted, in the
    given precision.

    Parameters
    ----------
    corners : array
        The rig points in the camera image as (row, column), in the order top left,
        bottom left, bottom right, top right.
    shape : tuple of int
        Rows and columns of the run grid.
    dtype : numpy dtype or type
        Floating precision of the calculation, e.g. the images' (the rig points
        are plain numbers).

    Returns
    -------
    array
        3 x 3 homography from run-grid pixel centres (column, row, 1) to camera
        pixel centres.

    """
    dtype = np.dtype(dtype)
    rows, cols = shape
    source = np.fliplr(np.asarray(corners)).astype(dtype)
    target = np.array(
        [[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]], dtype=dtype
    )
    xy1 = np.column_stack((source, np.ones(4, dtype=dtype)))
    system = np.zeros((8, 8), dtype=dtype)
    system[::2, :3], system[1::2, 3:6] = xy1, xy1
    system[::2, 6:] = -source * target[:, :1]
    system[1::2, 6:] = -source * target[:, 1:]
    with warnings.catch_warnings():
        # In single precision the system is ill-conditioned in pixel units; the
        # homography only centres whole-pixel searches.
        warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
        values = darsia.linalg.solve(system, target.ravel())
    to_grid = np.append(values, dtype.type(1)).reshape(3, 3)
    return darsia.linalg.solve(to_grid, np.eye(3, dtype=dtype))


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


def _project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map (x, y) rows through a homography in the points' precision."""
    x, y = _project(np.asarray(matrix, points.dtype), points[:, 0], points[:, 1])
    return np.column_stack((x, y))


def _log_bandpass(
    image: np.ndarray, dtype, settings: ProjectiveRegistrationSettings
) -> np.ndarray:
    """Difference of two Gaussians of the image's logarithm, in the dtype."""
    values = np.log(np.asarray(image, dtype) + dtype.type(settings.log_offset))
    return darsia.gaussian_blur(values, settings.sigma_small) - darsia.gaussian_blur(
        values, settings.sigma_large
    )


def _correlation(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalized cross-correlation at every placement of a template."""
    dtype = search.dtype
    height, width = template.shape
    centered = template - template.mean(dtype=dtype)
    energy = np.sum(centered * centered, dtype=dtype)
    shape = (search.shape[0] - height + 1, search.shape[1] - width + 1)
    if energy == 0:
        return np.ones(shape, dtype=dtype)

    def window_sums(values):
        """Sums over every placement of the template, from cumulative sums."""
        summed = np.pad(
            values.cumsum(axis=0, dtype=dtype).cumsum(axis=1, dtype=dtype),
            ((1, 0), (1, 0)),
        )
        return (
            summed[height:, width:]
            - summed[:-height, width:]
            - summed[height:, :-width]
            + summed[:-height, :-width]
        )

    sums = window_sums(search)
    variance = np.maximum(
        window_sums(search * search) - sums * sums / dtype.type(template.size), 0
    )
    numerator = scipy.signal.correlate(search, centered, mode="valid", method="fft")
    denominator = np.sqrt(variance * energy)
    scores = np.zeros(shape, dtype=dtype)
    np.divide(numerator, denominator, out=scores, where=denominator > 0)
    return np.clip(scores, -1, 1)


def _match(
    a: np.ndarray,
    b: np.ndarray,
    points,
    template_half: int,
    search_half: int,
    guess: Callable,
    min_correlation: float,
    minimum_template_std: float,
) -> np.ndarray:
    """Find reference patches in an image; keep interior, concave correlation peaks.

    Returns rows (x, y, matched x, matched y, correlation) in the images' precision.

    """
    dtype = a.dtype
    rows = []
    h, s = template_half, search_half
    for x, y in points:
        x, y = int(x), int(y)
        if x - h < 0 or y - h < 0 or x + h + 1 > a.shape[1] or y + h + 1 > a.shape[0]:
            continue
        qx, qy = guess(x, y)
        qx, qy = int(round(qx)), int(round(qy))
        if (
            qx - h - s < 0
            or qy - h - s < 0
            or qx + h + s + 1 > b.shape[1]
            or qy + h + s + 1 > b.shape[0]
        ):
            continue
        template = a[y - h : y + h + 1, x - h : x + h + 1]
        if float(template.std()) < minimum_template_std:
            continue
        search = b[qy - h - s : qy + h + s + 1, qx - h - s : qx + h + s + 1]
        response = _correlation(search, template)
        _, score, _, (px, py) = cv2.minMaxLoc(response)
        score = dtype.type(score)
        if (
            score < min_correlation
            or px in (0, response.shape[1] - 1)
            or py in (0, response.shape[0] - 1)
        ):
            continue
        # Sub-pixel peak of a parabola through the peak and its neighbours.
        cx = response[py, px - 1] + response[py, px + 1] - 2 * score
        cy = response[py - 1, px] + response[py + 1, px] - 2 * score
        if cx >= 0 or cy >= 0:
            continue
        dx = 0.5 * (response[py, px - 1] - response[py, px + 1]) / cx
        dy = 0.5 * (response[py - 1, px] - response[py + 1, px]) / cy
        rows.append((x, y, qx - s + px + dx, qy - s + py + dy, score))
    return np.asarray(rows, dtype=dtype).reshape(-1, 5)


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
