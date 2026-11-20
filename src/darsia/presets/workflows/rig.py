# TODO: Any connections to darsia.SimpleFluidFlower?
# TODO: integrate into darsia4porotwin.FluidFlower?
# TODO: Clean up concentration analyses. Can be removed?

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia
from darsia.presets.analysis.porosity import patched_porosity_analysis
from darsia.presets.workflows.config.corrections import (
    CorrectionsConfig,
    FrameRegistrationConfig,
    IlluminationCorrectionConfig,
    MaterialIlluminationCorrectionConfig,
    PatchwiseIlluminationCorrectionConfig,
)
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.facies_props import FaciesProps
from darsia.presets.workflows.setup.illustrations import save_scalar_map_illustration
from darsia.utils import results

if TYPE_CHECKING:
    from darsia.presets.workflows.config.roi_registry import RoiRegistry

logger = logging.getLogger(__name__)


def _correction_file(kind: str, index: int, correction) -> str:
    """The file name of a rig's ``index``-th ``kind`` correction ("shape" or
    "color"), as the rig saves and records it."""
    return f"{kind}_correction_{index}_{type(correction).__name__.lower()}.npz"


def _listed(folder: Path, prefix: str) -> list[Path]:
    """The ``.npz`` files of a folder whose names start with ``prefix``, sorted by
    name; a listing of a share that is out raises its error, never an empty one."""
    with os.scandir(folder) as entries:
        names = [
            entry.name
            for entry in entries
            if entry.name.startswith(prefix)
            and entry.name.endswith(".npz")
            and entry.is_file()
        ]
    return [Path(folder) / name for name in sorted(names)]


def _resolve_named_roi_voxels(
    roi_key: str | None,
    roi_registry: "RoiRegistry | None",
    image: darsia.Image,
    as_landmarks: bool = False,
) -> np.ndarray | None:
    """Resolve a named ROI registry key to voxel coordinates for use as a
    shape/color correction's 'roi' config value, as an alternative to
    auto-detecting a color checker via :func:`darsia.find_colorchecker`.

    Parameters
    ----------
    roi_key : str, optional
        Registry key to resolve, or None to skip (caller falls back to
        auto-detection).
    roi_registry : RoiRegistry, optional
        Registry to resolve `roi_key` against.
    image : darsia.Image
        Image providing the coordinate system for the physical-to-voxel
        conversion (the registry stores ROIs in physical coordinates).
    as_landmarks : bool
        If True, return the box as 4 ordered corner voxels, anti-clockwise
        from the top-left (row-min, col-min) corner - the convention
        :class:`darsia.ColorCorrection` expects for its reference (brown
        swatch) corner. If False, return the box corners/polygon vertices
        directly, which is sufficient for :class:`darsia.DriftCorrection`
        (it collapses any input to a bounding box anyway).

    Returns
    -------
    np.ndarray or None
        Voxel coordinates (matrix-indexed), or None if no key/registry given.
    """
    if roi_key is None or roi_registry is None:
        return None
    roi_config = roi_registry.resolve_rois([roi_key])[roi_key]
    # RoiConfig.roi itself raises for shapes with no plain geometric
    # representation (label/facies/intersection/union) - re-raise with
    # context specific to this call site (which of those two config
    # fields was involved, and that those shapes have no well-defined
    # bounding box at this point in setup, before any labels/facies image
    # exists).
    try:
        geometry = roi_config.roi
    except ValueError as exc:
        raise ValueError(
            f"ROI '{roi_key}' has shape={roi_config.shape!r}, but "
            "corrections.color.roi/corrections.drift.roi only support "
            "shape='box'/'polygon' (label/facies/intersection/union ROIs "
            "have no well-defined bounding box at this point in setup)."
        ) from exc
    # 'box' gives 2 corners, 'polygon' gives >= 3 vertices - both reduce to a
    # bounding box the same way below, so no shape-specific branching needed.
    voxels = image.coordinatesystem.voxel(geometry)
    if not as_landmarks:
        return voxels
    return darsia.bounding_box_inverse(darsia.bounding_box(voxels))


# TODO keep log? or use results?


class Rig:
    """Rig object for CO2 analysis."""

    grid_to_rig: list[list[float]] | None = None
    """The run's measured model of its grid, the homography from the grid's pixels
    to rig metres, by which each pixel is placed on the rig
    (:func:`darsia.grid_positions`); None for a grid in rig metres."""

    measurement_canvas: darsia.MeasurementCanvas | None = None
    """The canvas the run's pixels were measured on, in the grid's pixels, with
    the model (e.g. a padded canvas around the grid); None for the grid itself."""

    @property
    def corrections(self) -> list[darsia.BaseCorrection]:
        """Combined correction workflow in execution order."""
        return getattr(self, "shape_corrections", []) + getattr(
            self, "color_corrections", []
        )

    @staticmethod
    def _is_shape_correction(correction: object) -> bool:
        return isinstance(
            correction,
            (
                darsia.TypeCorrection,
                darsia.ResizeCorrection,
                darsia.DriftCorrection,
                darsia.CurvatureCorrection,
                darsia.ImageMappingCorrection,
                darsia.MetricScalingCorrection,
            ),
        )

    @staticmethod
    def _is_color_correction(correction: object) -> bool:
        return isinstance(
            correction,
            (
                darsia.ColorCorrection,
                darsia.RelativeColorCorrection,
                darsia.IlluminationCorrection,
                darsia.PatchwiseIlluminationCorrection,
                darsia.MaterialIlluminationCorrection,
            ),
        )

    def setup_reading(
        self,
        baseline_path: Path,
        experiment: darsia.ProtocolledExperiment,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
        log: Path | None = None,
        show_plot: bool = False,
    ):
        # Cache experiment for protocol-based datetime lookup
        self.experiment = experiment

        self.raw_decoding: dict = {}
        """Settings of the raw decode of the photographs; empty if not raw."""
        if corrections_config is not None and corrections_config.raw is not None:
            # Decode in the rig's precision, chosen once (float64 without one).
            precision = corrections_config.precision or np.float64
            self.raw_decoding = {
                "demosaic": corrections_config.raw.demosaic,
                "dtype": np.dtype(precision).name,
            }

        self.registration: darsia.FrameRegistration | None = None
        """The setup's reference of each photograph's registration, if any."""
        if corrections_config is not None and corrections_config.registration:
            # The baseline registered as a frame, on the reference's grid.
            pre_baseline = self.setup_registration(baseline_path, corrections_config)
        else:
            # Setup (based on baseline image without any corrections applied)
            pre_baseline = darsia.imread(baseline_path, **self.raw_decoding)

        if show_plot:
            # Allows to visually check the baseline image before any corrections
            # are applied, which is useful for debugging and understanding the
            # impact of corrections.
            plt.imshow(pre_baseline.img)
            plt.show()

        self.setup_shape_corrections(
            pre_baseline=pre_baseline,
            corrections_config=corrections_config,
            roi_registry=roi_registry,
        )

        # Define shape-corrected baseline.
        if self.registration is not None:
            self.shape_corrected_baseline = darsia.OpticalImage(
                pre_baseline.img.copy(),
                transformations=self.shape_corrections,
                color_space="RGB",
                dimensions=list(pre_baseline.dimensions),
                origin=list(pre_baseline.origin),
            )
        else:
            self.shape_corrected_baseline = darsia.imread(
                baseline_path,
                transformations=self.shape_corrections,
                **self.raw_decoding,
            )
        self.baseline = self.shape_corrected_baseline.copy()

        self.canvas = self._canvas(baseline_path, corrections_config)
        """Canvas of the corrected images, which label sets and phase models record."""

        # Plot corrected baseline - show and/or log.
        plt.figure("Corrected baseline")
        plt.imshow(self.baseline.img)
        plt.title("Corrected baseline")
        if show_plot:
            plt.show()
        if log:
            plt.savefig(
                log / "corrected_baseline.png",
                dpi=500,
            )
        plt.close()

        logger.info("Reading setup completed.")

    def setup_registration(
        self, baseline_path: Path, corrections_config: CorrectionsConfig
    ) -> darsia.OpticalImage:
        """Set up each raw photograph's registration from the baseline's raw file:
        the setup's reference and homography (:func:`darsia.setup_frame_registration`)
        for :meth:`read_image`; the baseline registered as a frame.

        The registration replaces the shape chain: its reference is the camera
        grid cropped to the rig points, so a stored mapping, a resize, a drift
        correction and a curvature correction are refused beside it. A baseline
        that records other lens tables than those the configured model was
        calibrated with (``lens_tables``) is refused.

        Parameters
        ----------
        baseline_path : Path
            The baseline's raw file.
        corrections_config : CorrectionsConfig
            The corrections, with [corrections.registration] and
            [corrections.raw], and without another shape correction.

        Returns
        -------
        darsia.OpticalImage
            The baseline on the reference's grid, NaN where invalid.

        Raises
        ------
        ValueError
            If the photographs are not raw, another shape correction is configured
            too, or the baseline's lens tables are not the calibration's.

        """
        config: FrameRegistrationConfig = corrections_config.registration
        if not self.raw_decoding:
            raise ValueError("[corrections.registration] needs [corrections.raw].")
        chain = [
            f"[corrections.{name}]"
            for name in ("mapping", "resize", "drift", "curvature")
            if getattr(corrections_config, name)
        ]
        if chain:
            raise ValueError(
                "[corrections.registration] replaces the shape chain: "
                f"{', '.join(chain)} cannot be configured beside it."
            )
        if config.lens_tables is not None:
            recorded = darsia.read_camera_lens_tables(baseline_path, config.camera_lens)
            if recorded != config.lens_tables:
                raise ValueError(
                    f"{baseline_path} records other lens tables than the lens model "
                    "was calibrated with."
                )
        crop = config.crop
        self.registration, baseline, self.setup_registration_quality = (
            darsia.setup_frame_registration(
                baseline_path,
                config.settings,
                config.lens,
                config.camera_shape,
                config.channel,
                self.raw_decoding["demosaic"],
                self.raw_decoding["dtype"],
                corners=None if crop is None else crop.pts_src,
                size=None if crop is None else (crop.width, crop.height),
            )
        )
        return darsia.OpticalImage(
            baseline,
            color_space="RGB",
            dimensions=list(self.registration.dimensions),
            origin=list(self.registration.origin),
        )

    def _canvas(
        self, baseline_path: Path, corrections_config: CorrectionsConfig | None
    ) -> dict:
        """The canvas of the corrected images, as a record of plain settings.

        Labels are drawn on this canvas and phase models are trained on it. The
        record has three parts:

        - ``source``: the file format of the photographs;
        - ``corrections``: the active corrections as plain settings, e.g. the
          precision, the mapping and the curvature correction, and so the
          decode of raw photographs or a lens model where the configuration
          holds them;
        - ``grid``: the shape, the physical dimensions (m) and the origin of the
          corrected images in rig coordinates (:func:`darsia.grid_record`), and
          under ``extent`` the dimensions and origin of the canvas before the
          scaling onto whole metric cells extends it, which do not depend on the
          resolution. It is None unless the corrections place the images in
          metres: the curvature correction's crop in metres, or, without a crop,
          the mapping onto its metric grid. Camera frames and a crop in pixels
          have no grid.

        The run's optical area factor is left out: it scales areas, not label pixels.

        """
        config = corrections_config or CorrectionsConfig()
        # The crop follows the mapping and gives the images the size it states.
        crop = config.curvature.crop if config.curvature else None
        crop = crop or (config.registration.crop if config.registration else None)
        metres = crop.in_meters if crop else config.mapping is not None
        grid = None
        if metres:
            grid = darsia.grid_record(self.shape_corrected_baseline)
            extent = {"dimensions": grid["dimensions"], "origin": grid["origin"]}
            for scaling in self.shape_corrections:
                if isinstance(scaling, darsia.MetricScalingCorrection):
                    # The canvas the scaling keeps, before whole cells extend it.
                    extent = {
                        "dimensions": list(scaling.native_dimensions),
                        "origin": list(scaling.native_origin),
                    }
            grid["extent"] = extent
        corrections = config.plain_settings()
        corrections.get("grid", {}).pop("area_factor", None)
        return {
            "source": {"format": baseline_path.suffix.lstrip(".").upper()},
            "corrections": corrections,
            "grid": grid,
        }

    def load_corrections(
        self,
        folder: Path,
        corrections_config: CorrectionsConfig | None = None,
        pixel_area: bool | None = None,
        shape_files: list[str] | None = None,
        color_files: list[str] | None = None,
    ) -> None:
        """Load persisted corrections from disk.

        The rig's record names its shape and colour correction files
        (``shape_files``, ``color_files``), which are read by name: one not found
        raises. A rig saved before the record has the split-format files
        (``shape_correction_*``, ``color_correction_*``) or legacy mixed-format
        files (``correction_*``) its folder lists, and a listing of a share that is
        out raises. The measured pixel areas of the run grid are read with them:
        where the rig's record says it has them (``pixel_area``), a file not found
        raises; a rig saved without them has none; a rig saved before the record has
        them where the file is found, a share that is out raising.
        """
        if corrections_config is None:
            corrections_config = CorrectionsConfig()

        self.shape_corrections = []
        self.color_corrections = []
        area = folder / "pixel_area.npz"
        if pixel_area is None:
            pixel_area = results.exists(area)
        self.pixel_area = darsia.imread(area) if pixel_area else None

        if shape_files is not None and color_files is not None:
            shape_paths = [folder / name for name in shape_files]
            color_paths = [folder / name for name in color_files]
        else:
            shape_paths = _listed(folder, "shape_correction_")
            color_paths = _listed(folder, "color_correction_")
        if shape_paths or color_paths:
            for correction_path in shape_paths:
                correction = darsia.read_correction(correction_path)
                if self._is_shape_correction(correction):
                    self.shape_corrections.append(correction)
                    logger.info(f"Loaded shape correction {type(correction).__name__}")
                else:
                    logger.warning(
                        f"Skipping non-shape correction in shape pipeline: "
                        f"{type(correction).__name__}"
                    )
            for correction_path in color_paths:
                correction = darsia.read_correction(correction_path)
                if self._is_color_correction(correction):
                    self.color_corrections.append(correction)
                    logger.info(f"Loaded color correction {type(correction).__name__}")
                else:
                    logger.warning(
                        f"Skipping non-color correction in color pipeline: "
                        f"{type(correction).__name__}"
                    )
            logger.info("Corrections loaded from split format.")
            return

        for correction_path in _listed(folder, "correction_"):
            correction = darsia.read_correction(correction_path)
            if self._is_shape_correction(correction):
                self.shape_corrections.append(correction)
                logger.info(f"Loaded shape correction {type(correction).__name__}")
            elif self._is_color_correction(correction):
                self.color_corrections.append(correction)
                logger.info(f"Loaded color correction {type(correction).__name__}")
            else:
                logger.warning(
                    f"Skipping unknown correction type {type(correction).__name__}"
                )
        logger.info("Corrections loaded from legacy format.")

    def setup_shape_corrections(
        self,
        pre_baseline: darsia.Image,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
    ) -> None:
        """Setup shape corrections that do not depend on labels/porosity."""
        if corrections_config is None:
            corrections_config = CorrectionsConfig()

        self.shape_corrections = []
        """Shape corrections applied prior to label-dependent corrections."""

        baseline_for_setup = pre_baseline

        # The precision of the run grid, or of the type correction.
        grid = corrections_config.grid
        target_type = corrections_config.precision

        if target_type is not None:
            # Aux: needed for rescaling not leaving the range of color space
            self.type_converter = darsia.TypeCorrection(target_type)
            """Type correction to convert images to float32."""
            baseline_for_setup = self.type_converter(baseline_for_setup)

            # Update corrections workflow
            self.shape_corrections.append(self.type_converter)

        # Track the combined resize factor for downstream corrections (e.g.
        # curvature).
        self.resize_factor = 1.0

        if corrections_config.resize:
            # Apply user-configured resize if present.
            current_shape = baseline_for_setup.shape[: baseline_for_setup.space_dim]
            target_shape, self.resize_factor = corrections_config.resize.resolve(
                current_shape
            )

            self.rescale_correction = darsia.ResizeCorrection(shape=target_shape)
            """User-configured resize correction."""

            # Update baseline and corrections workflow
            baseline_for_setup = self.rescale_correction(baseline_for_setup)
            self.shape_corrections.append(self.rescale_correction)

        if corrections_config.mapping:
            # Sample onto the target grid of a stored mapping. All later shape
            # corrections, and labels and facies, refer to the target grid.
            self.mapping_correction = darsia.ImageMappingCorrection()
            """Mapping onto the target grid at stored coordinates."""
            self.mapping_correction.load(corrections_config.mapping.path)

            # Update baseline and corrections workflow
            baseline_for_setup = self.mapping_correction(baseline_for_setup)
            self.shape_corrections.append(self.mapping_correction)

        # Resize for consistency (drift, curvature). Free no-op when shapes
        # already match, via early return in Resize.resize_array.
        self.resize_correction = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim]
        )
        """Resize correction to baseline shape (before drift/curvature)."""

        # Update baseline and corrections workflow
        baseline_for_setup = self.resize_correction(baseline_for_setup)
        self.shape_corrections.append(self.resize_correction)

        self.resize_correction_inter_nearest = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim],
            interpolation="inter_nearest",
        )
        """Resize for int data (labels/facies) before drift/curvature."""

        if corrections_config.drift:
            # Define translation correction object, based on the configured
            # reference mode: a named ROI, or auto-detected color checker.
            if corrections_config.drift.mode == "roi":
                roi_voxels = _resolve_named_roi_voxels(
                    corrections_config.drift.roi, roi_registry, baseline_for_setup
                )
                self.drift_correction = darsia.DriftCorrection(
                    baseline_for_setup, config={"roi": roi_voxels}
                )
                """Drift correction based on a named ROI."""
            else:
                try:
                    _, cc_voxels = darsia.find_colorchecker(
                        baseline_for_setup, corrections_config.drift.colorchecker
                    )
                    self.drift_correction = darsia.DriftCorrection(
                        baseline_for_setup, config={"roi": cc_voxels}
                    )
                    """Drift correction based on color checker alignment."""
                except Exception as e:
                    warn(
                        f"Color checker not found. Drift correction not setup. "
                        f"Error: {e}",
                        UserWarning,
                    )
                    self.drift_correction = darsia.DriftCorrection(baseline_for_setup)

            # Update baseline and corrections workflow
            ...  # baseline image does not need drift correction.
            self.shape_corrections.append(self.drift_correction)

        if corrections_config.curvature:
            # Define curvature correction as derived from analysis of laser grid
            # images. Pass resize_factor to rescale pixel-calibrated config values
            # (crop corners, bulge/stretch offsets) if images were resized.
            self.curvature_correction = darsia.CurvatureCorrection(
                config=corrections_config.curvature.to_dict(),
                resize_factor=self.resize_factor,
            )
            """Curvature correction based on laser grid analysis."""

            # Update baseline and corrections workflow
            baseline_for_setup = self.curvature_correction(baseline_for_setup)
            self.shape_corrections.append(self.curvature_correction)

        # Harmonize back to canonical baseline shape. Free no-op when shapes
        # already match, via early return in Resize.resize_array.
        self.resize_correction_final = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim]
        )
        """Final resize correction harmonizing output to baseline shape."""

        # Update baseline and corrections workflow
        baseline_for_setup = self.resize_correction_final(baseline_for_setup)
        self.shape_corrections.append(self.resize_correction_final)

        self.resize_correction_inter_nearest_final = darsia.ResizeCorrection(
            shape=baseline_for_setup.shape[: baseline_for_setup.space_dim],
            interpolation="inter_nearest",
        )
        """Final nearest-interpolation harmonizer for label/facies data."""

        self.pixel_area = None
        """Measured area [m2] of each pixel of the run grid; None without a grid."""
        if grid:
            # The coordinate plane's pixel area times the run's optical area factor.
            area = np.full(
                baseline_for_setup.shape[: baseline_for_setup.space_dim],
                float(np.prod(baseline_for_setup.voxel_size)) * grid.area_factor,
            )

        if grid and grid.pixel_size_mm:
            # Scale once onto whole metric cells of the chosen size, after the
            # corrections that give the images their physical dimensions.
            canvas = (
                grid.pixel_size_mm,
                baseline_for_setup.shape[: baseline_for_setup.space_dim],
                baseline_for_setup.dimensions,
                baseline_for_setup.origin,
            )
            self.resolution_correction = darsia.MetricScalingCorrection(*canvas)
            """Scaling of the images onto the cells of the run grid."""
            baseline_for_setup = self.resolution_correction(baseline_for_setup)
            self.shape_corrections.append(self.resolution_correction)
            # Labels and facies take the majority label of each cell.
            self.resize_correction_inter_nearest_final = darsia.MetricScalingCorrection(
                *canvas, labels=True
            )
            # A cell's area is the sum of its pixels' areas.
            area = darsia.cell_sums(self.resolution_correction.cells, area)

        if grid:
            self.pixel_area = darsia.ScalarImage(
                area.astype(target_type),
                dimensions=baseline_for_setup.dimensions,
                origin=baseline_for_setup.origin,
            )

        logger.info("Shape corrections setup complete.")

    def setup_color_corrections(
        self,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup label-dependent color corrections after labels/porosity are available.

        Execution order is fixed:
        1) illumination, 2) relative color, 3) color correction.
        Relative color can either be loaded from a precomputed correction file or
        calibrated from configured images (interactive mode only).
        """
        if corrections_config is None:
            corrections_config = CorrectionsConfig()
        if not hasattr(self, "shape_corrected_baseline"):
            raise RuntimeError(
                "Shape-corrected baseline missing. Run setup_shape_corrections first."
            )

        self.color_corrections = []
        """Color corrections initialized after labels and porosity are available."""

        # 1) Illumination correction.
        if corrections_config.illumination:
            self.illumination_correction = self.setup_illumination_correction(
                corrections_config.illumination,
                log=log,
                show_plot=show_plot,
            )
            self.color_corrections.append(self.illumination_correction)

        # 1b) patchwise illumination correction.
        if corrections_config.patchwise_illumination:
            assert (
                not corrections_config.illumination
            ), "Can only use a single illumination correction method at a time."
            self.illumination_correction = self.setup_patchwise_illumination_correction(
                corrections_config.patchwise_illumination,
                show_plot=show_plot,
            )
            self.color_corrections.append(self.illumination_correction)

        # 1c) Material illumination correction.
        if corrections_config.material_illumination:
            assert not (
                corrections_config.illumination
                or corrections_config.patchwise_illumination
            ), "Can only use a single illumination correction method at a time."
            self.illumination_correction = self.setup_material_illumination_correction(
                corrections_config.material_illumination
            )
            self.color_corrections.append(self.illumination_correction)

        # 2) Relative color correction (reserved in ordering; setup currently guarded).
        if corrections_config.relative_color:
            relative_config = corrections_config.relative_color
            if relative_config.path is not None:
                self.relative_color_correction = darsia.RelativeColorCorrection(
                    self.shape_corrected_baseline
                )
                self.relative_color_correction.load(relative_config.path)
                self.color_corrections.append(self.relative_color_correction)
            elif relative_config.images:
                if not relative_config.interactive:
                    raise ValueError(
                        "Interactive calibration is required when using "
                        "corrections.relative_color.images. Set "
                        "corrections.relative_color.interactive=true or provide "
                        "corrections.relative_color.path."
                    )
                relative_color_images = []
                for path in relative_config.images:
                    img = darsia.imread(path)
                    for correction in self.shape_corrections:
                        img = correction(img)
                    relative_color_images.append(img)
                self.relative_color_correction = darsia.RelativeColorCorrection(
                    self.shape_corrected_baseline,
                    relative_color_images,
                    relative_config.to_options_dict(),
                )
                self.color_corrections.append(self.relative_color_correction)

        # 3) Color correction.
        if corrections_config.color:
            # Define color correction, based on the configured reference
            # mode: a named ROI (as 4 ordered landmark corners), or
            # auto-detected color checker.
            if corrections_config.color.mode == "roi":
                roi_voxels = _resolve_named_roi_voxels(
                    corrections_config.color.roi,
                    roi_registry,
                    self.shape_corrected_baseline,
                    as_landmarks=True,
                )
                self.color_correction = darsia.ColorCorrection(
                    self.shape_corrected_baseline,
                    config={
                        "roi": roi_voxels,
                        "clip": False,
                    },
                )
                """Color correction based on a named ROI."""
            else:
                try:
                    _, cc_voxels = darsia.find_colorchecker(
                        self.shape_corrected_baseline,
                        corrections_config.color.colorchecker,
                    )
                    self.color_correction = darsia.ColorCorrection(
                        self.shape_corrected_baseline,
                        config={
                            "roi": cc_voxels,
                            "clip": False,
                        },
                    )
                    """Color correction based on color checker alignment."""
                except Exception as e:
                    warn(
                        f"Color checker not found. Color correction not setup. "
                        f"Error: {e}",
                        UserWarning,
                    )
                    self.color_correction = darsia.ColorCorrection(
                        self.shape_corrected_baseline
                    )
            self.color_corrections.append(self.color_correction)

        # Apply the configured color correction pipeline to the shape-corrected baseline.
        # At this point, setup has already been run for each configured correction.
        self.baseline = self.shape_corrected_baseline.copy()
        for correction in self.color_corrections:
            self.baseline = correction(self.baseline)

        logger.info("Color corrections setup complete.")

    # ! ---- GEOMETRY ----

    def setup_depth(
        self,
        path: Path,
        log: Path | None = None,
    ) -> None:
        """Setup depth map for the rig object.

        Parameters
        ----------
        path : Path | None
            Path to the precomputed depth map file.
            If provided, it will be loaded and used as the depth map.
        log : Path | None
            Path to the log folder where geometry images will be saved.
        """
        # Load depth map from file and reshape to baseline shape
        pre_depth = darsia.imread(path)
        self.depth: darsia.Image = darsia.resize(pre_depth, ref_image=self.baseline)
        """Depth map for the rig object."""

        # Log results
        if log:
            # Plot depth map
            plt.figure()
            plt.imshow(self.depth.img)
            plt.colorbar()
            plt.title("Depth map")
            plt.savefig(log / "depth_map.png", dpi=500)
            plt.close()

        logging.info("Depth map setup completed.")

    def setup_geometry(self) -> None:
        """Setup geometry for volumetric integration of images."""
        shape_meta = self.baseline.shape_metadata()
        self.geometry = darsia.ExtrudedPorousGeometry(
            depth=self.depth, porosity=self.porosity, **shape_meta
        )
        """Geometry for volumetric integration of images."""
        logging.info("Geometry setup completed.")

    # ! ---- LABELS -----

    def setup_labels(
        self, path: Path, apply_corrections: bool = False, log: Path | None = None
    ) -> None:
        """Setup labels for the rig object.

        This method loads labels from a specified path and applies corrections if needed.

        Parameters
        ----------
        path : Path
            Path to the labels file. If the file exists, it will be loaded.
        apply_corrections : bool
            If True, applies corrections to the labels based on
            the baseline image.
        log : Path | None
            Path to the log folder where label images will be saved.
        """
        if apply_corrections:
            # Assume that the labels are based on non-corrected baseline image.
            # Thus, need to apply the relevant corrections from the read routine.
            plain_labels = darsia.imread(path)
            labels = plain_labels
            if hasattr(self, "resize_correction_inter_nearest"):
                labels = self.resize_correction_inter_nearest(labels)
            if hasattr(self, "curvature_correction"):
                labels = self.curvature_correction(labels)
            labels = self.resize_correction_inter_nearest_final(labels)
            self.labels = labels

        else:
            # Assume the labels are aligned with the corrected baseline.
            pre_labels = darsia.imread(path)
            self.labels = darsia.resize(
                pre_labels,
                ref_image=self.baseline,
                interpolation="inter_nearest",
            )

        if log:
            # Plot labels
            plt.figure()
            plt.imshow(self.labels.img)
            plt.colorbar()
            plt.title("Labels")
            plt.savefig(Path(log) / "labels.png", dpi=500)
            plt.close()

        logger.info("Labels setup completed.")

    def setup_inner_labels(self, log: Path | None = None) -> None:
        """Define boolean mask which excludes the label boundaries."""
        # Identify the boundary pixels.
        boundary_mask = skimage.segmentation.find_boundaries(
            self.labels.img, mode="outer", connectivity=1
        )

        # Extend the boundary pixels by a certain number of pixels (e.g., 2)
        # to create a buffer zone.
        # TODO: Allow for configuration of the buffer size and structuring element.
        buffer_zone = skimage.morphology.binary_dilation(
            boundary_mask, footprint=skimage.morphology.disk(2)
        )

        # Create a boolean mask where the inner labels (excluding the boundary)
        # are True and the rest is False.
        inner_labels = ~buffer_zone

        self.inner_labels = darsia.Image(
            inner_labels.astype(bool), **self.labels.metadata()
        )
        """Boolean mask for inner labels, excluding boundaries."""

    def setup_facies(
        self,
        path: Path,
        apply_corrections: bool = False,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup facies.

        This method loads facies from a specified path and applies corrections if needed.

        Parameters
        ----------
        path : Path
            Path to the facies file. If the file exists, it will be loaded.
        apply_corrections : bool
            If True, applies corrections to the facies based on
            the baseline image.
        log : Path | None
            Path to the log folder where facies images will be saved.
        """
        if apply_corrections:
            # Assume that the facies are based on non-corrected baseline image.
            # Thus, need to apply the relevant corrections from the read routine.
            plain_facies = darsia.imread(path)
            facies = plain_facies
            if hasattr(self, "resize_correction_inter_nearest"):
                facies = self.resize_correction_inter_nearest(facies)
            if hasattr(self, "curvature_correction"):
                facies = self.curvature_correction(facies)
            facies = self.resize_correction_inter_nearest_final(facies)
            self.facies = facies

        else:
            # Assume the facies are aligned with the corrected baseline.
            pre_facies = darsia.imread(path)
            self.facies = darsia.resize(
                pre_facies,
                ref_image=self.baseline,
                interpolation="inter_nearest",
            )

        if log:
            # Plot facies
            plt.figure()
            plt.imshow(self.facies.img)
            plt.colorbar()
            plt.title("Facies")
            plt.savefig(log / "facies.png", dpi=500)
            plt.close()

        if show_plot:
            # Display facies on top of baseline for visual inspection.
            plt.figure()
            plt.imshow(self.baseline.img)
            plt.imshow(self.facies.img, alpha=0.5)
            plt.title("Facies overlay on baseline")
            plt.show()

        logger.info("Facies setup completed.")

    def setup_facies_props(
        self,
        props_path: Path | None = None,
        porosity: Path | None = None,
        permeability: Path | None = None,
    ) -> None:
        """Define facies properties like porosity.

        Parameters
        ----------
        props_path : Path | None
            Path to the facies properties CSV file.
            If provided, it will load the facies properties from this path.
        porosity : Path | None
            Path to the porosity image file. If provided,
            it will load the porosity image from this path.
        """
        if props_path:
            facies_props = FaciesProps.load(facies=self.facies, path=props_path)
            self.porosity = facies_props.porosity
            """Porosity for the rig object."""
            self.permeability = facies_props.permeability
            """Permeability for the rig object."""
        else:
            if not (porosity and permeability):
                raise FileNotFoundError("No facies properties provided.")
            self.porosity = darsia.imread(porosity)
            self.permeability = darsia.imread(permeability)

    # ! ---- ILLUMINATION CORRECTION ----
    def setup_illumination_correction(
        self,
        config: IlluminationCorrectionConfig | None,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> darsia.IlluminationCorrection:
        """Setup and return illumination correction.

        Parameters
        ----------
        config : IlluminationCorrectionConfig | None
            Configuration for the illumination
            correction. If provided, it will set up the illumination correction based
            on this configuration.
        log : Path | None
            Path to the log folder where diagnostic plots will be saved.
        show_plot : bool
            Whether to show diagnostic plots during setup (default: False).

        Notes
        -----
            Illumination calibration in Rig intentionally uses the shape-corrected
            baseline as setup input.
        """
        illumination_correction = darsia.IlluminationCorrection()

        # Fetch samples for illumination correction based on labels and baseline.
        if config is not None:
            sample_groups = []
            if not config.labels:
                # If no labels specified, use random samples from the whole image.
                samples = illumination_correction.select_random_samples(
                    mask=darsia.ones_like(self.shape_corrected_baseline, dtype=bool),
                    config=config,
                )
                sample_groups.append(samples)
            else:
                for label in config.labels:
                    assert (
                        label in self.labels.img
                    ), f"Label {label} not found in labels image."
                    mask = self.labels.img == label
                    samples = illumination_correction.select_random_samples(
                        mask=mask, config=config
                    )
                    sample_groups.append(samples)

            # Determine illumination correction based on inputs
            illumination_correction.setup(
                # Use shape-corrected baseline as explicit setup input.
                base=self.shape_corrected_baseline,
                sample_groups=sample_groups,
                mask=self.boolean_porosity,
                outliers=config.outliers,
                filter=lambda x: skimage.filters.gaussian(x, sigma=config.sigma),
                colorspace=config.colorspace,
                interpolation=config.interpolation,
                show_plot=show_plot,
                log=log,
            )

        return illumination_correction

    def setup_material_illumination_correction(
        self, config: MaterialIlluminationCorrectionConfig
    ) -> darsia.MaterialIlluminationCorrection:
        """Fit and return the material illumination correction.

        The fit uses the shape-corrected baseline and the facies as materials.
        Pixels of facies without a target are corrected with the common field.
        With a configured reference run, the fit uses the reference's fitting grid.

        Parameters
        ----------
        config : MaterialIlluminationCorrectionConfig
            Configuration of the material illumination correction.

        Returns
        -------
        darsia.MaterialIlluminationCorrection
            Correction for images on the grid of the shape-corrected baseline.
        """
        facies = self.facies.img
        if config.fit_mask is None:
            fit_mask = np.isin(facies, list(config.targets))
        else:
            fit_mask = darsia.imread(config.fit_mask).img.astype(bool)
        reference = None
        if config.reference is not None:
            baseline = darsia.imread(config.reference.baseline)
            reference_facies = darsia.imread(config.reference.facies).img
            if config.reference.fit_mask is None:
                reference_mask = np.isin(reference_facies, list(config.targets))
            else:
                reference_mask = darsia.imread(config.reference.fit_mask).img
            reference = darsia.MaterialFitReference(
                baseline=baseline.img,
                material_map=reference_facies,
                fit_mask=reference_mask.astype(bool),
                dimensions=tuple(baseline.dimensions),
                homography=np.asarray(config.reference.homography),
            )
        return darsia.fit_material_illumination(
            self.shape_corrected_baseline.img,
            facies,
            fit_mask,
            config.targets,
            dimensions=self.shape_corrected_baseline.dimensions,
            fit_shape=config.fit_shape,
            block_size=config.block_size,
            edge_margin=config.edge_margin,
            dn_scale=config.dn_scale,
            settings=config.fit,
            reference=reference,
        )

    # ! ---- ILLUMINATION CORRECTION ----
    def setup_patchwise_illumination_correction(
        self,
        config: PatchwiseIlluminationCorrectionConfig | None,
        show_plot: bool = False,
    ) -> darsia.PatchwiseIlluminationCorrection:
        """Setup and return patchwise illumination correction.

        Parameters
        ----------
        config : PatchwiseIlluminationCorrectionConfig | None
            Configuration for
            the patchwise illumination correction. If provided, it will set up the
            correction based on this configuration.

        Notes
        -----
            Patchwise illumination calibration uses the labels image to enable
            per-label reference color computation within each patch, ensuring
            heterogeneous regions contribute equally by material.
        """
        baseline_images = [self.read_image(path) for path in config.baseline_paths]

        illumination_correction = darsia.PatchwiseIlluminationCorrection(
            baseline_images=baseline_images,
            labels=self.labels,
            nw=config.nw,
            eps=config.eps,
            show_images=show_plot,
        )
        return illumination_correction

    # ! ---- POROSITY ----

    def setup_image_porosity(
        self,
        path: Path | None = None,
        log: Path | None = None,
        config: ImagePorosityConfig | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup image porosity based on the baseline image.

        Behaviour is controlled by *config*:

        * ``mode="full"`` (default): constant porosity of ``1`` over the full domain.
        * ``mode="from_image"``: porosity derived from the baseline image via
          :func:`~darsia.patched_porosity_analysis` using the parameters stored in
          *config* (``patches``, ``num_clusters``, ``sample_width``,
          ``tol_color_distance``, ``tol_color_gradient``).

        When *path* is provided the image is always loaded from disk regardless of
        *config*, which is useful for restoring a previously saved rig.

        Parameters
        ----------
        path : Path | None
            Path to a previously saved porosity ``.npz`` file.
            When given, the file is loaded and *config* is not used.
        log : Path | None
            Folder for diagnostic output.  When given, a JPG
            illustration is stored to ``log/image_porosity/image_porosity.jpg``.
        config : ImagePorosityConfig | None
            Porosity configuration.  Defaults to
            ``ImagePorosityConfig()`` (i.e. ``mode="full"``) when not provided.
        show_plot : bool
            When ``True`` the image porosity is displayed interactively.
            Pass ``True`` when calling from a GUI or user-interface workflow.
        """
        if config is None:
            config = ImagePorosityConfig()
        self._image_porosity_config = config

        if path is not None:
            self.image_porosity = darsia.imread(path)
        elif config.mode == "from_image":
            self.image_porosity = patched_porosity_analysis(
                baseline=self.baseline,
                patches=config.patches,
                labels=self.labels,
                num_clusters=config.num_clusters,
                sample_width=config.sample_width,
                tol_color_distance=config.tol_color_distance,
                tol_color_gradient=config.tol_color_gradient,
            )
        else:
            # mode == "full"
            self.image_porosity = darsia.ones_like(
                self.baseline, mode="voxels", dtype=np.float32
            )
        """Image porosity for the rig object."""

        if log:
            out_dir = Path(log)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_scalar_map_illustration(
                self.image_porosity.img,
                out_dir / "image_porosity.jpg",
                title="Image porosity",
                colorbar_label="Porosity",
            )

        if show_plot:
            self.image_porosity.show(title="Image porosity")

        logger.info("Porosity setup completed.")

    def setup_boolean_image_porosity(
        self,
        threshold: float | None = None,
        log: Path | None = None,
        config: ImagePorosityConfig | None = None,
        show_plot: bool = False,
    ) -> None:
        """Setup boolean porosity based on the defined threshold.

        In ``mode="full"`` the boolean mask is always all-``True`` (full image domain),
        regardless of *threshold* / *tol*.  In ``mode="from_image"`` the mask is
        derived by thresholding ``self.image_porosity`` with the effective tolerance.

        The effective tolerance is resolved in order of precedence:

        1. *threshold* argument (when explicitly passed).
        2. ``config.tol`` (when *config* is given).
        3. ``self._image_porosity_config.tol`` (stored by :meth:`setup_image_porosity`).
        4. ``0.9`` (hard-coded default).

        Parameters
        ----------
        threshold : float | None
            Override tolerance value.  Deprecated in favour of
            ``config.tol``; kept for backward compatibility.
        log : Path | None
            Folder for diagnostic output.  When given, a JPG
            illustration is stored to
            ``log/image_porosity/boolean_porosity.jpg``.
        config : ImagePorosityConfig | None
            Porosity configuration.  Falls back to
            the config stored by the last call to :meth:`setup_image_porosity`, and
            finally to ``ImagePorosityConfig()`` (``mode="full"``).
        show_plot : bool
            When ``True`` the boolean porosity is displayed
            interactively.  Pass ``True`` when calling from a GUI or
            user-interface workflow.
        """
        if config is None:
            config = getattr(self, "_image_porosity_config", ImagePorosityConfig())

        # Resolve effective threshold: explicit argument wins over config.
        tol = threshold if threshold is not None else config.tol

        if config.mode == "full":
            # Always full boolean mask regardless of tol.
            self.boolean_porosity = darsia.ones_like(
                self.baseline, mode="voxels", dtype=bool
            )
        else:
            # from_image: threshold the continuous porosity map.
            self.boolean_porosity = self.image_porosity > tol

        if log:
            out_dir = Path(log)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_scalar_map_illustration(
                self.boolean_porosity.img.astype(float),
                out_dir / "boolean_porosity.jpg",
                title="Boolean porosity",
                colorbar_label="Porosity (boolean)",
            )

        if show_plot:
            self.boolean_porosity.show(title="Boolean porosity")

        logger.info("Boolean porosity setup completed.")

    def setup(
        self,
        experiment: darsia.ProtocolledExperiment,
        baseline_path: Path,
        depth_map_path: Path | None,
        labels_path: Path,
        facies_path: Path | None = None,
        facies_props_path: Path | None = None,
        corrections_config: CorrectionsConfig | None = None,
        roi_registry: "RoiRegistry | None" = None,
        image_porosity_config: ImagePorosityConfig | None = None,
        # ref_colorchecker_path: Path,
        log: Path | None = None,
        show_plot: bool = False,
    ) -> None:
        """Fast setup.

        Without a depth map (``depth_map_path`` None) the rig has no depth and no
        geometry yet: the caller sets the depth, e.g. by a depth rule evaluated on
        the rig's grid, and then the geometry (:meth:`setup_geometry`).
        """
        # Create log directory if it doesn't exist
        if log:
            assert isinstance(log, Path), "Log path must be a Path object."
            log.mkdir(parents=True, exist_ok=True)

        # Cache baseline path - TODO: Really needed?
        self.baseline_path = baseline_path

        # Cache reference date
        self.reference_date = experiment.experiment_start

        # Initialize the rig - responsible for reading/preprocessing photographs
        self.setup_reading(
            baseline_path,
            experiment,
            corrections_config=corrections_config,
            roi_registry=roi_registry,
            log=log,
            show_plot=show_plot,
        )

        # Fetch depth map
        if depth_map_path is not None:
            self.setup_depth(
                depth_map_path,
                log=log,
            )

        # Add labels
        self.setup_labels(
            path=labels_path,
            apply_corrections=True,
            log=log,
        )

        # Setup inner labels
        self.setup_inner_labels(
            log=log,
        )

        # Add facies
        if facies_path is not None:
            self.setup_facies(
                path=facies_path,
                apply_corrections=True,
                log=log,
                show_plot=show_plot,
            )
        else:
            self.facies = self.labels.copy()

        # Setup facies props
        self.setup_facies_props(facies_props_path)

        # Define geometry for integration
        if depth_map_path is not None:
            self.setup_geometry()

        # Pre-setup image and boolean porosity based on baseline (redo after color correction)
        self.setup_image_porosity(
            log=log, config=image_porosity_config, show_plot=show_plot
        )
        self.setup_boolean_image_porosity(log=log, show_plot=show_plot)

        # Setup color corrections (wait until here to use label and porosity information)
        self.setup_color_corrections(
            corrections_config=corrections_config,
            roi_registry=roi_registry,
            log=log,
            show_plot=show_plot,
        )

        # Redo the porosity setup after color corrections, as the latter can change the image.
        if len(self.color_corrections) > 0:
            self.setup_image_porosity(
                log=log, config=image_porosity_config, show_plot=show_plot
            )
            self.setup_boolean_image_porosity(log=log, show_plot=show_plot)

        # TODO Setup concentration analysis and transformations?
        # self.co2_analysis = ...
        # self.co2_g_analysis = ...

        # Setup co2 mass analysis.
        # TODO Add?
        # self.setup_mass_analysis(
        #    atmospheric_pressure=kwargs.get("atmospheric_pressure", 1.010),
        #    temperature=kwargs.get("temperature", 23.0),
        # )

        logger.info("Rig setup completed.")

        # ! ---- AVERAGING ----

        # Use porosity-based averaging for restoration
        restoration = darsia.porosity_based_averaging(
            self.labels, self.image_porosity, self.baseline
        )

        # Cache
        self.restoration = restoration
        """Restoration model based on porosity-based averaging."""

        clipping = darsia.ClipModel(min_value=0.0)
        self.upscaling = darsia.CombinedModel([clipping] + 2 * [restoration])
        logger.info("Upscaling setup completed.")

        # ! ---- COMBINED MODELS ----

        # self.concentration_analysis_aq = darsia.CombinedModel(
        #    [self.co2_analysis, self.upscaling]
        # )
        # self.concentration_analysis_g = darsia.CombinedModel(
        #    [self.co2_g_analysis, self.upscaling]
        # )
        logger.info("Concentration analysis setup completed.")

    def setup_mass_analysis(
        self,
        atmospheric_pressure,
        atmospheric_temperature,
    ) -> None:
        # Define plain mass analysis (combining gas and aqueous phase)
        self.co2_mass_analysis = darsia.CO2MassAnalysis(
            self.baseline,
            atmospheric_pressure=atmospheric_pressure,
            atmospheric_temperature=atmospheric_temperature,
        )
        logger.info("Mass analysis setup completed.")

    # ! ---- ANALYSIS ----

    def mass_analysis(self, img: darsia.Image) -> darsia.MassAnalysisResults:
        """Mass analysis of the image."""
        raise NotImplementedError

    def threshold_analysis(
        self, mass_analysis_result: darsia.MassAnalysisResults
    ) -> darsia.ThresholdAnalysisResults:
        """Threshold analysis of the mass analysis result."""
        raise NotImplementedError

    # ! ---- I/O ----

    def save(self, folder: Path) -> None:
        """Save rig object to file."""

        # Create folder if not exists
        folder.mkdir(parents=True, exist_ok=True)

        # Dump meta data to json file
        # TODO: Needed for what?
        meta_data = {
            "baseline_path": str(self.baseline_path),
            "canvas": getattr(self, "canvas", None),
        }
        if getattr(self, "raw_decoding", {}):
            meta_data["raw_decoding"] = self.raw_decoding
        if self.grid_to_rig is not None:
            meta_data["grid_to_rig"] = np.asarray(self.grid_to_rig, float).tolist()
        if self.measurement_canvas is not None:
            meta_data["measurement_canvas"] = {
                "origin": [int(v) for v in self.measurement_canvas.origin],
                "shape": [int(v) for v in self.measurement_canvas.shape],
            }
        # The setup's reference of each photograph's registration, or none, and
        # which of the rig's optional files it has; the record says which, so that
        # a file not found is not taken for none.
        registration = getattr(self, "registration", None)
        meta_data["registration"] = registration is not None
        meta_data["shape_corrected_baseline"] = hasattr(
            self, "shape_corrected_baseline"
        )
        meta_data["pixel_area"] = getattr(self, "pixel_area", None) is not None
        meta_data["shape_corrections"] = [
            _correction_file("shape", i, c)
            for i, c in enumerate(self.shape_corrections)
        ]
        meta_data["color_corrections"] = [
            _correction_file("color", i, c)
            for i, c in enumerate(self.color_corrections)
        ]
        with open(folder / "meta_data.json", "w") as f:
            json.dump(meta_data, f)

        if registration is not None:
            darsia.Image(
                registration.reference,
                dimensions=list(registration.dimensions),
                origin=list(registration.origin),
            ).save(folder / "registration_reference.npz")
            record = {
                "homography": registration.homography.tolist(),
                "settings": asdict(registration.settings),
                "lens": [[type(m).__name__, asdict(m)] for m in registration.lens],
                "camera_shape": list(registration.camera_shape),
                "channel": registration.channel,
                "demosaic": registration.demosaic,
            }
            with open(folder / "registration.json", "w") as f:
                json.dump(record, f)
        else:
            (folder / "registration.json").unlink(missing_ok=True)
            (folder / "registration_reference.npz").unlink(missing_ok=True)

        # Save reading information
        self.baseline.save(folder / "baseline.npz")
        if hasattr(self, "shape_corrected_baseline"):
            self.shape_corrected_baseline.save(folder / "shape_corrected_baseline.npz")

        # Save split correction pipelines.
        saved_corrections = set()
        for kind, corrections in (
            ("shape", self.shape_corrections),
            ("color", self.color_corrections),
        ):
            for i, correction in enumerate(corrections):
                path = folder / _correction_file(kind, i, correction)
                correction.save(path)
                saved_corrections.add(path)

        # An earlier save's split corrections must not outlive the pipeline.
        for prefix in ("shape", "color"):
            for path in folder.glob(f"{prefix}_correction_*.npz"):
                if path not in saved_corrections and path.is_file():
                    path.unlink()

        # The pixel areas, or none: an earlier save's areas must not outlive them.
        if getattr(self, "pixel_area", None) is not None:
            self.pixel_area.save(folder / "pixel_area.npz")
        else:
            (folder / "pixel_area.npz").unlink(missing_ok=True)

        # Save geometry information
        try:
            assert isinstance(self.depth, darsia.Image)
            self.depth.save(folder / "depth.npz")
        except Exception:
            warn("Depth not available for saving.", UserWarning)

        # Save labels information
        try:
            self.labels.save(folder / "labels.npz")
        except Exception:
            warn("Labels not available for saving.", UserWarning)

        # Save facies information
        try:
            self.facies.save(folder / "facies.npz")
        except Exception:
            warn("Facies not available for saving.", UserWarning)

        # Save facies properties
        try:
            self.porosity.save(folder / "porosity.npz")
        except Exception:
            warn("Porosity not available for saving.", UserWarning)

        try:
            self.permeability.save(folder / "permeability.npz")
        except Exception:
            warn("Permeability not available for saving.", UserWarning)

        # Save porosity information
        try:
            self.image_porosity.save(folder / "image_porosity.npz")
        except Exception:
            warn("Image porosity not available for saving.", UserWarning)

        # Use green color and hyperlink
        folder_uri = Path(folder).absolute().as_uri()
        logger.info(
            """\033[92mRig object saved to \033]8;;"""
            f"""{folder_uri}\033\\{folder}\033]8;;\033\\\033[0m"""
        )

    @classmethod
    def load(
        cls, folder: Path, corrections_config: CorrectionsConfig | None = None
    ) -> "Rig":
        """Load rig object from file.

        Mimick the save method.

        Parameters
        ----------
        folder : Path
            Path to the folder where the rig object is saved.
        """
        # Create rig object
        rig = cls()

        # TODO: move into single function self.setup_basics() or so.
        # Load meta data
        with open(folder / "meta_data.json", "r") as f:
            meta_data = json.load(f)
        rig.baseline_path = Path(meta_data["baseline_path"])
        rig.raw_decoding = meta_data.get("raw_decoding", {})
        rig.grid_to_rig = meta_data.get("grid_to_rig")
        canvas = meta_data.get("measurement_canvas")
        rig.measurement_canvas = None
        if canvas is not None:
            rig.measurement_canvas = darsia.MeasurementCanvas(
                tuple(canvas["origin"]), tuple(canvas["shape"])
            )
        # A rig saved without a canvas has an unknown canvas.
        rig.canvas = meta_data.get("canvas")
        rig.registration = None
        if meta_data.get("registration", False):
            with open(folder / "registration.json", "r") as f:
                record = json.load(f)
            reference = darsia.imread(folder / "registration_reference.npz")
            rig.registration = darsia.FrameRegistration(
                reference=reference.img,
                homography=np.asarray(record["homography"], reference.img.dtype),
                settings=settings_from_table(
                    darsia.ProjectiveRegistrationSettings, record["settings"]
                ),
                lens=tuple(
                    settings_from_table(getattr(darsia, name), values)
                    for name, values in record["lens"]
                ),
                camera_shape=tuple(record["camera_shape"]),
                channel=record["channel"],
                demosaic=record["demosaic"],
                dimensions=tuple(reference.dimensions),
                origin=tuple(float(v) for v in reference.origin),
            )

        # Load data for reading images: a shape-corrected baseline the record names
        # is read, and not found it raises; a rig saved before the record has one
        # where the file is found, else its baseline.
        with darsia.timed("baseline"):
            rig.baseline = darsia.imread(folder / "baseline.npz")
            corrected = meta_data.get("shape_corrected_baseline")
            if corrected is None:
                corrected = results.exists(folder / "shape_corrected_baseline.npz")
            if corrected:
                rig.shape_corrected_baseline = darsia.imread(
                    folder / "shape_corrected_baseline.npz"
                )
            else:
                rig.shape_corrected_baseline = rig.baseline.copy()
        logger.info("Baseline setup complete.")

        # Load corrections
        with darsia.timed("corrections"):
            rig.load_corrections(
                folder,
                corrections_config=corrections_config,
                pixel_area=meta_data.get("pixel_area"),
                shape_files=meta_data.get("shape_corrections"),
                color_files=meta_data.get("color_corrections"),
            )

        # Load depth map
        with darsia.timed("depth"):
            rig.setup_depth(path=folder / "depth.npz")

        # Load labels information - corrections not needed assuming labels are aligned with
        # the corrected baseline.
        with darsia.timed("labels"):
            rig.setup_labels(path=folder / "labels.npz", apply_corrections=False)

        # Setup inner labels information
        with darsia.timed("inner labels"):
            rig.setup_inner_labels()

        # Load facies information - corrections not needed assuming facies are aligned with
        with darsia.timed("facies"):
            rig.setup_facies(path=folder / "facies.npz", apply_corrections=False)

        # Load facies properties
        with darsia.timed("facies properties"):
            rig.setup_facies_props(
                porosity=folder / "porosity.npz",
                permeability=folder / "permeability.npz",
            )

        # Setup geometry information
        with darsia.timed("geometry"):
            rig.setup_geometry()

        # Load image porosity information
        with darsia.timed("image porosity"):
            rig.setup_image_porosity(path=folder / "image_porosity.npz")
            rig.setup_boolean_image_porosity()

        # TODO setup color_analysis, and pw_transformation.

        logger.info("Rig object loaded.")
        return rig

    # ! ---- I/O ----

    def import_from_csv(
        self,
        path: Path,
        *,
        delimiter: str = ",",
        date=None,
        reference_date=None,
        time=None,
        name: str | None = None,
        is_extensive: bool = False,
    ) -> darsia.ScalarImage | darsia.ExtensiveImage:
        """Import scalar result data from CSV."""
        if not path.exists():
            raise FileNotFoundError(f"CSV file {path} does not exist.")

        # Read csv
        try:
            data = np.loadtxt(path, delimiter=delimiter)
        except ValueError:
            data = np.loadtxt(path, delimiter=delimiter, skiprows=1)

        # Apply lex sort
        # Sort, such that 0th column changes fasters, then 1st column.
        indices = np.lexsort((data[:, 0], data[:, 1]))
        data = data[indices]

        # Extract coordinates (x, y) and values if in coordinate format,
        # otherwise treat as array data
        coordinates_x = data[:, 0]
        coordinates_y = data[:, 1]
        values = data[:, 2]

        # Determine the shape = frequency of x_coordinates
        # (fastest changing) and y_coordinates (slowest changing)
        unique_x = np.unique(coordinates_x)
        unique_y = np.unique(coordinates_y)

        row = len(unique_y)
        col = len(unique_x)
        shape = (row, col)

        # Determine the cell size in dx and dy direction
        dx = np.min(np.diff(unique_x))
        dy = np.min(np.diff(unique_y))

        # Determine origin
        origin = (unique_x[0] - dx / 2, unique_y[-1] + dy / 2)

        # Determine the dimensions
        dimensions = (
            np.max(coordinates_y) - np.min(coordinates_y) + dy,
            np.max(coordinates_x) - np.min(coordinates_x) + dx,
        )

        # Reshape values to the determined shape, remember that the
        # values are ordered wrt. Euclidean coordinates, with x changing
        # fastest, so we need to reshape accordingly. Also, we need to
        # switch to row-col order, with origin at the top-left corner,
        # so we need to flip the y-coordinates and reshape accordingly.
        values_reshaped = values.reshape(
            shape, order="C"
        )  # Reshape to (row, col) with C order (x changes fastest)
        values_reshaped = np.flip(values_reshaped, axis=0)  # Flip

        # Collect the metadata
        metadata = {
            "origin": origin,
            "cell_size": (dx, dy),
            "dimensions": dimensions,
            "name": name,
            "time": time,
            "date": date,
            "reference_date": reference_date,
            "series": False,
            "scalar": True,
        }

        if is_extensive:
            return darsia.ExtensiveImage(values_reshaped, **metadata)
        else:
            return darsia.ScalarImage(values_reshaped, **metadata)

    @darsia.timed("read image")
    def read_image(self, path: Path) -> darsia.Image:
        """Read image from file and apply corrections.

        The respective date is extracted from the path using the imaging protocol.

        Parameters
        ----------
        path : Path
            Path to the image file.

        Returns
        -------
        darsia.Image
            Image object with applied corrections.
        """
        assert hasattr(
            self, "experiment"
        ), "Experiment not defined. Run load_experiment() first."
        if getattr(self, "registration", None) is not None:
            return self.read_registered_image(path)[0]

        # Convert date from path
        date = self.experiment.get_datetime(path)

        # Read image from file and apply corrections
        img = darsia.imread(
            path,
            transformations=self.corrections,
            date=date,
            reference_date=self.reference_date,
            name=path.name,
            **getattr(self, "raw_decoding", {}),
        )

        return img

    def read_registered_image(
        self, path: Path
    ) -> tuple[darsia.Image, darsia.ProjectiveRegistration]:
        """Read a raw photograph registered to the setup's reference and sampled once
        from its sensor onto the reference's grid (:func:`darsia.register_raw_frame`),
        then apply the rig's corrections.

        Parameters
        ----------
        path : Path
            Path to the raw file.

        Returns
        -------
        darsia.Image
            Image object with applied corrections.
        darsia.ProjectiveRegistration
            The photograph's homography from the reference's grid to its camera
            view, and the registration's quality.
        """
        registration = self.registration
        values, found = darsia.register_raw_frame(
            path, registration, self.raw_decoding["dtype"]
        )
        img = darsia.OpticalImage(
            values,
            transformations=self.corrections,
            date=self.experiment.get_datetime(path),
            reference_date=self.reference_date,
            name=path.name,
            color_space="RGB",
            dimensions=list(registration.dimensions),
            origin=list(registration.origin),
        )
        return img, found

    # ! ---- EXPERIMENT AND PROTOCOLS ----

    def load_experiment(self, experiment: darsia.ProtocolledExperiment) -> None:
        """Load experiment and associated protocols.

        This is required to read images and compute the injected mass correctly.

        Parameters
        ----------
        experiment : darsia.ProtocolledExperiment
            Experiment object containing imaging,
            injection, and pressure/temperature protocols.
        """
        self.experiment = experiment
        self.injection_protocol = experiment.injection_protocol
        self.pressure_temperature_protocol = experiment.pressure_temperature_protocol
        self.reference_date = experiment.experiment_start
        logger.info("Experiment and protocols loaded.")

    def update(self, path: Path) -> None:
        """Update the state of the rig based on the image path.

        This method updates the current date, time, pressure, and temperature
        of the rig object based on the provided image path.

        Parameters
        ----------
        path : Path
            Path to the image file.
        """
        # Convert date from path
        date = self.experiment.get_datetime(path)
        self.current_date = date
        self.current_time = (date - self.reference_date).total_seconds() / 3600.0
        state = self.pressure_temperature_protocol.get_state(date)
        self.current_pressure = state.pressure
        self.current_temperature = state.temperature
        self.setup_mass_analysis(
            atmospheric_pressure=self.current_pressure,
            atmospheric_temperature=self.current_temperature,
        )
        logger.info(f"State updated to {self.current_date}.")
