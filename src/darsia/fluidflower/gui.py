"""Compatibility imports for the modular FluidFlower Qt application.

New integrations should import page implementations from ``fluidflower.pages`` and
the application shell from ``fluidflower.ui``. This module keeps the original public
import locations stable for downstream users.
"""

from darsia.fluidflower.pages.calibration.page import CalibrationPage
from darsia.fluidflower.pages.data_integrity.page import (
    DataIntegrityPage,
    IntegrityWorker,
)
from darsia.fluidflower.pages.execution.page import (
    ExecutionHardwareWorker,
    ExecutionPage,
    ExecutionQueueWorker,
)
from darsia.fluidflower.pages.geometry_facies.page import (
    FaciesAnnotationExportWorker,
    FaciesModelTrainingWorker,
    GeometryFaciesPage,
    GeometryImageView,
    GeometryWorker,
)
from darsia.fluidflower.pages.image_correction.page import (
    CorrectionInventoryWorker,
    CorrectionWorker,
    ImageCorrectionPage,
    ZoomableImageView,
)
from darsia.fluidflower.pages.mass_analysis.page import MassAnalysisPage
from darsia.fluidflower.pages.optical_calibration.page import OpticalCalibrationPage
from darsia.fluidflower.pages.overflow.page import OverflowPage, OverflowScanWorker
from darsia.fluidflower.pages.overview.page import OverviewPage
from darsia.fluidflower.pages.phase_interpretation.page import PhaseInterpretationPage
from darsia.fluidflower.pages.post_analysis.page import PostAnalysisPage
from darsia.fluidflower.pages.reporting.page import ReportingPage
from darsia.fluidflower.pages.setup.incremental_runs import (
    AddRunsDialog,
    EstablishedSourceSetupEditor,
    RunImportPreviewDialog,
    UpdateRunSourcesDialog,
)
from darsia.fluidflower.pages.setup.steps.identity import SeriesIdentityPage
from darsia.fluidflower.pages.setup.steps.image_sources import (
    ImageSourcesPage,
    SourceDiscoveryWorker,
)
from darsia.fluidflower.pages.setup.steps.rig_protocol import RigProtocolPage
from darsia.fluidflower.pages.setup.steps.summary import SummaryPage
from darsia.fluidflower.pages.setup.wizard import NewSeriesWizard
from darsia.fluidflower.pages.temporal_sync.page import (
    TimelineEventDialog,
    TimelineImportDialog,
    TimelinePage,
    TimelinePlot,
)
from darsia.fluidflower.pages.wasserstein.page import WassersteinPage
from darsia.fluidflower.ui.analysis_domains import (
    AnalysisDomainDialog,
    AnalysisDomainImageView,
)
from darsia.fluidflower.ui.common import (
    DIRECTORY_DIALOG_OPTIONS,
    FILE_DIALOG_OPTIONS,
    INJECTOR_RECORD_ROLE,
    STATUS_COLORS,
    STATUS_LABELS,
    ModulePage,
    PathField,
)
from darsia.fluidflower.ui.home import HomePage
from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage
from darsia.fluidflower.ui.phase_model_studio import PhaseModelStudioWindow
from darsia.fluidflower.ui.shell import (
    SUBSECTION_CONTEXT_HELP,
    WORKSPACE_CONTEXT_HELP,
    FluidFlowerMainWindow,
    ProjectWorkspace,
    application_icon,
)

__all__ = [
    "DIRECTORY_DIALOG_OPTIONS",
    "FILE_DIALOG_OPTIONS",
    "INJECTOR_RECORD_ROLE",
    "STATUS_COLORS",
    "STATUS_LABELS",
    "SUBSECTION_CONTEXT_HELP",
    "WORKSPACE_CONTEXT_HELP",
    "AddRunsDialog",
    "AnalysisDomainDialog",
    "AnalysisDomainImageView",
    "CalibrationPage",
    "CorrectionInventoryWorker",
    "CorrectionWorker",
    "DataIntegrityPage",
    "EstablishedSourceSetupEditor",
    "ExecutionHardwareWorker",
    "ExecutionPage",
    "ExecutionQueueWorker",
    "FaciesAnnotationExportWorker",
    "FaciesModelTrainingWorker",
    "FluidFlowerMainWindow",
    "GeometryFaciesPage",
    "GeometryImageView",
    "GeometryWorker",
    "HomePage",
    "ImageCorrectionPage",
    "ImageSourcesPage",
    "IntegrityWorker",
    "MassAnalysisPage",
    "ModulePage",
    "NewSeriesWizard",
    "OpticalCalibrationPage",
    "OverflowPage",
    "OverflowScanWorker",
    "OverviewPage",
    "PathField",
    "PhaseInterpretationPage",
    "PhaseModelStudioWindow",
    "PostAnalysisPage",
    "ProjectWorkspace",
    "ReportingPage",
    "RigProtocolPage",
    "RunImportPreviewDialog",
    "SeriesIdentityPage",
    "SharedPhaseModelTrainingPage",
    "SourceDiscoveryWorker",
    "SummaryPage",
    "TimelineEventDialog",
    "TimelineImportDialog",
    "TimelinePage",
    "TimelinePlot",
    "UpdateRunSourcesDialog",
    "WassersteinPage",
    "ZoomableImageView",
    "application_icon",
]
