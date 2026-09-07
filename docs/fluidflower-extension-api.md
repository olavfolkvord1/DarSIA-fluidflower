# FluidFlower extension API

The FluidFlower application separates scientific services from Qt presentation.
An integrating application may therefore use a complete page, one nested tab, or
only the underlying service module. Importing the registry does not import Qt.

## Stable boundaries

Each top-level workflow page is described by a `FluidFlowerPagePlugin` in
`darsia.fluidflower.pages`. The descriptor records:

- a stable page key shared with the workflow dependency graph;
- a lazy page factory;
- the GUI-independent service modules used by the page;
- whether the DarSIA shell should provide scrolling.

Existing nested tabs are described by `FluidFlowerSubpagePlugin`. A subpage has
its own parent key, stable key, visible title, factory, and ordering value. This
allows a feature such as injection-point placement, active-learning review, or
model validation to be integrated incrementally.

The built-in page factories are lazy. PySide6 is required only when a Qt widget
is actually created. Scientific code should import the service module listed in
the descriptor rather than importing `gui.py`.

## Programmatic composition

```python
from darsia.fluidflower.page_registry import (
    FluidFlowerPagePlugin,
    FluidFlowerSubpagePlugin,
    default_page_registry,
    default_subpage_registry,
)

pages = default_page_registry(discover_external=False)
pages.register(
    FluidFlowerPagePlugin(
        key="geometry_facies",
        factory="my_darsia_gui.geometry:GeometryPage",
        service_modules=("darsia.fluidflower.geometry_facies",),
    ),
    replace_existing=True,
)

subpages = default_subpage_registry(discover_external=False)
subpages.register(
    FluidFlowerSubpagePlugin(
        parent_key="phase_interpretation",
        key="custom_review",
        title="Custom review",
        factory="my_darsia_gui.phase:build_review_tab",
        order=25,
    )
)
```

`ProjectWorkspace(project, page_registry=pages)` embeds the selected top-level
pages. `GeometryFaciesPage` and `PhaseInterpretationPage` accept a
`subpage_registry` when nested composition is needed directly.

## Installed drop-ins

An external Python package can be discovered without changing DarSIA by
publishing entry points:

```toml
[project.entry-points."darsia.fluidflower.pages"]
geometry = "my_darsia_gui.geometry:PAGE_PLUGIN"

[project.entry-points."darsia.fluidflower.subpages"]
phase-review = "my_darsia_gui.phase:SUBPAGE_PLUGINS"
```

A page entry point returns one `FluidFlowerPagePlugin`. A subpage entry point
returns one plugin or an iterable of `FluidFlowerSubpagePlugin` objects. A
matching stable key explicitly replaces the built-in registration; a new
subpage key adds another tab according to its ordering value. Discovery failures
are retained in `registry.discovery_errors` instead of silently changing the
workflow.

## Scientific compatibility

The page registry is presentation-only. It does not modify Simple Flash,
concentration analysis, image registration, 2D-to-3D conversion, or any other
DarSIA numerical method. Each scientific extension remains an explicit service
and must write its model, parameters, precision, source hashes, and upstream
revision identifiers into result provenance.

The original CPU/float64 DarSIA route remains the reference. New image
correction, learned phase/facies interpretation, Titration Flash, Optuna
objectives, or accelerator implementations must be selected explicitly and
validated separately. GUI packaging is not evidence that a scientific model is
approved.

## Source layout

Every built-in workflow page is a package with a Qt-free descriptor and a lazy
Qt implementation:

```text
darsia/fluidflower/pages/<page>/
    __init__.py       # PAGE_PLUGIN and optional SUBPAGE_PLUGINS
    page.py           # Qt page implementation
    subpages/         # independently replaceable nested tabs, when applicable
```

Shared controls and the application shell live in `darsia.fluidflower.ui`.
`darsia.fluidflower.gui` is now only a backward-compatible re-export module;
new integrations must not add implementations there. Consequently, Jakub can
copy one built-in page package, depend only on its listed service modules, or
replace it through an entry point without adopting unrelated pages.

Geometry/facies and phase interpretation demonstrate nested composition. Each
visible tab is built by a separate module under `subpages`, while the parent
page owns the shared state and workflow actions. This preserves one project
transaction while allowing tabs to be integrated, replaced, and tested in
small batches. The new-series wizard follows the same registry contract: its
identity, protocol, image-source, overflow, and summary steps live under
`pages/setup/steps` and can be replaced independently while retaining stable
wizard attributes.

Cross-series optical training is a project-independent scientific service under
`darsia.fluidflower.phase_training`. Its trainer, model, optimization primitives,
and review-region extraction are authoritative there; historical FF_AC scripts
are compatibility aliases only. Image-change channels are supplied through the
Qt-free `ChangeEvidenceProvider` contract in `phase_features`, so another DarSIA
detector can be added and ablated without forking the trainer. The start-screen
training page is only a client of these services and does not contain numerical
model logic.

Ordinary deployment review uses two Qt-free, append-only contracts. A
`PhaseInterpretationRevisionStore` revision records the selected model and a
visible-2D approval for every experiment. A `CalibrationResultReviewStore`
revision then binds per-run concentration-map and mass-plot decisions to the
exact calibration-result hash. The calibration and mass-analysis pages are UI
clients of these contracts; alternative GUIs and command-line workflows can
validate and store the same records without importing Qt.

Measured indicator response follows the same separation. The Qt-independent
service `darsia.fluidflower.indicator_response` owns robust buffer-image
sampling, full-colour features, monotone colour-to-pH fitting, holdout
validation, uncertainty estimates, source fingerprints, and immutable model
revisions. It exposes array-level functions for scientific tests and a
project-level `IndicatorResponseModelStore` for applications and scripts.

The optional Qt component
`darsia.fluidflower.pages.calibration.indicator_response_panel.IndicatorResponsePanel`
only collects reviewed chemistry values, starts a background fit, and attaches
the selected model reference to a calibration plan. The calibration service
then freezes the measured optical response and limits Optuna to explicitly
reviewed chemistry bounds. An integrating GUI can therefore embed this panel,
replace it, or call the service directly without adopting the FluidFlower
application shell.

Frame screening is likewise split at a stable boundary. The Qt-free module
`darsia.fluidflower.frame_quality` computes run-local raw and corrected temporal
signatures, stores immutable assessments, appends human decisions, and derives a
content-addressed analysis blacklist. Its automatic assessment is advisory and
cannot remove a frame. Consumers can call `FrameQualityStore.filter_frames()`
and retain `FrameQualityStore.dependency_reference()` in provenance. The optional
Qt class
`darsia.fluidflower.pages.image_correction.frame_quality_panel.FrameQualityPanel`
adds the timeline, filters, full-resolution inspection, and explicit keep/exclude
actions. It can therefore be integrated independently of the complete Image
correction page.
