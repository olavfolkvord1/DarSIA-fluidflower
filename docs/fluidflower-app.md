# FluidFlower Experiments app

The FluidFlower app is a Qt front end over a GUI-independent DarSIA project
model. It is intended to provide one traceable workflow from experimental-series
setup through image correction, phase interpretation, calibration, mass
analysis, Wasserstein comparison, and post-analysis exports.

## Launch

Install DarSIA with the optional GUI dependency and run:

```text
darsia-fluidflower
```

Within the FF_AC development checkout on Windows, use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\launch_darsia_fluidflower_app.ps1
```

The launcher uses `pythonw.exe`, so the interactive Qt application does not open
a separate terminal window.

Transferable optical phase-model development is temporarily hosted in a separate
Qt application:

```text
darsia-fluidflower-phase-studio
```

Within the FF_AC development checkout on Windows, use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\launch_darsia_phase_model_studio.ps1
```

The studio embeds the same `SharedPhaseModelTrainingPage` intended for final
DarSIA integration. It uses the same review-session format, full-resolution
evidence viewer, feature cache, DarSIA image-change channels, immutable model
bundles, and complete-run holdout checks. Separating the top-level process does
not fork the scientific workflow or require later data migration; it only lets
model development continue independently while the production project shell is
completed.

## Modular integration

Every top-level workflow menu is now resolved through a Qt-independent page
registry. Each built-in menu is a self-contained package under
`darsia.fluidflower.pages`, with a Qt-free descriptor and a lazily imported Qt
implementation. The existing tabs in Geometry and facies and Phase
interpretation are separate modules behind the corresponding subpage registry.
Installed packages can add or explicitly replace either level without editing
the application shell.

Each descriptor lists the GUI-independent service modules behind the page, so a
different DarSIA GUI can integrate the scientific workflow without adopting our
Qt layout. The complete public contract and entry-point examples are documented
in [fluidflower-extension-api.md](fluidflower-extension-api.md).

`darsia.fluidflower.gui` remains as a compatibility import for existing users;
the actual navigation shell, shared controls, pages, and nested tabs no longer
reside in that file.

## Implemented sequences

The first sequence provides:

- A `FluidFlower Experiments` home screen.
- A setup wizard for FF_BM, FF_AC, FF_UM, and custom series.
- A4, Medium, Large, and custom rig choices.
- Optional protocol, experiment-image, baseline, clear-water, ColorChecker,
  measured-pH indicator-buffer, other known full-rig color-state, and overflow
  source candidates. The active overflow source is selected later, after data
  integrity has been reviewed. A series-level scan first proposes the immediate child
  folders as runs; users can exclude unrelated folders and edit run IDs.
- Setup drafts autosave independently of the final project manifest, so a long
  import can be resumed without rewriting any source. Image discovery supports
  one experiment folder or run-specific subfolders. One representative run can
  define reusable roles for CO2 image sequences, clear-water/indicator flushes,
  known-pH buffers, non-indicator full-rig color states, and overflow images or
  tables.
  Each indicator reference retains its image, measured pH or state label, and
  series/run scope. DarSIA applies those relative patterns to all
  selected runs, proposes the first frame in the CO2 injection sequence as the
  default baseline, and lets exceptional runs be categorized folder by folder
  before the user's final source-to-run mapping is frozen.
- Any number of folders may share the same source role. CO2 injection and
  continuation folders form one explicit ordered sequence per run, with image
  counts and first/last files retained for every segment. Users can bulk-assign
  a category and move selected CO2 segments earlier or later; the baseline is
  always taken from the first ordered injection segment.
- The home screen lists every incomplete setup with its last saved time, saved
  wizard page, and planned destination. A selected draft can be continued from
  that page or discarded. Discarding removes it from the active list but keeps a
  recovery copy and never touches source data or an established project.
- The same home screen can append experiments to a selected established series.
  DarSIA rescans the chosen parent folder, locks folders already registered in
  the project, reuses the reviewed run-layout template, and requires an explicit
  source review plus a final read-only preview before committing anything. Run
  IDs and run-root paths are checked for duplicates.
- An independent action updates the active source mapping for one existing run.
  The previous mapping remains in that run's history. Both operations snapshot
  `series.json` and `source_manifest.json` before an atomic replacement, retain
  existing results, and record a complete audit event. Generated `runs/<run-id>`
  indexes make the active mapping inspectable without becoming a second editable
  source of truth.
- New protocol templates are always editable Excel workbooks: one workbook for
  all runs, one sheet per run, or one workbook per run. CSV and TSV remain
  supported for importing historical data. A downloadable universal workbook
  template defines experiment anchors,
  injection changes, image acquisition, overflow, flushing, and interventions;
  FF_UM data is an example of this schema rather than a hard-coded template.
- A local registry for established projects.
- A versioned project manifest and immutable source manifest.
- Project storage and raw experiment sources are deliberately separate. The
  first setup page chooses the parent folder in which DarSIA creates a dedicated
  project. Configuration, review records, cache, and generated results live
  there by default; the experimental-series folder selected on the source page
  is only a read-only reference to existing camera, protocol, and overflow data.
- Generated products default to `results/<module>` below the project root.
  **Execution and workers** can override the destination independently for each
  analysis module, so large mass-analysis NPZ products may be published to a
  separate local disk or NAS without moving the project or changing raw sources.
- A modular project workspace with explicit dependencies and quality gates.
- A context-aware information button in the project header explains the active
  main module and, for complex work surfaces, the selected subsection. This
  keeps routine pages compact while leaving room for fixed introductory text on
  selected pages that need additional guidance.
- Automatic downstream invalidation when an approved upstream interpretation is
  reopened or changed.

The second sequence adds two operational work surfaces:

- **Image correction** owns the correction-model choice and supports either a
  side-by-side `uniform24` preview or an explicit no-correction identity control.
  It runs on the approved geometrically registered grid and before color paths,
  phase interpretation, and chemistry calibration. Uniform24 uses six neutral
  ColorChecker fields for the tone response, all 24 fields for a robust
  regularized affine residual, explicit transform bounds, and field-wise
  before/after QC. The detected or reused ColorChecker region is outlined on
  both previews so the user can verify that the physical chart was found. The
  chart may be embedded in the rig/source images; a
  dedicated ColorChecker folder is optional. Images of the experiment's own
  indicator at measured pH are kept as chemistry-response references, while a
  different known liquid or color state is stored as a labelled alternate
  optical control. Neither category is silently substituted for a light/color target. The
  selected policy, ColorChecker ROI when relevant, reviewer, reason, and
  diagnostics are stored as an append-only revision.
- The setup UI defines these source roles explicitly. A baseline is the rig
  filled with the experiment's actual pH-indicator solution before CO2 is
  present, and normally is the first CO2-series frame. A clear-water reference
  is indicator-free and CO2-free, normally at the clear end of a wash/flush
  sequence. Setup only records whether the physical 24-patch ColorChecker is
  visible in source frames, stored in a dedicated folder, or unavailable. It does
  not select a correction algorithm. That explicit choice is made and validated
  under **Image correction**. The current Uniform24 implementation is a global
  transform rather than a spatial illumination-field model. Other known rig colors
  remain labelled optical controls and never enter the measured pH-response curve.
- Correction revisions and phase/color-path revisions are append-only and linked
  by revision IDs and SHA-256 fingerprints. Changing the active correction marks
  phase interpretation, calibration, mass analysis, Wasserstein analysis, and
  post-analysis as outdated without deleting earlier artifacts. Before mass
  analysis, the user must explicitly select the compatible correction plus
  color-path branch. This branch identity is copied into mass, post-analysis, and
  Wasserstein result provenance so outputs from alternative corrections remain
  distinguishable and comparable.
- **Frame quality and exclusion review** runs on the immutable image inventory
  attached to the active correction revision. It compares compact spatial,
  exposure, colour-balance, sharpness, and transient signatures independently
  for every experiment, before and after correction. This exposes abrupt light
  changes, one-sided lamp failures, temporary camera obstruction, severe blur,
  and correction residuals without confusing a repairable raw-image problem with
  an unusable corrected frame. Automatic results are advisory: no source image is
  modified or excluded. A named reviewer must explicitly confirm an exclusion,
  which is appended to the decision history and materialized as JSON plus CSV
  blacklists. Changing the blacklist invalidates dependent phase and mass work;
  each downstream branch records the exact blacklist fingerprint it consumed.
  Read concurrency is backed by a bounded dynamic queue. The desired worker count
  remains editable during a scan: increases submit additional frames after the
  next completion, while decreases allow current reads to finish before reducing
  concurrency. Completed signatures and progress are retained throughout.
- **Measured indicator response** is a separate, optional calibration service.
  It registers every known-pH buffer on the approved geometry, applies the same
  saved radiometric policy as experiment frames, and extracts robust full-color
  features globally and per enabled facies. An ordered polyline through linear
  RGB, Lab, chromaticity, and channel-ratio features gives a monotone pH response
  without assuming a blue-yellow direction. Leave-one-reference-image-out tests
  provide pH error and uncertainty diagnostics. The pH estimate is converted to
  DIC with the same BTB/carbonate charge balance used by `TitrationFlash`.
  Chemistry defaults remain explicitly provisional until a reviewer confirms
  measured or protocol-derived values and uncertainty bounds.
- **Temporal synchronization** imports CSV, TSV, or Excel protocol and
  measurement tables, displays all sources on one experiment-time axis, and
  allows direct local adjustment. Raw and effective values remain visible next
  to each other. Reverting an adjustment appends a new record; it does not erase
  history.
- New projects use the version-2 universal protocol template. Each run records
  its experimental date and an explicit `experiment_start` defining `t=0`.
  Actual camera interval changes identify the first image in each interval, and
  injections can be split by injection-point ID into ramp-up, steady-rate, and
  ramp-down segments. Full dates preserve intervals spanning several days.
- Time-stamped pH measurements distinguish indicator solution from the rig
  immediately before injection and retain method, instrument, temperature, and
  uncertainty when available. Temperature, pressure, overflow, flushing,
  sampling, camera/lighting changes, and manual interventions use the same
  event schema. Timezone is optional run metadata rather than a required value
  on every event.
- If both dated timestamps and relative hours are supplied, the approved
  experiment origin controls the effective timeline while both originals are
  retained. A disagreement greater than one second is reported by timeline QC.

The third sequence adds an operational **Data and integrity** work surface:

- Fast scans read filesystem metadata, image headers, and EXIF timestamps in
  parallel. Sampled hashes are computed only for files whose sizes make them
  possible duplicates; matching candidates are then hashed fully.
- The application detects logical and physical CPU cores plus available RAM and
  derives both a conservative starting value and a machine-specific upper limit
  for parallel reads. The user can change the requested read-worker count while
  a scan is running; increases or decreases take effect as active file reads
  finish. The selected values and every change are retained in report provenance.
- Returning to the project list does not destroy an open project workspace.
  Integrity scans continue in the background, and reopening the project restores
  the same progress view. The application cancels and joins the scanner safely
  only when the application itself is closed or the user selects Cancel.
- Page scrolling cannot silently change combo-box or numeric values. Closed
  drop-downs ignore the mouse wheel, and a numeric field accepts wheel input only
  after that exact field has been clicked directly.
- Verified scans additionally validate every complete image stream and compute
  a full SHA-256 digest for every file.
- Missing sources, unreadable images, duplicate files, resolution outliers,
  sequence gaps, and timestamp problems are reported per registered source.
- Transient network failures are retried. Every completed report and its
  compressed file inventory are immutable; only a small `current.json` pointer
  advances to the latest report.
- A report containing errors blocks the quality gate. Warnings remain visible
  for reviewer assessment without silently changing source data.
- Sources belonging to a newly appended or updated run are selected
  automatically when the established series already has an approved integrity
  gate. Their scan and approval are run-specific: previously approved runs and
  global downstream results are not invalidated merely because a later
  experiment was added.

The next module is **Overflow measurements**:

- Setup discovery is not treated as a measurement. For every experiment, the
  user explicitly chooses no overflow, camera images of a scale display, or an
  existing CSV/Excel table.
- `Scan source` inventories and validates the selected folders or workbook. It
  does not run OCR. `Open scale-image reader` starts the established OCR and
  manual-review workflow for the selected experiment without opening a console
  window.
- Multiple scale-image folders can belong to one experiment. Their files are
  read as one time-ordered series while retaining their original paths.
- The reviewed workbook produced from camera images remains inside
  `results/overflow/scale_review/<run>`. It can be selected as the active table,
  or the user can later choose another Excel file.
- Changing from images to a table, or back again, creates a new append-only
  overflow revision. Earlier decisions and source inventories remain available;
  raw source files are never edited or deleted.
- Temporal synchronization cannot be approved until the active overflow
  decision has passed this quality gate, including an explicit `not used`
  decision for experiments without overflow data.

The fourth sequence adds an operational **Geometry and facies** work surface:

- Existing DarSIA rig artifacts can be imported without modifying the source
  files. Baseline, depth, detailed labels, an optional facies map, material
  properties, and protocol injection-point coordinates remain explicitly
  referenced.
- Detailed labels retain the geometry and curvature of every physical layer.
  Equal materials may either share one facies definition or remain independent
  per layer. Analysis roles distinguish porous media, seals, free water, frame,
  ColorChecker, and excluded regions.
- Baseline, labels, facies, depth, porosity, and permeability can be inspected
  as synchronized overlays. Protocols identify injection points as `ip1`, `ip2`,
  and so on. Their x/y coordinates are optional in Excel: supplied coordinates
  appear in the GUI, while blank coordinates can be entered numerically or
  placed directly on a baseline or early CO2 image. A GUI override preserves
  the Excel position and raises a visible discrepancy warning. No z-coordinate
  is requested for this 2D placement.
- Any selected experiment frame can be passed through the saved rig shape and
  color corrections and used as the overlay background. A strict shape check
  prevents labels from being shown over an unregistered image grid.
- For a rig without existing labels, an unsupervised optical draft combines
  SLIC superpixels with clustering of Lab color, local texture, image gradient,
  and vertical position. It proposes detailed connected regions and shared
  optical classes. Every proposal starts excluded, with no inferred porosity,
  permeability, or physical role; expert review is mandatory before approval.
- The **Material examples** tab provides a supervised alternative for a new or
  unfamiliar rig. The user defines and names the available sand types, assigns
  each a porous-medium or seal role, and marks tight homogeneous interior boxes.
  The default readiness gate requires eight examples and at least three spatial
  areas per material; this prevents one local lighting patch from defining a
  complete material class.
- A new series starts with guided examples. Once a transferable facies model is
  mature, the same page can instead make a hash-verified blind prediction before
  any local examples are supplied. The user can then add corrections and compare
  them with the blind result rather than leaking local guidance into the transfer
  test.
- Both modes use the shared full RGB, Lab, chromaticity, texture, and gradient
  feature pipeline. Probabilities are regularized within optical superpixels,
  then connected components propose separate physical layers that may share one
  named material. Every generated layer is disabled and marked unreviewed until
  the researcher assigns properties and passes the ordinary geometry gate.
- Material definitions and boxes autosave atomically to a current document and
  immutable revision files. Prediction artifacts include source hashes,
  full-resolution facies, layer and uncertainty maps, and never modify imported
  images or an approved geometry revision.
- Clicking a region in the image selects its physical layer in the editor.
  Material properties and roles can be propagated to all regions sharing the
  same facies while their separate geometry and curvature remain intact.
- Layer boundaries can be corrected with polygons or a physical-radius brush.
  Screen coordinates are converted to SI coordinates and rasterized against
  the original full-resolution label map, so zoom and preview downsampling do
  not alter the edit.
- Every edit, undo, and redo is autosaved as an append-only event. The imported
  label map remains immutable. Materialization creates a new `labels.npz` and
  records both its base map and complete event log in the next geometry
  revision's provenance.
- The **Training data** tab exports a saved geometry revision as an immutable,
  portable facies-annotation bundle. It contains the corrected baseline,
  detailed layer labels, shared facies, semantic roles, depth, context and
  training masks, material properties, a visual preview, and source hashes.
- Exports are assigned to `train`, `validation`, `holdout`, or a neutral `pool`.
  Versioned split-plan JSON files can assign pooled leakage groups to a campaign
  without copying full-resolution bundles. A leakage group
  may occur in only one split, including when multiple project stores are
  collected. This prevents frames sharing one rig or acquisition context from
  inflating transfer performance.
- Bundles from an approved geometry gate are eligible for model training.
  Valid but unapproved revisions are retained as explicit drafts and excluded
  by the default collector. Identical exports are idempotent.
- The same tab can train a first supervised research baseline from one or more
  annotation stores. A chemistry-agnostic feature contract combines full RGB,
  Lab, chromaticity, channel ratios, multiscale intensity, texture, and image
  gradients. Absolute image position is disabled by default to reduce geometry
  leakage between FluidFlower layouts.
- The baseline has three independent heads. The universal head predicts stable
  spatial roles such as porous medium, seal, and free water. The material head
  uses canonical expert material names; generic names such as `Layer 7` remain
  provisional and are excluded from cross-series material training. The
  hydraulic head predicts fixed permeability intervals in mD; expert
  permeability is target metadata and is never used as an input feature.
- The estimator is an explicitly CPU-based ExtraTrees baseline. Probability
  calibration is enabled only when group-isolated folds contain every class.
  Otherwise the uncalibrated status and reason are recorded in the model card.
- Validation and holdout predictions are generated tile-wise at full source
  resolution. Each immutable model contains class maps, confidence and entropy
  maps, per-class IoU/F1, boundary F1, unseen-material coverage, previews, a
  human-readable model card, complete bundle/split provenance, and software
  versions. Holdout groups are never used for fitting or calibration.
- This baseline interprets facies in the registered 2D camera plane. It does
  not infer CO2 phase, concentration, mass, depth, or 3D distribution; those
  remain separate downstream DarSIA modules.
- Validation checks array alignment, complete label mapping, physical material
  properties, depth, injection-point bounds, and shared-facies consistency. It also
  reports active area and estimated pore volume.
- The user explicitly chooses one approved geometry for the whole series,
  reviewed geometry overrides for selected runs, or mandatory review for every
  listed run. Attaching a run map for inspection does not make it authoritative.
- When comparable registered layer-label maps exist, an optional advisory screen
  calculates symmetric boundary displacement in physical units. The warning
  threshold is expressed as a fraction of rig height together with a minimum
  affected-boundary fraction, so it remains meaningful across rig resolutions.
  A warning asks for review; it never selects or edits a geometry.
- Each selected override references a registered source image and optional
  detailed-label/facies maps on the same corrected grid. Porosity and
  permeability stay inherited from the approved series materials and are never
  changed automatically from an optical displacement.
- Per-run geometry sources are tagged as clear-water, indicator-filled, or
  unknown. Indicator-filled sources carry a residual-indicator/carryover warning,
  and every accepted override becomes a separate fingerprinted artifact.
- Saving creates an append-only geometry revision, a canonical DarSIA facies
  artifact, and a byte-for-byte copy of the approved depth map. Downstream
  workflows read the revision-local depth artifact rather than the mutable or
  externally mounted source. The original source path and both source and
  artifact SHA-256 fingerprints remain in the revision provenance. Approval is
  blocked until Data and integrity is approved and the current revision has no
  validation errors.

The fifth sequence adds an operational **Phase interpretation** surface:

- A portable cross-series optical model bundle can be inspected and registered
  without rewriting it. Its model heads, feature contracts, source annotations,
  inventory, and SHA-256 fingerprints are stored in an append-only phase-learning
  plan.
- FF_AC and FF_UM evidence can share the same hierarchical optical model. The
  first head detects CO2-related optical change and the second separates free gas
  within that support. Mutually exclusive water, aqueous-CO2, and free-gas
  probabilities are then produced on the registered 2D image grid.
- The generic feature contract obtains registered reference and previous-frame
  change evidence through an explicit DarSIA `ChangeEvidenceProvider`, together
  with chromaticity, channel ratios, local texture, and frame dynamics. This is
  one authoritative implementation rather than a second feature copy in a
  project script. An optional chemistry adapter may add indicator-specific
  evidence without changing the generic model.

- Existing phase-campaign manifests and full-resolution review packages are
  registered by reference. DarSIA does not move or rewrite campaign products.
- Every candidate is listed by run, model role, family, candidate identifier,
  physics status, and deployment status. One final candidate can be assigned
  independently to each run.
- Saved manual-review JSON files can be attached to the revision. Their absolute
  paths, sizes, modification times, and SHA-256 hashes are frozen for audit.
- **Active learning review** presents model-proposed native-resolution regions
  near uncertain phase decisions. Predictive entropy is balanced across hidden
  phase hypotheses, facies, time, and position; the proposal is not revealed
  until the reviewer answers. Existing answers are retained and new rounds add
  distinct regions. Manual boxes and freehand regions remain available.
- **Model validation** presents one selected model on the complete corrected 2D
  image sequence. The ordinary inspection profile removes A/B comparison,
  verdicts, boxes, and annotation controls while retaining full-resolution zoom
  and pan, the pre-injection reference, timeline, phase-map and layer controls,
  diagnostics, playback, and injection-point markers. Approval is recorded per
  experiment in the project. A separate contribution action opens the editable
  active-learning surface only when localized correction evidence is needed.
- **Shared model training** is a separate research application rather than a tab
  inside the production project application. It freezes one or more FF_AC/FF_UM review
  bundles into a new immutable model-version directory before starting the
  hierarchical CO2-presence and gas-within-CO2 trainer. Complete runs remain
  isolated for validation. An optional non-deployable ablation repeats fitting
  without the structured DarSIA change channels and reports the holdout benefit.
  A successful candidate is inspected but is never silently deployed over the
  registered model. Normal project users register an approved model, contribute
  targeted evidence when necessary, and verify its predictions locally.
- Each append-only revision records the exact geometry revision, image-correction
  revision, effective-timeline digest, and upstream quality-gate states.
- Model registration and campaign intake are explicitly restricted to the
  visible registered 2D camera plane. The screen can execute versioned optical
  training, but does not infer concentration, mass, depth, or 3D distribution.
  Those remain separate, experiment-specific DarSIA stages.
- A phase quality gate cannot be approved until a valid revision exists, one
  candidate has been selected for every campaign run, and the visible 2D phase
  map has been reviewed and approved for every experiment. Changing a selected
  candidate invalidates that experiment's prior approval. Unavailable review folders
  remain warnings because a network share may be temporarily offline; missing or
  malformed campaign manifests are blocking errors.

The sixth sequence adds an execution-free **Calibration** planning surface:

- A plan is bound to the exact current phase revision and its SHA-256 digest.
- Simple Flash, Titration, or a balanced comparison can be selected together
  with explicit initial-pH bounds, temperature, and pressure.
- Every physical objective remains explicit, including disabled terms. Each
  active term has a recorded weight, direction, and normalization rule.
- Every searchable Optuna parameter is equally explicit: bounds, linear/log
  scale, sharing scope, applicable chemistry models, physical unit, and
  transformation are part of the immutable contract. Aqueous and free-gas
  transition windows, maximum gas saturation, initial pH, and titration blend
  are visible in the GUI.
- A deployable measured indicator-response model can replace the empirical
  aqueous optical transition. Its fitted color path and feature scaling are
  immutable during Optuna optimization. Only initial pH, alkalinity, indicator
  concentration, and aqueous saturation concentration with non-zero, reviewed
  uncertainty intervals can become searchable, and their Optuna bounds must
  remain inside those intervals. Fixed measured values stay fixed.
- The total-CO2 support threshold remains a separate fixed transfer contract.
  Spatial diffusion is listed but disabled until a spatial evaluator can apply
  it while conserving total mass.
- A plan may be saved before evaluation data is prepared. It becomes
  execution-ready only when an attached, hash-verified evaluator manifest
  matches the selected phase revision and supports every enabled objective and
  parameter. Unsupported terms are never silently omitted.
- Optimization and final winner selection share one immutable objective-contract
  digest. A plan with different selection logic is rejected.
- Trial budget, deterministic seed, worker count, and the planned
  `cpu-float32`, `cpu-float64`, CUDA/OpenCL `float32` or `float64`, or
  `metal-float32` backend
  are recorded. Precision and analysis-grid resolution are independent choices.
  CPU float64 remains the authoritative parity reference with explicit absolute
  and relative tolerances.
- Depth and porosity stay fixed to the approved geometry revision. Conversion
  from the visible 2D slice to concentration, saturation, and 3D mass remains a
  downstream mass-analysis responsibility.
- A second, separate result-review section becomes active after optimization.
  For every experiment the reviewer inspects a concentration map together with
  the injected-versus-detected aqueous, free-gas, and total-mass plot. The review
  is bound to the exact calibration-result hash. A changed result invalidates the
  prior decision, and a run may explicitly be sent back for recalibration.
- The Calibration quality gate cannot be approved until every experiment has an
  approved result review. This is the boundary between parameter estimation and
  production mass analysis.
- Saving is append-only and never starts Optuna, phase training, facies training,
  mass analysis, or backend execution.

The response fitter is implemented in the Qt-independent
`darsia.fluidflower.indicator_response` service. The GUI is a separate drop-in
`IndicatorResponsePanel`, and the calibration page only composes the panel and
its public plan constraint. This keeps the scientific model, persistence, and
Qt integration independently reusable.

The seventh sequence adds an execution-free **Mass analysis** task surface:

- Every run is bound to the phase candidate selected by the current phase
  revision. The task is also bound to the exact calibration-plan and objective
  fingerprints.
- A calibration-result manifest is a separate required production input. It
  identifies the selected Simple Flash or Titration model, a hash-verified
  parameter artifact, and successful tolerance-based parity against
  `cpu-float64`. A missing result leaves a valid draft; an inconsistent result is
  a blocking error.
- The mass task is also bound to the immutable calibration-result review. The
  task cannot become execution-ready unless every experiment's concentration
  map and mass plot were approved against the same result manifest.
- A separate prepared-input manifest binds every selected run and phase candidate
  to one synchronized time grid and six hash-indexed numerical products:
  corrected optical acidification, visible CO2 probability, free-gas
  probability, local depth, porosity, and permeability. It also records injected
  mass and pressure/temperature state per frame. Rendered overlays are never used
  as numerical input.
- Representative images are selected against the same target times for every
  run, within an explicit drift tolerance, and published chronologically.
- `reference`, `production`, and `exploratory` quality profiles are editable
  presets. The user independently chooses backend precision and a native,
  scale-factor, or maximum-dimension numerical grid. Any reduced grid is marked
  as a scientific approximation and stores both source and effective shapes.
  Preview resolution remains separate and is never used as numerical input.
- A deterministic time/publication budget preserves requested landmark times,
  constrains the experiment-time window, and can cap images, solver tasks, or
  publication size. Exact selected source indices and times are frozen before
  execution.
- The selected phase candidate supplies spatial phase support. The approved
  corrected optical signal remains continuous inside that support, and DarSIA's
  CO2 mass model computes concentration, saturation, density, and solubility.
- Aqueous CO2 and free gas are mutually exclusive fractions of the same pore
  volume. Physical volume uses pixel area, approved local depth, and approved
  local porosity.
- Raw detected total, aqueous, and free-gas mass fields are authoritative.
  Injected mass remains a comparison reference; an injected-mass projection is
  exported only as a clearly named diagnostic and mass conservation is never
  silently forced.
- The mass result also exports a static, dimensionless inverse-relative-
  permeability `transport_weight` field, synchronized injected-mass references,
  and explicit reference plume radii. Downstream comparison code therefore does
  not reconstruct geometry, permeability, or chemistry from hidden defaults.
- Numerical products use NPZ with the dtype encoded by the selected backend.
  CPU can explicitly use float32 on memory- or bandwidth-limited machines;
  this is a non-reference production choice with the same parity gate as a GPU.
  CUDA and compatible OpenCL devices may expose float32 and float64 variants;
  Metal remains float32. Every accelerated variant requires CPU-float64 parity.
- A mass-preserving Wasserstein cache can be requested only with an explicit
  physical cell size. The old dimensionless `0.1` factor is not accepted as a
  cross-rig resolution contract.
- Worker count, one-image bounded prefetch, CSV checkpoints, local read-through
  cache, and atomic publication are part of the task fingerprint.
- Plans are append-only. Approval is blocked until the calibration result and all
  upstream quality gates are ready, and saving never starts image analysis.

The built-in `mass-analysis-cpu-float64-v1` adapter is the authoritative generic
reference implementation. The separate `mass-analysis-cpu-float32-v1` adapter
provides a lower-memory CPU path without changing the reference contract. The
learned phase model supplies visible 2D support
and phase probability, while DarSIA supplies the Simple Flash or Titration
transfer, CO2 solubility, gas density, exclusive pore-volume partition, and
physical 2D-to-3D conversion. It writes independently hash-verified total,
aqueous, and free-gas mass fields, concentration and saturation fields,
permeability-derived transport weights, integrated tables, diagnostic injection
comparisons, and complete provenance. It never rescales a field to injected mass.

`MassInputRunWriter` is the corresponding streaming preparation boundary. It
writes corrected optical acidification, total visible-CO2 probability, and
free-gas probability frame by frame on the native registered grid, plus one
approved depth, porosity, and permeability field per run. Every frame and index
is independently hashed before the final manifest is exposed. Free-gas support
outside total CO2 support is rejected, and non-finite geometry inside the active
domain is a blocking error. Optional JPEG previews remain visually useful but
cannot be substituted for the numerical products.

Experiment-specific readers and frozen phase models remain outside this generic
writer. The FF_AC/FF_UM reference bridge in
`scripts/materialize_phase_mass_inputs.py` runs the reviewed cross-series phase
model against the established registered and Uniform24-corrected images, then
feeds the resulting native arrays to this contract. This keeps historical paths,
legacy rig loaders, and campaign model locations out of the reusable DarSIA
core while preserving their hashes in transitive provenance.

The eighth sequence adds an execution-free **Wasserstein analysis** planning
surface:

- The plan is bound to the exact current mass-analysis plan and immutable mass
  task digest. A separate completed mass-result manifest must contain every run,
  one shared synchronized target-time grid, physical rig dimensions, native cell
  size, a whole-domain field, synchronized injection/plume references, and
  hash-verified indexes for total, aqueous, free-gas, and transport-weight fields.
- Analysis boxes are drawn interactively on a registered preview that uses the
  current shape correction, crop, large-rig curvature unwrapping, and saved
  ColorChecker correction. Coordinates may instead be imported from JSON, CSV,
  or TSV. Every box stores normalized image bounds, full corrected-grid pixel
  bounds, physical SI bounds, and exact geometry/correction provenance in an
  append-only revision.
- The user explicitly selects participating runs and then the unordered run
  pairs to compare. Total, aqueous, and free-gas distributions can be selected
  independently over the whole active rig and any enabled custom boxes. Custom
  boxes are cropped from the authoritative full mass fields at comparison time.
- Spatial resolution is expressed as a common physical cell size in metres. The
  old dimensionless `0.1` image factor is rejected, conservative area resampling
  preserves integrated mass, and a plan finer than the coarsest source grid is
  blocked because it would imply unsupported spatial precision.
- The variational L1 transport solver uses the existing cell-based,
  Raviart-Thomas, inverse-relative-permeability formulation. Each pair is scaled
  to its mean detected mass only inside the balanced transport solve. Raw masses,
  absolute mass difference, and relative mass difference remain required output
  columns and are never overwritten by that normalization.
- Raw transport cost, mass-normalized distance in metres, the FF_UM-compatible
  rig normalization, a rig-diagonal normalization, and the plume-based
  normalization are recorded together. This makes the smaller FF_AC rig
  comparable without confusing physical distance, mass difference, and a
  dimensionless rig-relative score.
- The currently validated solver is explicitly `cpu-float64`, with 12 workers by
  default. Tasks are ordered by target time so loaded mass fields can be reused,
  cached locally, checkpointed individually, retried after transient storage
  outages, resumed by immutable identity, and published atomically.
- A missing or temporarily unavailable mass result leaves a valid draft. A
  mismatched plan, changed file hash, inconsistent time grid, unavailable domain,
  or incomplete run is blocking. Saving never loads a mass field or starts a
  Wasserstein solve.

The built-in `wasserstein-cpu-float64-v1` adapter executes that contract with
DarSIA's weighted Bregman solver. Mass-per-pixel fields are conservatively
downsampled and converted to physical mass density before transport. Permeability
weights use nearest-neighbour reduction to retain facies barriers. Every
pair/time/domain/component row is saved independently, while final CSV, XLSX,
intermediate index, result manifest, and provenance files are published together.
Solver convergence and warnings are retained per row rather than hidden.

The ninth sequence adds an execution-free **Post-analysis** planning surface:

- Runs and products are selected explicitly. Products include filled phase-map
  movies on corrected or black backgrounds, contour movies and contact sheets,
  whole-rig and analysis-box mass curves, cross-run grids, per-frame diagnostics,
  integrated mass tables, and complete provenance.
- Every movie uses all mass-analysis frames in chronological order and starts
  with a pre-injection reference. The image background is the approved corrected,
  cropped, registered, and, where required, curvature-unwrapped experiment frame.
- Filled phase maps are mutually exclusive. Aqueous `c/csat` and free-gas
  saturation contour levels remain explicit display settings; they do not alter
  the saved concentration, saturation, or mass fields.
- Raw detected total, aqueous, and free-gas masses remain authoritative.
  Synchronized injected mass is a comparison curve only; post-analysis cannot
  force conservation or project detected mass onto the injection protocol.
- The plan references the exact mass-task digest, mass-result manifest, analysis
  box revision, and hash-verified phase/image indexes. A mass-only product set can
  be prepared without requesting phase images.
- Frame preparation is planned as a bounded CPU-parallel stage. Hardware video
  encoding may be selected automatically when available, while CPU software
  encoding remains explicit. Local cache, per-run checkpoints, verified resume,
  transient-I/O retry, and atomic publication are mandatory execution semantics.
- Saving is append-only and never renders frames, generates figures, or starts an
  encoder inside the Qt process. Approval requires an execution-ready task and an
  approved mass-analysis gate.

The built-in `post-analysis-v1` execution adapter materializes this plan without
recomputing chemistry, concentration, saturation, or mass. It reads standardized,
time-indexed field products, verifies every index and frame hash, integrates the
stored physical mass fields over the complete corrected grid and selected analysis
domains, and renders the exact selected figures, tables, movies, contact sheets,
cross-run grids, and provenance. Frame rendering is parallel and bounded; video
encoding and final publication happen from host-local staging. Expected and
observed artifact signatures must match exactly before the result is published.

Mass materializers connect to post-analysis through three small, generic index
contracts: `darsia-fluidflower-field-index-v1` for physical mass, concentration,
and gas-saturation fields; `darsia-fluidflower-field-frame-v1` for one independently
hash-verified time point; and `darsia-fluidflower-preview-index-v1` for approved
corrected display images. A worker can therefore stream, resume, or replace one
verified frame without loading an entire experiment into memory. Physical mass is
stored as g/pixel. A concentration frame may also carry a separate dimensionless
`c/csat` display field, but the display field never replaces the physical values.

Post-analysis is the final active workflow module. Its result manifest contains
the selected figures, tables, movies, file hashes, and numerical provenance
directly. There is no separate report gate, watermark, publication state, or
mandatory signature workflow. Expensive rendering and analysis remain separate
task-runner responsibilities.

## Project layout

```text
project/
  series.json
  source_manifest.json
  annotations/
    facies/
      examples/
        current.json
        history.jsonl
        revisions/<revision-id>.json
      index.jsonl
      bundles/
        <bundle-id>/
          manifest.json
          preview.png
          regions.csv
          targets.npz
          images/
            baseline.npz
            registered_review_frame.npz
  cache/
  history/audit.jsonl
  overrides/
    geometry_facies/
      current.json
      revisions.jsonl
      drafts/
        <draft-id>/
          baseline.npz
          depth.npz
          labels.npz
          facies.npz
      edit_sessions/
        active.json
        <session-id>/
          session.json
          events.jsonl
          materializations/
            <materialization-id>/
              labels.npz
              materialization.json
    image_correction/
      current.json
      revisions.jsonl
    phase_interpretation/
      current.json
      revisions.jsonl
    calibration/
      current.json
      revisions.jsonl
    mass_analysis/
      current.json
      revisions.jsonl
    analysis_domains/
      current.json
      revisions.jsonl
    wasserstein/
      current.json
      revisions.jsonl
    post_analysis/
      current.json
      revisions.jsonl
  results/
    facies_example_predictions/
      <prediction-id>/
        manifest.json
        facies.npz
        labels.npz
        uncertainty.npz
        preview.png
    facies_models/
      index.jsonl
      <model-id>/
        manifest.json
        model_card.md
        model.joblib
        metrics.json
        predictions/
          <bundle-id>/
            metrics.json
            prediction.npz
            preview.png
    geometry/
      <revision-id>/
        facies.npz
        facies_properties.csv
        revision.json
    integrity/
      current.json
      <scan-id>/
        report.json
        inventory.jsonl.gz
    phase_interpretation/
      <revision-id>/
        revision.json
    calibration/
      plans/
        <plan-id>/
          plan.json
    mass_analysis/
      plans/
        <plan-id>/
          plan.json
      products/
        <mass-task-id>/
          mass_index.json
          concentration_index.json
          gas_saturation_index.json
          preview_index.json
          frames/
            <run-id>/
              <frame-id>.npz
    analysis_domains/
      <revision-id>/
        revision.json
    wasserstein/
      plans/
        <plan-id>/
          plan.json
    post_analysis/
      plans/
        <plan-id>/
          plan.json
      <post-task-id>/
        post_analysis_result.json
        provenance.json
        tables/
        figures/
        movies/
        contact_sheets/
        cross_run/
  runs/
  timeline/
    raw_events.jsonl
    overrides.jsonl
```

`source_manifest.json` contains references and file fingerprints. DarSIA never
renames, copies, or edits the registered raw image and measurement sources.
Configuration changes are atomic, and audit records are append-only.

Absolute timestamps imported without an explicit experiment time are displayed
relative to the first timestamp in that source. This is intentionally marked in
the event provenance and must be aligned to the experiment anchor before the
temporal quality gate is approved.

## Module quality gates

The current dependency graph is:

```text
Series setup
  -> Data and integrity
     -> Overflow measurements
        -> Temporal synchronization
     -> Image correction
        -> Geometry and facies

Image correction + Geometry and facies + Temporal synchronization
  -> Optical and chemical calibration
     -> Phase interpretation
        -> Mass calibration
        -> Mass analysis
           -> Wasserstein analysis
           -> Post-analysis
```

Image correction establishes the registered and radiometrically corrected grid
before facies or optical response paths are built. Optical calibration binds the
active correction, geometry, and synchronized observations before phase
interpretation. Mass analysis also rechecks geometry and temporal synchronization.
Changing an approved upstream module marks completed dependent modules as
`Outdated`; it never deletes prior results.

## Design boundary

Qt is a presentation layer. Project storage and quality-gate logic live in
`darsia.fluidflower` and can be called from a CLI, workers, tests, or a different
GUI. Experiment-specific FF_AC and FF_UM tools will be connected through adapters
rather than embedded as hard-coded network paths.

## Execution layer

Compute is kept outside the Qt pages. Every expensive operation is described by
the same versioned task document whether it runs in a local process pool or
through a trusted shared network queue. `Execution and workers` is a
cross-cutting work surface rather than a scientific quality gate: it consumes
saved immutable plans without changing their approval state.

The hardware inventory will record CPU topology, available RAM, accelerator
models, accelerator memory, driver/runtime support, and measured backend
throughput. Recommendations will be module-specific rather than based only on
the presence of a GPU. The user can accept or override the recommendation.

The explicit numerical backends are:

- `cpu-float64`, the authoritative parity reference;
- `cpu-float32`, an optional lower-memory CPU backend that must pass sampled
  parity against `cpu-float64`;
- `cuda-float32` for supported NVIDIA devices;
- `cuda-float64` when an installed adapter and the selected NVIDIA device support it;
- `opencl-float32` for supported OpenCL devices;
- `opencl-float64` when the OpenCL device reports double-precision support;
- `metal-float32` for supported Apple devices.

Backend, device identity, precision, package revision, task configuration, and
input fingerprints belong to every result's provenance. New accelerated
backends must pass tolerance-based parity tests against `cpu-float64`; they must
never silently replace a production backend or change result semantics.

Quality is selected independently for each saved workflow plan. Geometry and
facies editing retain the native registered image grid and categorical labels;
calibration can therefore use native-grid float64 while a later mass-analysis
plan uses CPU/GPU float32 and a reduced scientific grid. Preview dimensions and
JPEG quality never alter the scientific grid. The current backend, compute
dtype, scientific grid, and preview-only quality are shown in the Qt work
surface, task monitor, and result provenance. Changing these controls affects
the next immutable plan and newly queued tasks only. Paused or running tasks
resume with their original quality contract, so one result can never mix dtype
or resolution silently.

The implemented runner provides:

- a hardware inventory for CPU topology, RAM, CUDA, OpenCL, and Metal devices;
- module-specific, editable worker recommendations with accelerator headroom;
- one immutable task and one lease per queue file;
- local and shared queues with the same directory and task contracts;
- exclusive atomic claims, heartbeats, bounded retries, stale-lease recovery,
  checkpoints, and work stealing;
- host-local staging/read-through cache and checksum-verified atomic publish;
- independent paths for the small task-control queue, each host's local cache,
  and the final result destination. A coordinator share can therefore contain
  only task documents while workers publish large products directly to a NAS;
- idempotent task submission and first-writer-wins result publication;
- cooperative pause after an atomic work chunk, hash-verified durable partial
  results, and resume without consuming an additional retry or trusting an
  incomplete chunk;
- hidden worker processes so Windows does not open terminal windows;
- an adapter registry loaded from trusted installed Python code.

Queue documents cannot contain executable shell commands. Scientific engines
are connected through the `darsia.fluidflower.execution_adapters` entry-point
group and must declare supported modules, backends, versions, and precision.
The built-in `plan-verification-v1` adapter exercises the complete plan,
fingerprint, lease, checkpoint, and publication path without claiming to create
scientific results. Four built-in adapters currently perform real materialization:

- `mass-analysis-cpu-float64-v1` verifies prepared optical, phase, geometry,
  chemistry, and protocol inputs and converts them to raw physical mass fields;
- `mass-analysis-cpu-float32-v1` runs the same contract with float32 CPU arrays
  and blocks publication if its deterministic CPU-float64 parity sample fails;
- `mass-analysis-accelerated-v1` performs the same phase-transfer, chemistry-LUT,
  saturation, phase-split, and mass algebra with an explicitly selected CUDA,
  OpenCL, or Metal backend;
- `wasserstein-cpu-float64-v1` verifies standardized mass and transport-weight
  indexes, performs physical mass-preserving reduction, computes weighted
  pairwise distances, and writes raw and normalized metrics with solver status;
- `post-analysis-v1` verifies the immutable post-analysis plan and all indexed
  mass/phase/image frames, then creates the planned diagnostics and media without
  modifying the scientific fields.
Publication remains atomic. Post-analysis records that no scientific result was
recalculated or mass-conserved during rendering.

The accelerated mass adapter has no implicit fallback. The backend saved in the
analysis plan, the queued task backend, and the numerical dtype must agree. CUDA
uses one fused CuPy elementwise kernel; OpenCL uses one fused PyOpenCL kernel with
a process-local compiled-program cache; Metal uses vectorized PyTorch MPS tensor
operations. Fixed geometry, activity masks, voxel volumes, and chemistry lookup
tables are retained on the device for the run. Image loading, input resampling,
pressure/temperature field preparation, artifact encoding, and storage remain CPU
or I/O work, so end-to-end speedup depends on the selected products and storage.

Every accelerated run recomputes a deterministic sample of complete frames with
the authoritative CPU-float64 DarSIA implementation. All six numerical fields
must satisfy the saved absolute and relative tolerances. A failed or missing
parity report prevents publication of the result. Provenance records the backend,
precision, physical device, driver/runtime, parity frames, tolerances, and
field-wise errors. Reduced image resolution is an independent quality-budget
choice and is never inferred from float32 or from the selected accelerator.

Optional runtime dependencies are installed per host capability:

- `darsia[fluidflower-cuda]` for CuPy CUDA;
- `darsia[fluidflower-opencl]` for PyOpenCL;
- `darsia[fluidflower-metal]` for PyTorch MPS on Apple Silicon.

An inventory may show a physical accelerator while marking its numerical backend
unavailable when the corresponding runtime dependency is missing. Calibration
materializers can be added independently without changing queue semantics.
FF_AC/FF_UM-specific launchers remain outside the generic engine until they emit
the same versioned numerical inputs and pass parity against the CPU-float64
reference.

The worker command is available as:

```text
darsia-fluidflower-worker inventory
darsia-fluidflower-worker init --queue <local-or-shared-folder>
darsia-fluidflower-worker worker --queue <folder> --cache <local-folder> \
  --worker-id <name> --backends cpu-float64
darsia-fluidflower-worker status --queue <folder>
darsia-fluidflower-worker recover --queue <folder> \
  --lease-timeout-seconds 900
```

Final artifacts are visible at their independently selected destination only
after every staged file has been copied and hash-verified. A worker that loses
the NAS can continue its local calculation; publication retries when storage
returns. If its lease expires, another compatible worker can take over from the
shared checkpoint.
