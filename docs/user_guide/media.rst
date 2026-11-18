Photo and mass films
===================

The ``film-tiles`` and ``films`` steps share their panel functions with other
callers. They read stored corrected images, phase maps and mass maps. Display
conversion, averaging and overlays never alter those scientific arrays.

To place two panels horizontally, set a film's ``layout = "side-by-side"`` and
provide exactly two panels, in left-to-right order. Existing ``pyramid`` and
``grid`` layouts keep their three-panel and one-panel meanings.

The following are additional settings, merged with the existing required
``[tiles]``, ``[photograph]`` and ``[style]`` fields::

    [tiles]
    mass_quantity = "area"
    normalization = "sqrt"
    # colour_scale is REQUIRED: one finite positive maximum in g/m².
    # Select it from the complete comparison population, including early gas,
    # and pin the same value for every run. There is no automatic range.

    [photograph]
    co2_outline = true
    gas_contour = true
    gas_contour_fraction = 0.9
    co2_colour = "white"
    gas_colour = "red"
    outline_width_pt = 1.0

    [style]
    over_range = "#ff00ff"

The default mass quantity is ``"pore-volume"`` (g/L), and the default
normalization is ``"linear"``. Omitting the new fields retains the original
display math and tile metadata. ``"area"`` multiplies the stored density by
1000 litres per cubic metre, the measured pore volume, and the reciprocal of the
measured pixel area. Display blocks are area-weighted, conserving their known
mass. Unknown density in positive pore volume stays unknown; zero pore volume
has zero mass; invalid area or unknown phase remains unknown. A depth display
mean still operates on density before area conversion.

Square-root normalization changes colours only. The fixed maximum, colour-bar
ticks and captions stay in physical units; there is no square-root unit. Area
and nonlinear views colour values above the maximum with ``over_range``. Their
tile records store the maximum, quantity, normalization, colour, known and
clipped display-cell counts, and ``clipped_share``. The percentage in the film
is the count of displayed finite cells above the maximum divided by all
displayed finite cells after display averaging. It is not a mass fraction or
an original-grid pixel fraction. With no known cells it is unavailable, not
zero. Films reject new tiles made with a different configured scale.

By default, the CO2 outline encloses known dissolved CO2 and gas. The gas contour uses
``saturation >= gas_contour_fraction * (1 - Swi)`` within known CO2 and explicitly
marked ``GAS_BRIDGE`` pixels on known water; allocated gas can retain a dissolved
or marked-water categorical label. A bridge does not enlarge the white optical
CO2 outline. Unknown and unmarked water pixels are not added to these default contours.
Invalid Swi does not yield a gas contour. Thin gas below the threshold remains
present in the mass map. Missing
phase labels, saturation or required geometry raise an explicit error when
their display option is enabled. Legacy contours are extracted on the full
grid before raster reduction, retaining their original behavior.

New views require the new film tiles; legacy tiles cannot be silently
relabelled as mass per area. Their per-frame contour and clipping descriptions
are included in both the rendered captions and the existing film record.

Fixed first-frame photographs and grid-sized pairs
------------------------------------------------

``photograph.transfer = "first-frame"`` measures the configured ``percentiles``
(by default 0.5 and 99.5) on the earliest selected photograph after INTER_AREA
reduction. Those two limits remain fixed throughout that run's film. Display
values are clipped between the limits and raised to ``1/gamma`` (default 2.2).
The camera-to-sRGB matrix is unused in this explicit transfer mode. Nonfinite
RGB remains unknown and uses ``invalid_grey``; it does not enter the percentiles.
The limits, percentile rule and gamma are stored with each photograph tile.

``photograph.co2_source = "mass"`` changes the white contour to positive finite
total CO2 mass per measured area. Its caption says CO2 mass, not optical
detection. Positive hidden bridge mass therefore participates. Unknown phase or
nonfinite mass stays excluded; this display option never changes scientific
labels, masses or saturation. The independently configured red saturation
contour still admits only known CO2 or marked known-water bridges with valid Swi.
In the first-frame producer mode, each occupancy mask is reduced with
INTER_AREA and thresholded at greater than zero before its display-grid contour
is traced. A finely fragmented occupied region therefore retains an outline
rather than having its interior painted by reduced full-grid boundaries.

``tiles.grid_divisor = 4`` derives each panel's width and height by integer
division of its measured grid dimensions. Area maps then use INTER_AREA
reduction on their physical g/m² values with a fixed square-root scale; any
unknown source in a display footprint remains unknown. Cells with known zero
pore volume are dark grey. ``layout = "paired-grid"`` places photo left and map
right without resizing either, followed by ``key_width_px`` (default 120).
Set the encoded ``size_px`` to the resulting logical size rounded UP to even
dimensions. At most one dark row/column is added; no logical pixel is cropped.
Both logical panel/frame sizes and encoded size are recorded. A separate graph
film uses ``layout = "single"`` with the graph panel: the existing grid renderer
runs independently for each run, with common whole-domain axes derived from
all selected domain-mass series. No cross-run time alignment is introduced.

The generic ``transfer.probe_indices`` selects distinct ordered indices n/8,
n/4, n/2 and last (integer division). One ``population_scale`` call receives
all selected runs' finite positive g/m² probe values. It computes the exact
pooled NumPy-linear percentile, retaining a bounded upper tail rather than
averaging per-run percentiles. Its maximum sample bound is the sum of the
probe grid sizes. All positive values are counted; values no larger than the
retained tail's monotone cutoff can then be discarded. Survivors accumulate in
one additional tail-sized buffer before partition, with a final exact flush.
The tail is not repartitioned for every source chunk.
The resulting value is the ONE ``tiles.colour_scale`` used
by every run; no last-frame or historical scale is substituted. Empty positive
populations fail explicitly. Authoring must retain the accepted producer/frame
identities with this display-only scale receipt before creating film jobs.
