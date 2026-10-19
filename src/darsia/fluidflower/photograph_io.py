"""Photograph decoding for retained annotation and display surfaces."""
from pathlib import Path
import numpy as np


def is_camera_raw(path: str | Path) -> bool:
    """Whether ``path`` is a camera RAW file (ARW, NEF, ...), by its suffix."""

    from darsia.fluidflower.image_formats import image_family

    return image_family(path) == "raw"


def _rawpy():
    try:
        import rawpy
    except ImportError as error:
        raise ValueError(
            "Reading camera RAW files needs the rawpy package (pip install rawpy)."
        ) from error
    return rawpy


def _raw_size(sizes) -> tuple[int, int]:
    """Width and height of LibRaw's output: the visible area, turned upright."""

    width, height = int(sizes.width), int(sizes.height)
    return (height, width) if int(sizes.flip) in (5, 6) else (width, height)


def raw_photo_size(path: str | Path) -> tuple[int, int]:
    """Width and height of a camera RAW file: LibRaw's visible area, upright.

    This is the grid the RAW decoders and the RAW display rendering use. It is
    not the camera's JPEG grid, which is cropped and, for some cameras,
    lens-corrected. ``series_geometry.photo_size`` uses it for RAW files.
    """

    rawpy = _rawpy()
    try:
        with rawpy.imread(str(path)) as raw:
            return _raw_size(raw.sizes)
    except rawpy.LibRawError as error:
        raise ValueError(f"Could not read the RAW file {Path(path).name}: {error}") from error


def raw_capture_time(path: str | Path):
    """Capture time of a camera RAW file from its EXIF, or None.

    ARW, NEF and DNG files are TIFF-based. DateTimeOriginal of the EXIF block
    is used, else DateTime of the first directory, to whole seconds as for
    JPEG files. Returns a naive ``datetime`` in the camera's clock.
    """

    from datetime import datetime

    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tiff:
            page = tiff.pages[0]
            exif = page.tags.get("ExifTag")
            value = exif.value.get("DateTimeOriginal") if exif and isinstance(exif.value, dict) else None
            if not value:
                tag = page.tags.get("DateTime")
                value = tag.value if tag is not None else None
    except (OSError, ValueError, tifffile.TiffFileError):
        return None
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _render_raw_rgb8(path: Path, factor: int) -> np.ndarray:
    """A display rendering of a camera RAW file at 1/factor, as uint8 RGB.

    LibRaw demosaics with the camera's white balance and the standard output
    gamma, so the photo looks like a camera JPEG; this is for display and
    painting, never for measurements. From factor 2 it decodes at half size
    first, and the rest of the reduction is a box filter.
    """

    from PIL import Image

    rawpy = _rawpy()
    try:
        with rawpy.imread(str(path)) as raw:
            width, height = _raw_size(raw.sizes)
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=factor >= 2)
    except rawpy.LibRawError as error:
        raise ValueError(f"Could not read the RAW file {path.name}: {error}") from error
    target = (max(1, width // factor), max(1, height // factor))
    if (rgb.shape[1], rgb.shape[0]) != target:
        rgb = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).resize(target, Image.BOX)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def load_reduced_rgb8(path: Path, factor: int) -> np.ndarray:
    """Read a photograph at 1/factor resolution as uint8 RGB.

    JPEG files are decoded at reduced size directly (DCT scaling). Camera RAW
    files get LibRaw's display rendering on their own grid (see
    ``raw_photo_size``). A full resolution photograph (factor 1) costs one
    byte per value, a quarter of the float copy that ``load_reduced`` returns.
    """

    if is_camera_raw(path):
        return _render_raw_rgb8(Path(path), factor)

    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        if factor > 1:
            image.draft("RGB", (width // factor, height // factor))
        image = image.convert("RGB")
        target = (max(1, width // factor), max(1, height // factor))
        if image.size != target:
            image = image.resize(target, Image.BOX)
        return np.asarray(image, dtype=np.uint8)


def load_reduced(path: Path, factor: int) -> np.ndarray:
    """Read a photograph at 1/factor resolution as float32 RGB in [0, 1].

    JPEG files are decoded at reduced size directly (DCT scaling); camera RAW
    files as in ``load_reduced_rgb8``.
    """

    return np.asarray(load_reduced_rgb8(path, factor), dtype=np.float32) / 255.0
