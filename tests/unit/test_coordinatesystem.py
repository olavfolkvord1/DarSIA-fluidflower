"""Module testing the coordinate system of Image.

"""

import numpy as np
import pytest

import darsia


def test_init_unit_coordinatesystem_2d():
    """Test the definition of all attributes."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Check all attributes defined in CoordinateSystem.__init__()
    assert cs.dim == 2, "space dimension not correct"
    assert cs.shape == (3, 4), "shape not correct"
    assert np.allclose(cs.dimensions, [1, 1]), "dimensions not correct"
    assert cs.axes == "xy", "axes not correct"
    assert np.allclose(
        [cs.voxel_size["x"], cs.voxel_size["y"]], [1 / 4, 1 / 3]
    ), "voxel_size not correct"
    assert np.allclose(cs._coordinate_of_origin_voxel, [0, 1]), "origin not correct"
    assert np.allclose(cs._coordinate_of_opposite_voxel, [1, 0]), "opposite not correct"
    assert np.isclose(cs.domain["xmin"], 0), "xmin not correct"
    assert np.isclose(cs.domain["xmax"], 1), "xmax not correct"
    assert np.isclose(cs.domain["ymin"], 0), "ymin not correct"
    assert np.isclose(cs.domain["ymax"], 1), "ymax not correct"


def test_voxels_unit_coordinatesystem_2d():
    """Test CoordinateSystem.voxels()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Check all attributes defined in CoordinateSystem.__init__()
    assert np.allclose(
        cs.voxels,
        np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
                [0, 3],
                [1, 3],
                [2, 3],
            ]
        ),
    ), "voxels not correct"


def test_coordinates_unit_coordinatesystem_2d():
    """Test CoordinateSystem.coordinates()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Check all attributes defined in CoordinateSystem.__init__()
    assert np.allclose(
        cs.coordinates,
        np.array(
            [
                [0, 1],
                [0, 2 / 3],
                [0, 1 / 3],
                [1 / 4, 1],
                [1 / 4, 2 / 3],
                [1 / 4, 1 / 3],
                [2 / 4, 1],
                [2 / 4, 2 / 3],
                [2 / 4, 1 / 3],
                [3 / 4, 1],
                [3 / 4, 2 / 3],
                [3 / 4, 1 / 3],
            ]
        ),
    ), "coordinates not correct"


def test_length_unit_coordinatesystem_2d():
    """Test CoordinateSystem.length()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    for i in range(4):
        assert np.isclose(cs.length(i, "x"), i / 4), "length not correct in x"
    for i in range(3):
        assert np.isclose(cs.length(i, "y"), i / 3), "length not correct"


def test_num_voxels_unit_coordinatesystem_2d():
    """Test CoordinateSystem.num_voxels()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    assert np.allclose(cs.num_voxels(1, "x"), 4), "num_voxels not correct in x"
    assert np.allclose(cs.num_voxels(1, "y"), 3), "num_voxels not correct in y"
    assert np.allclose(cs.num_voxels(0.7, "x"), 3), "num_voxels not correct in x"
    assert np.allclose(cs.num_voxels(0.6, "y"), 2), "num_voxels not correct in y"
    assert np.allclose(cs.num_voxels(0.8, "x"), 4), "num_voxels not correct in x"
    assert np.allclose(cs.num_voxels(0.7, "y"), 3), "num_voxels not correct in y"


def test_coordinate_unit_coordinatesystem_2d():
    """Test CoordinateSystem.coordinate()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Define test for single voxel
    voxel_list = [1, 2]

    def check_voxel(voxel):
        coordinate = cs.coordinate(voxel)
        assert isinstance(coordinate, darsia.Coordinate), "coordinate type not correct"
        assert np.allclose(coordinate, [0.5, 2 / 3]), "coordinate values not correct"

    # Input: np.ndarray
    voxel = np.array(voxel_list)
    check_voxel(voxel)

    # Input: list
    voxel = voxel_list
    check_voxel(voxel)

    # Input: tuple
    voxel = tuple(voxel_list)

    # Input: Voxel
    voxel = darsia.Voxel(voxel_list)
    check_voxel(voxel)

    # Define test for multiple voxels
    voxels_list = [[1, 2], [2, 3]]

    def check_voxels(voxels):
        coordinates = cs.coordinate(voxels)
        assert isinstance(
            coordinates, darsia.CoordinateArray
        ), "coordinate type not correct"
        assert np.allclose(
            coordinates, [[0.5, 2 / 3], [0.75, 1 / 3]]
        ), "coordinate values not correct"

    # Input: np.ndarray
    voxels = np.array(voxels_list)
    check_voxels(voxels)

    # Input: list
    voxels = voxels_list
    check_voxels(voxels)

    # Input: VoxelArray
    voxels = darsia.VoxelArray(voxels_list)
    check_voxels(voxels)


def test_voxel_unit_coordinatesystem_2d():
    """Test CoordinateSystem.voxel()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Define test for single coordinate
    coordinate_list = [0.5, 2 / 3]

    def check_coordinate(coordinate):
        voxel = cs.voxel(coordinate)
        assert isinstance(voxel, darsia.Voxel), "voxel type not correct"
        assert np.allclose(voxel, [1, 2]), "voxel values not correct"

    # Input: np.ndarray
    coordinate = np.array(coordinate_list)
    check_coordinate(coordinate)

    # Input: list
    coordinate = coordinate_list
    check_coordinate(coordinate)

    # Input: tuple
    coordinate = tuple(coordinate_list)

    # Input: Coordinate
    coordinate = darsia.Coordinate(coordinate_list)
    check_coordinate(coordinate)

    # Define test for multiple coordinates
    coordinates_list = [[0.5, 2 / 3], [0.75, 1 / 3], [0.6, 0.9]]

    def check_coordinates(coordinates):
        voxels = cs.voxel(coordinates)
        assert isinstance(voxels, darsia.VoxelArray), "voxel type not correct"
        assert np.allclose(voxels, [[1, 2], [2, 3], [0, 2]]), "voxel values not correct"

    # Input: np.ndarray
    coordinates = np.array(coordinates_list)
    check_coordinates(coordinates)

    # Input: list
    coordinates = coordinates_list
    check_coordinates(coordinates)

    # Input: CoordinateArray
    coordinates = darsia.CoordinateArray(coordinates_list)
    check_coordinates(coordinates)


def test_coordinate_vector_unit_coordinatesystem_2d():
    """Test CoordinateSystem.coordinate_vector()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Input: np.ndarray for single vector
    voxel_vector = np.array([1, 2])
    coordinate_vector = cs.coordinate_vector(voxel_vector)
    assert np.allclose(
        coordinate_vector, [0.5, -1 / 3]
    ), "coordinate_vector values not correct"

    # Input: np.ndarray for multiple vectors
    voxel_vectors = np.array([[1, 2], [2, 2]])
    coordinate_vectors = cs.coordinate_vector(voxel_vectors)
    assert np.allclose(
        coordinate_vectors, [[0.5, -1 / 3], [0.5, -2 / 3]]
    ), "coordinate_vector values not correct"


def test_pixel_centres_lie_half_a_pixel_inside_the_top_left_corner():
    """x runs right from the origin, y down from it, half a voxel inside."""
    image = darsia.Image(np.zeros((2, 4)), dimensions=[0.2, 0.8], origin=[1.0, 3.0])

    x, y = darsia.pixel_centres(image)

    assert np.allclose(x, [1.1, 1.3, 1.5, 1.7])
    assert np.allclose(y, [2.95, 2.85])


def test_grid_positions_are_the_measured_model_at_the_pixels_else_their_centres():
    """With a measured model, each pixel's position is the model at its column and
    row in the given precision, and with a stride the pitch the median spacing
    along every stride-th row and down every stride-th column (none without a
    stride); without one, the pixel centres and the voxel size."""
    image = darsia.Image(np.zeros((130, 200)), dimensions=[0.13, 0.2])
    plain = darsia.grid_positions(image)
    x, y = darsia.pixel_centres(image)
    np.testing.assert_array_equal(plain.x, np.broadcast_to(x[None, :], (130, 200)))
    np.testing.assert_array_equal(plain.y, np.broadcast_to(y[:, None], (130, 200)))
    assert plain.pitch_mm == (image.voxel_size[0] * 1000, image.voxel_size[1] * 1000)

    angle = 0.004
    model = np.array(
        [
            [1.003e-3 * np.cos(angle), -1e-3 * np.sin(angle), 0.0002],
            [-1e-3 * np.sin(angle), -0.998e-3 * np.cos(angle), 0.1299],
            [1e-6, -2e-6, 1.0],
        ]
    )
    for dtype in (np.float32, np.float64):
        assert darsia.grid_positions(image, model.tolist(), dtype).pitch_mm is None
        found = darsia.grid_positions(image, model.tolist(), dtype, pitch_stride=64)
        m = model.astype(dtype)
        c = np.arange(200, dtype=dtype)[None, :]
        r = np.arange(130, dtype=dtype)[:, None]
        den = m[2, 0] * c + m[2, 1] * r + m[2, 2]
        np.testing.assert_array_equal(
            found.x, (m[0, 0] * c + m[0, 1] * r + m[0, 2]) / den
        )
        np.testing.assert_array_equal(
            found.y, (m[1, 0] * c + m[1, 1] * r + m[1, 2]) / den
        )
        assert found.x.dtype == dtype and found.y.dtype == dtype
        dx = float(np.median(np.abs(np.diff(found.x[::64], axis=1)))) * 1000
        dy = float(np.median(np.abs(np.diff(found.y[:, ::64], axis=0)))) * 1000
        assert found.pitch_mm == (dy, dx) and found.pitch_mm != plain.pitch_mm


def test_the_pitch_is_taken_over_the_canvas_the_pixels_were_measured_on():
    """On a measurement canvas around the grid (its first pixel at the grid's
    column and row ``origin``), the pitch is the median spacing of the model at
    the canvas' whole pixels along every stride-th canvas row and down every
    stride-th canvas column, in the positions' precision; the positions stay
    the grid's. A canvas that does not hold the grid is refused."""
    image = darsia.Image(np.zeros((130, 200)), dimensions=[0.13, 0.2])
    model = [
        [0.000166, 4.3e-08, -6.0e-05],
        [1.6e-09, -0.000166, 0.0467],
        [2e-5, 1e-5, 1.0],
    ]
    canvas = darsia.MeasurementCanvas((-41, -23), (176, 282))
    f32 = np.float32
    found = darsia.grid_positions(image, model, f32, canvas, 64)
    plain = darsia.grid_positions(image, model, f32, None, 64)
    np.testing.assert_array_equal(found.x, plain.x)
    np.testing.assert_array_equal(found.y, plain.y)
    m = np.asarray(model, f32)
    c = (np.arange(282, dtype=f32) - 41)[None, :]
    r = (np.arange(176, dtype=f32) - 23)[:, None]
    x, y = darsia.project_xy(m, c, r)  # the whole canvas, as measured
    dx = float(np.median(np.abs(np.diff(x[::64], axis=1)))) * 1000
    dy = float(np.median(np.abs(np.diff(y[:, ::64], axis=0)))) * 1000
    assert found.pitch_mm == (dy, dx)
    assert found.pitch_mm[0] != plain.pitch_mm[0]
    assert found.pitch_mm[1] != plain.pitch_mm[1]
    with pytest.raises(ValueError, match="does not hold the grid"):
        darsia.grid_positions(
            image, model, f32, darsia.MeasurementCanvas((1, 0), (130, 200)), 64
        )
    assert darsia.MeasurementCanvas((-41, -23), (176, 282)).holds(130, 200)
    assert not darsia.MeasurementCanvas((0, 0), (129, 200)).holds(130, 200)


def test_grid_pixel_areas_are_the_measured_models_jacobian_in_its_precision():
    """A pixel's area is the model's Jacobian at its column and row: the model in
    millimetres, cast to the precision and inverted twice there, the area in
    double precision stored in single precision as square millimetres, the area
    factor applied in single precision, and square metres in the precision; a
    model crossing its horizon on the grid is refused."""
    image = darsia.Image(np.zeros((300, 400)), dimensions=[0.05, 0.066])
    model = [
        [0.00016634789062663913, 4.3004735772456115e-08, -6.023491368978284e-05],
        [1.642236013843501e-09, -0.00016619959205854684, 0.046765316277742386],
        [6.102835570231946e-09, 2.6038710743137017e-08, 0.9966205954551697],
    ]
    f32 = np.float32
    x = np.arange(400, dtype=np.float64)[None, :]
    y = np.arange(300, dtype=np.float64)[:, None]

    def rule(dtype, inversions=2, factor=None, factor_dtype=f32):
        matrix = np.asarray(np.diag([1000.0, 1000.0, 1.0]) @ np.array(model), dtype)
        for _ in range(inversions):
            matrix = np.linalg.inv(np.asarray(matrix, dtype))
        w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        area = (abs(np.linalg.det(matrix)) / np.abs(w) ** 3).astype(f32)
        if factor is not None:
            area = (area.astype(factor_dtype) * factor).astype(f32)
        return np.asarray(area, dtype) / 1e6

    for dtype in (f32, np.float64):
        for factor in (None, 1.0344):
            areas = darsia.grid_pixel_areas(image, model, dtype, factor)
            assert areas.dtype == dtype and areas.shape == (300, 400)
            np.testing.assert_array_equal(areas, rule(dtype, factor=factor))
    areas = darsia.grid_pixel_areas(image, model, f32, 1.0344)
    assert np.unique(areas).size > 100  # each pixel its own
    # Each choice of the rule tells: the precision of the matrices, the two
    # inversions and the factor in single precision.
    assert not np.array_equal(areas, rule(np.float64, factor=1.0344).astype(f32))
    assert not np.array_equal(areas, rule(f32, inversions=0, factor=1.0344))
    assert not np.array_equal(areas, rule(f32, factor=1.0344, factor_dtype=np.float64))
    horizon = [[1e-4, 0, 0], [0, -1e-4, 0.05], [0, -1e-2, 1.0]]  # w = 0 at row 100
    with pytest.raises(ValueError, match="horizon"):
        darsia.grid_pixel_areas(image, horizon)


def test_grid_record_tells_images_on_one_grid():
    """Images on one grid have one record, whatever their channels; another
    origin is another grid."""
    grid = {"dimensions": [0.2, 0.8], "origin": [1.0, 3.0]}
    labels = darsia.Image(np.zeros((2, 4), np.uint8), **grid)
    colours = darsia.Image(np.zeros((2, 4, 3)), **grid)
    moved = darsia.Image(np.zeros((2, 4)), dimensions=[0.2, 0.8], origin=[1.0, 3.1])

    assert darsia.grid_record(labels) == darsia.grid_record(colours)
    assert darsia.grid_record(labels) == {"shape": [2, 4], **grid}
    assert darsia.grid_record(moved) != darsia.grid_record(labels)


def test_optical_pitch_takes_the_area_factor_of_the_pixel_areas():
    """The coordinate pitch times the root of the median area over the cell."""
    area = np.full((2, 4), 0.1 * 0.2 * 1.21)
    area[0, 0] = 1.0  # one outlier leaves the median
    image = darsia.Image(area, dimensions=[0.2, 0.8])

    pitch = darsia.optical_pitch_m(image, np.ones((2, 4), bool))

    assert np.allclose(pitch, (0.11, 0.22))


def test_optical_pitch_takes_the_measured_coordinate_pitch_with_positions():
    """With measured positions the coordinate pitch is the mean distance, in the
    positions' precision and averaged in double, between neighbouring pixels both
    inside the domain."""
    image = darsia.Image(np.zeros((30, 40)), dimensions=[0.005, 0.0066])
    model = [
        [0.00016634789062663913, 4.3004735772456115e-08, -6.023491368978284e-05],
        [1.642236013843501e-09, -0.00016619959205854684, 0.046765316277742386],
        [6.102835570231946e-09, 2.6038710743137017e-08, 0.9966205954551697],
    ]
    positions = darsia.grid_positions(image, model, np.float32)
    area = darsia.Image(
        darsia.grid_pixel_areas(image, model, np.float32, 1.0344),
        dimensions=[0.005, 0.0066],
    )
    domain = np.ones((30, 40), bool)
    domain[:5, :7] = False
    x, y = positions.x, positions.y
    down = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))[domain[1:] & domain[:-1]]
    along = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))[
        domain[:, 1:] & domain[:, :-1]
    ]
    assert down.dtype == np.float32
    coordinate = [
        float(np.mean(down, dtype=np.float64)),
        float(np.mean(along, dtype=np.float64)),
    ]
    factor = float(np.median(area.img.astype(np.float64)[domain])) / np.prod(coordinate)
    expected = tuple(p * float(np.sqrt(factor)) for p in coordinate)
    assert darsia.optical_pitch_m(area, domain, positions) == expected
    assert darsia.optical_pitch_m(area, domain) != expected
    whole = darsia.optical_pitch_m(area, np.ones((30, 40), bool), positions)
    assert whole != expected


def test_the_optical_grid_is_sized_on_the_canvas_of_its_pitch():
    """A pitch with the counts of its canvas sizes the optical grid on the canvas'
    pixels: the canvas' dimensions at the pitch in the dtype, over its pixels; the
    pitch stays the pair it is."""
    image = darsia.Image(np.zeros((100, 150)), dimensions=[0.02, 0.03])
    pitch = (1.6714851236783506e-4, 1.6712130886473872e-4)
    sized = darsia.OpticalPitch(pitch, (101, 202))
    assert sized == pitch and sized.counts == (101, 202)
    assert darsia.OpticalPitch(pitch).counts is None
    found = darsia.optical_grid(image, sized, np.float32).voxel_size
    expected = [np.float32(n * p) / n for n, p in zip((101, 202), pitch)]
    assert found == expected and all(v.dtype == np.float32 for v in found)
    own = darsia.optical_grid(image, pitch, np.float32).voxel_size
    assert own == [np.float32(n * p) / n for n, p in zip((100, 150), pitch)]
    assert own[0] != found[0] and own[1] != found[1]


def test_optical_grid_sizes_the_pixels_with_the_pitch_in_the_given_dtype():
    """The image's values on the dimensions of its pixels at the pitch."""
    sand_map = darsia.Image(np.arange(8).reshape(2, 4), dimensions=[0.2, 0.8])

    grid = darsia.optical_grid(sand_map, (0.11, 0.22), np.float32)

    assert grid.dimensions == [np.float32(2 * 0.11), np.float32(4 * 0.22)]
    assert all(type(length) is np.float32 for length in grid.dimensions)
    np.testing.assert_array_equal(grid.img, sand_map.img)
