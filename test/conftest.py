"""Small, analytical meshes shared by the tests."""

import numpy as np
import pyvista as pv
import pytest

from tps_interface import TwoDInterface


@pytest.fixture
def truncated_triangle() -> pv.UnstructuredGrid:
    """Two triangles covering R(z) = 1 - z for 0.2 <= z <= 0.8.

    returns:
        An unstructured grid with two triangular cells and scalar point fields
        'constant' and 'linear'
    """

    points = np.array([[0.0, 0.2, 0.0],
                       [0.8, 0.2, 0.0],
                       [0.2, 0.8, 0.0],
                       [0.0, 0.8, 0.0]])

    cells = np.array([3, 0, 1, 2, 3, 0, 2, 3])

    cell_types = np.array([pv.CellType.TRIANGLE, pv.CellType.TRIANGLE])

    mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # f(r, z) = 1
    mesh.point_data['constant'] = np.ones(len(points))
    # f(r, z) = 1 + r + 2z
    mesh.point_data['linear'] = 1 + points[:, 0] + 2*points[:, 1]

    return mesh


@pytest.fixture
def interface(truncated_triangle: pv.UnstructuredGrid) -> TwoDInterface:
    """Creates an interface to the truncated triangle.

    args:
        truncated_triangle: Mesh supplied by the truncated_triangle fixture.

    returns:
        A `TwoDInterface` with a sufficient number of radial sampling points,
        for the test suite tolerance
    """
    return TwoDInterface(truncated_triangle, n_points=21)
