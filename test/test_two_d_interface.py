"""Check the interface against analytical geometry and scalar fields."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pyvista as pv
import pytest
from numpy.testing import assert_allclose, assert_equal

from tps_interface import TwoDInterface, save_torch_radius, load_torch_radius

# Include the inlet and outlet; both have strictly positive radii.
Z = np.array([0.2, 0.35, 0.5, 0.65, 0.8])

# Radius finding uses rtol=1e-7. Integrals also have O(h**2) trapezoidal
# error for the linear field, with h=1/20 in normalized radius.
RADIUS_RTOL = 2e-7
INTEGRAL_RTOL = 1e-3


def exact_integral(field: str, z: float | np.ndarray) -> float | np.ndarray:
    """Evaluates the analytical cross-sectional integral of a test field.

    args:
        field: Test field name, either 'constant' or 'linear'.
        z: Axial position or array of axial positions.

    returns:
        The axisymmetric integral at each supplied axial position.
    """

    radius = 1 - z

    if field == 'constant':
        return np.pi*radius**2

    return np.pi*radius**2*(1 + 2*z) + (2*np.pi/3)*radius**3


@pytest.mark.parametrize('z', Z)
def test_radius(interface: TwoDInterface, z: float):
    """Checks the recovered radius against the linear wall geometry.

    args:
        interface: Interface fixture for the truncated triangular domain.
        z: Axial position within the domain.
    """
    assert_allclose(interface.torch_radius(z), 1 - z, rtol=RADIUS_RTOL)


@pytest.mark.parametrize('r,z', [(0.0, 0.4), (0.1, 0.5), (0.6, 0.3)])
def test_point_evaluation(interface: TwoDInterface, r: float, z: float):
    """Checks interpolation of constant and linear fields at an interior point.

    args:
        interface: Interface fixture for the truncated triangular domain.
        r: Radial coordinate of the evaluation point.
        z: Axial coordinate of the evaluation point.
    """
    assert_allclose(interface('constant', r, z), 1.0, atol=1e-12)
    assert_allclose(interface('linear', r, z), 1 + r + 2*z, atol=1e-12)


@pytest.mark.parametrize('z', Z)
def test_wall_value(interface: TwoDInterface, z: float):
    """Checks field values sampled at the recovered wall radius.

    args:
        interface: Interface fixture for the truncated triangular domain.
        z: Axial position within the domain.
    """
    assert_allclose(interface.wall_value('constant', z), 1.0, atol=1e-12)
    assert_allclose(interface.wall_value('linear', z), 2 + z, atol=2e-7)


@pytest.mark.parametrize('z', Z)
@pytest.mark.parametrize('field', ['constant', 'linear'])
def test_radial_sample(interface: TwoDInterface, field: str, z: float):
    """Checks radial sample values and the number of returned points.

    args:
        interface: Interface fixture for the truncated triangular domain.
        field: Test field name, either 'constant' or 'linear'.
        z: Axial position within the domain.
    """

    samples = interface.radial_sample(field, z)

    assert_equal(samples.shape, (interface.n_points,))

    expected = np.ones(interface.n_points)

    if field == 'linear':
        expected = 1 + 2*z + (1 - z)*interface.r_norm

    assert_allclose(samples, expected, atol=2e-7, rtol=0)


@pytest.mark.parametrize('z', Z)
@pytest.mark.parametrize('field', ['constant', 'linear'])
def test_cross_sectional_integral(interface: TwoDInterface, field: str,
                                  z: float):
    """Checks numerical integration against the analytical field integral.

    args:
        interface: Interface fixture for the truncated triangular domain.
        field: Test field name, either 'constant' or 'linear'.
        z: Axial position within the domain.
    """
    assert_allclose(interface.cs_integral(field, z), exact_integral(field, z),
                    rtol=INTEGRAL_RTOL)


@pytest.mark.parametrize('z', Z)
@pytest.mark.parametrize('field', ['constant', 'linear'])
def test_radial_profile(interface: TwoDInterface, field: str, z: float):
    """Checks the normalized profile and its cross-sectional integral.

    args:
        interface: Interface fixture for the truncated triangular domain.
        field: Test field name, either 'constant' or 'linear'.
        z: Axial position within the domain.
    """

    profile, integral = interface.radial_profile(field, z)
    expected = np.ones(interface.n_points)

    if field == 'linear':
        radius = 1 - z
        expected = ((1 + 2*z + radius*interface.r_norm)
                    / (1 + 2*z + 2*radius/3))

    assert_allclose(profile, expected, rtol=INTEGRAL_RTOL)
    assert_allclose(integral, exact_integral(field, z), rtol=INTEGRAL_RTOL)


def test_array_inputs(interface: TwoDInterface):
    """Checks array output shapes, analytical values, and scalar consistency.

    args:
        interface: Interface fixture for the truncated triangular domain.
    """

    assert_allclose(interface.torch_radius(Z), 1 - Z, rtol=RADIUS_RTOL)
    assert_allclose(interface.wall_value('linear', Z), 2 + Z, atol=2e-7)

    samples = interface.radial_sample('linear', Z)
    profiles, integrals = interface.radial_profile('linear', Z)

    assert_equal(samples.shape, (len(Z), interface.n_points))
    assert_equal(profiles.shape, samples.shape)
    assert_equal(integrals.shape, (len(Z),))

    assert_allclose(integrals, exact_integral('linear', Z),
                    rtol=INTEGRAL_RTOL)
    assert_allclose(interface.cs_integral('linear', Z), integrals)

    for i, z in enumerate(Z):
        assert_allclose(samples[i], interface.radial_sample('linear', z))
        assert_allclose(profiles[i], interface.radial_profile('linear', z)[0])


def test_set_mesh_invalidates_radius_cache(
        interface: TwoDInterface, truncated_triangle: pv.UnstructuredGrid):
    """Checks that replacing the mesh discards previously cached radii.

    args:
        interface: Interface fixture whose mesh will be replaced.
        truncated_triangle: Original mesh fixture used to create a wider mesh.
    """

    assert_allclose(interface.torch_radius(0.5), 0.5, rtol=RADIUS_RTOL)

    wider_mesh = truncated_triangle.copy(deep=True)
    wider_mesh.points[:, 0] *= 2
    interface.set_mesh(wider_mesh)

    assert_allclose(interface.torch_radius(0.5), 1.0, rtol=RADIUS_RTOL)


def test_missing_field(interface: TwoDInterface):
    """Checks that sampling an unknown field raises a RuntimeError.

    args:
        interface: Interface fixture for the truncated triangular domain.
    """

    with pytest.raises(RuntimeError, match='not in the dataset'):
        interface.radial_sample('missing', 0.5)


@pytest.mark.parametrize('z', [0.1, 0.9])
def test_axial_position_outside_domain(interface: TwoDInterface, z: float):
    """Checks that radius queries outside the axial bounds raise an error.

    args:
        interface: Interface fixture for the truncated triangular domain.
        z: Axial position outside the domain.
    """

    with pytest.raises(RuntimeError, match='outside domain axial bounds'):
        interface.torch_radius(z)


def test_point_outside_domain(interface: TwoDInterface):
    """Checks that field evaluation outside the mesh raises a RuntimeError.

    args:
        interface: Interface fixture for the truncated triangular domain.
    """

    with pytest.raises(RuntimeError, match='not in the domain'):
        interface('constant', 0.75, 0.5)


def test_zero_field_has_no_normalized_profile(interface: TwoDInterface):
    """Checks that a zero field integrates to zero but cannot be normalized.

    args:
        interface: Interface fixture to which a zero scalar field is added.
    """

    interface.mesh.point_data['zero'] = np.zeros(interface.mesh.n_points)
    assert_allclose(interface.cs_integral('zero', 0.5), 0.0)
    with pytest.raises(ValueError, match='Normalized profile is undefined'):
        interface.radial_profile('zero', 0.5)


@pytest.mark.parametrize('overwrite', [False, True])
def test_radius_file_reuse(interface: TwoDInterface, tmp_path: Path,
                           monkeypatch: pytest.MonkeyPatch, overwrite: bool):
    """Checks loading without measurement and saving over an existing file.

    args:
        interface: Interface fixture for the truncated triangular domain.
        tmp_path: Temporary output directory supplied by pytest.
        monkeypatch: Fixture for replacing the radius measurement method.
        overwrite: Whether the saved radius should be recomputed.
    """

    folder = str(tmp_path)
    z, original = save_torch_radius(interface, 'radius.h5', 5,
                                    folder=folder, z_max=0.8)
    replacement = np.full(5, 0.6)
    measure = Mock(return_value=replacement)
    monkeypatch.setattr(interface, 'torch_radius', measure)

    if overwrite:
        saved_z, radius = save_torch_radius(interface, 'radius.h5', 5,
                                            folder=folder, z_max=0.8)
    else:
        saved_z, radius = load_torch_radius('radius.h5', folder=folder)

    expected = replacement if overwrite else original

    assert_allclose(saved_z, z)
    assert_allclose(radius, expected)
    assert_equal(measure.call_count, int(overwrite))

    # Reopening the file also confirms that an overwrite was persisted.
    _, reloaded = load_torch_radius('radius.h5', folder=folder)
    assert_allclose(reloaded, expected)
    assert_equal(measure.call_count, int(overwrite))
