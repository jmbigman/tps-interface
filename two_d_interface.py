"""Class to open a .pvtu file or equivalent and evaluate the terms in a 1-D
governing equation."""

import numpy as np
from scipy.integrate import simpson

import pyvista as pv

# (NOTE): The SciPy implementation of Simpson's rule allows for an even number
#         of sample points, but it is less accurate, so an odd number is
#         enforced.


def _simpson_check(n_points: int) -> None:
    """Checks the number of points for Simpson's rule.

    args:
        n_points: Number of sample points
    """

    if n_points % 2 == 0:
        msg = "Simpson's rule requires an odd number of sample points"
        raise ValueError(msg)

    if not isinstance(n_points, int):
        msg = "n_points must be an integer"
        raise TypeError(msg)

    if n_points < 2:
        msg = "n_points must be at least two"
        raise ValueError(msg)


class TwoDInterface:
    """Calculates radial integrals and profiles from 2-D data in a .pvtu file.

    args:
        file: .pvtu file or equivalent
        rtol: Relative tolerance convergence criterion for torch radius
    """

    # Cache torch radii
    _radius_cache = {}

    def __init__(self, file: str, rtol: float = 1e-7):

        # Load file with PyVista
        self._mesh = pv.read(file)

        # (NOTE): rtol = 1e-7 is found to generally work well.
        #         Smaller tolerances do not always converge
        self._rtol = rtol

    @property
    def field_names(self) -> None:
        """Names of point data fields"""
        return self._mesh.point_data.keys()

    def _field_check(self, field: str) -> None:
        """Checks that a field is in the dataset.

        args:
            field: Field name
        """

        if field not in self.field_names:
            msg = "Specified field is not in the dataset"
            raise RuntimeError(msg)

    def torch_radius(self, z: float) -> float:
        """Finds the torch radius at one axial position by bisection. Returns
        a point guaranteed to be inside the domain, but it is possible it is
        not exactly at the torch wall.

        args:
            z: Axial position

        returns:
            The torch radius
        """

        radius_cache = self._radius_cache

        if z in radius_cache:
            return radius_cache[z]

        mesh = self._mesh

        z_min, z_max = mesh.bounds[2:4]
        if z < z_min or z > z_max:
            msg = f"Axial position ({z}) is outside domain axial bounds: " \
                + f"{z_min}, {z_max}"
            raise RuntimeError(msg)

        # Base step: define end points
        r_l = 0.0
        r_r = 1.1*np.max(mesh.points[:, 0])

        avg = 0.5*(r_l + r_r)
        dist = r_r - r_l

        # Iterative step:
        #   If midpoint is outside domain, replace right end
        #   If midpoint is inside domain, replace left end
        # Until convergence is achieved
        while dist/avg > self._rtol:

            cell_idx = mesh.find_containing_cell([avg, z, 0.0])

            if cell_idx == -1:
                r_r = avg
            else:
                r_l = avg

            avg = 0.5*(r_l + r_r)
            dist = r_r - r_l

        radius_cache[z] = r_l

        # Left point is always in the domain
        return r_l

    def __call__(self, field: str, r: float, z: float) -> float | np.ndarray:
        """Evaluates a field at the given position.

        args:
            field: Field name
            r: Radial position
            z: Axial position

        returns:
            The evaluated field
        """

        mesh = self._mesh

        point = [r, z, 0.0]

        if mesh.find_containing_cell(point) == -1:
            msg = "Specified point is not in the domain"
            raise RuntimeError(msg)

        self._field_check(field)

        probe = pv.PolyData(point)

        field_value = probe.sample(mesh).point_data[field][0]

        return field_value

    def wall_value(self, field: str, z: float) -> float | np.ndarray:
        """Evaluates a field at the wall for a given axial position.

        args:
            field: Field name
            z: Axial position

        returns:
            The field evaluated at the wall
        """

        r = self.torch_radius(z)

        wall_value = self(field, r, z)

        return wall_value

    def area_integral(self, field: str, z: float, n_points: int = 101) \
            -> float | np.ndarray:
        """Evaluates the cross-sectional area integral of a field at the given
        axial position, under the assumption of axisymmetry. For a function f,
        the integral is

            ⟨f⟩(z) = 2 * pi * int_0^R(z) f(r) * r dr

        and is evaluated via Simpson's rule.

        args:
            field: Field name
            z: Axial position
            n_points: Number of sample points along the line, must be odd

        returns:
            The radial integral
        """

        _simpson_check(n_points)

        self._field_check(field)

        mesh = self._mesh

        r = self.torch_radius(z)

        line = pv.Line([0.0, z, 0.0], [r, z, 0.0], n_points - 1)

        r_pts = line.points[:, 0]
        field_values = line.sample(mesh).point_data[field]

        integral = 2*np.pi*simpson(r_pts*field_values, r_pts)

        return integral

    def linear_average(self, field: str, z: float, n_points: int = 101) \
            -> float | np.ndarray:
        """Evaluates the linear average of a field over [0, R] at the given
        axial position (no weighting by the radius). For a function f, the
        average is

            f̅(z) = (1 / R(z)) * int_0^R(z) f(r) dr

        and is evaluated via Simpson's rule. This is essentially a regular
        average if the cylindrical geometry is ignored.

        args:
            field: Field name
            z: Axial position
            n_points: Number of sample points along the line, must be odd

        returns:
            The linear average
        """

        _simpson_check(n_points)

        self._field_check(field)

        mesh = self._mesh

        r = self.torch_radius(z)

        line = pv.Line([0.0, z, 0.0], [r, z, 0.0], n_points - 1)

        r_pts = line.points[:, 0]
        field_values = line.sample(mesh).point_data[field]

        average = simpson(field_values, r_pts) / r

        return average

    def radial_profile(self, field: str, z: float, n_points: int = 101) \
            -> np.ndarray:
        """Evaluates the radial profile of a field at the given axial position.
        The profile of a quantity q is the normalized function

            f(r) = q(r)/⟨q⟩

        The area integral is evaluated via Simpson's rule, as in
        `area_integral`.

        args:
            field: Field name
            z: Axial position
            n_points: Number of sample points along the line, must be odd

        returns:
            A 2-D NumPy array. The first column is the radial position and the
            second is the profile.
        """

        _simpson_check(n_points)

        self._field_check(field)

        mesh = self._mesh

        r = self.torch_radius(z)

        line = pv.Line([0.0, z, 0.0], [r, z, 0.0], n_points - 1)

        r_pts = line.points[:, 0]
        field_values = line.sample(mesh).point_data[field]

        integral = 2*np.pi*simpson(r_pts*field_values, r_pts)

        out = np.column_stack((r_pts, field_values/integral))

        return out
