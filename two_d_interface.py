"""Class to open a .pvtu file or equivalent and evaluate quantities relevant to
the 1-D governing equations."""

from typing import Callable

import numpy as np

import h5py

import pyvista as pv

TORCH_LENGTH = 0.32

# Operator acting on UnstructuredGrid
UnstructuredGridOperator = Callable[[pv.UnstructuredGrid], pv.UnstructuredGrid]


class TwoDInterface:
    """Calculates radial integrals and profiles from 2-D data in an
    UnstructuredGrid. For generality, this works with data in memory; it does
    not load files.

    args:
        mesh: The 2-D data (for a single time point)
        rtol: Relative tolerance convergence criterion for torch radius
    """

    # Cache torch radii
    _radius_cache = {}

    def __init__(self, mesh: pv.UnstructuredGrid = None, rtol: float = 1e-7):

        self._mesh = mesh
        self._rtol = rtol

        if rtol < 1e-7:
            print('rtol of 1e-7 generally works well. Smaller tolerances do '
                  + 'not always converge.')

    @property
    def mesh(self) -> pv.UnstructuredGrid:
        """Current mesh"""
        return self._mesh

    def set_mesh(self, mesh: pv.UnstructuredGrid) -> None:
        """Sets a new mesh.

        args:
            mesh: The new 2-D data
        """
        self._mesh = mesh

    def clear_radius_cache(self) -> None:
        """Resets the torch radius cache."""
        self._radius_cache = {}

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

    def save_torch_radius(self, filename: str, n_points: int,
                          z_max: float = TORCH_LENGTH) -> None:
        """Saves the torch radius as an HDF5 file.

        args:
            filename: Name of HDF5 file
            n_points: Number of axial points
            z_max: Maximum axial position
        """

        z = np.linspace(0.0, z_max, n_points)
        r = np.array([self.torch_radius(z_) for z_ in z], dtype=float)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('axial position', data=z)
            f.create_dataset('torch radius', data=r)
            f.attrs['units'] = 'm'

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

    def cs_integral(self, field: str, z: float, n_points: int = 101) \
            -> float | np.ndarray:
        """Evaluates the cross-sectional integral of a field at the given axial
        position, under the assumption of axisymmetry. For a function f, the
        integral is

            ⟨f⟩(z) = 2 * pi * int_0^R(z) f(r, z) * r dr

        and is evaluated via trapezoidal rule.

        args:
            field: Field name
            z: Axial position
            n_points: Number of sample points along the line, must be odd

        returns:
            The radial integral
        """

        self._field_check(field)

        mesh = self._mesh

        r = self.torch_radius(z)

        line = pv.Line([0.0, z, 0.0], [r, z, 0.0], resolution=n_points-1)

        r_pts = line.points[:, 0]
        field_values = line.sample(mesh).point_data[field]

        integral = 2*np.pi*np.trapezoid(r_pts*field_values, r_pts)

        return integral

    def radial_profile(self, field: str, z: float, n_points: int = 101) \
            -> np.ndarray:
        """Evaluates the radial profile of a field at the given axial position.
        The profile of a quantity q is the normalized, dimensionless function

            f(r, z) = pi * R(z)^2 * q(r, z) / ⟨q⟩(z)

        The area integral is evaluated via trapezoial rule, as in
        `area_integral`.

        args:
            field: Field name
            z: Axial position
            n_points: Number of sample points along the line, must be odd

        returns:
            A 2-D NumPy array. The first column is the radial position and the
            second is the profile.
        """

        self._field_check(field)

        mesh = self._mesh

        r = self.torch_radius(z)

        line = pv.Line([0.0, z, 0.0], [r, z, 0.0], resolution=n_points-1)

        r_pts = line.points[:, 0]
        field_values = line.sample(mesh).point_data[field]

        integral = 2*np.pi*np.trapezoid(r_pts*field_values, r_pts)

        out = np.column_stack((r_pts, np.pi*r**2*field_values/integral))

        return out


def time_statistics(reader: pv.PVDReader, t1: int = 0, t2: int = -1,
                    field_names: list[str] = None,
                    operator: UnstructuredGridOperator = lambda x: x) \
                        -> pv.UnstructuredGrid:
    """Calculates the average and standard deviation of fields in a .pvd
    dataset inside the given time range.

    args:
        reader: .pvd file reader
        t1: Index of first time points, optional. Default is 0.
        t2: Index of last time point, optional. Default is -1.
        field_names: List of field names to use, optional. Default is all.
        operator: Operator to apply at each time point before calculating the
                  statistics, optional. Default is identity.

    returns:
        Averages and standard deviations of the fields. Field names are
        appended with '_avg' or '_std'
    """

    if t2 == -1:
        t2 = len(reader.time_values) - 1

    ###########################################################################
    # First time point
    ###########################################################################

    reader.set_active_time_point(t1)

    mesh = operator(reader.read()[0])

    if field_names is None:
        field_names = mesh.point_data.keys()

    # Initialize empty
    out = pv.UnstructuredGrid(mesh.cells, mesh.celltypes, mesh.points)

    data = {}

    for fn in field_names:

        data[fn] = [mesh.point_data[fn]]

    ###########################################################################
    # Iterate over remaining time points
    ###########################################################################

    for t in range(t1+1, t2+1):

        reader.set_active_time_point(t)

        mesh = operator(reader.read()[0])

        for fn in field_names:

            data[fn].append(mesh.point_data[fn])

    ###########################################################################
    # Calculate statistics
    ###########################################################################

    t_pts = np.array(reader.time_values[t1:t2+1])
    t_int = t_pts[-1] - t_pts[0]

    for fn in field_names:

        field_data = np.stack(data[fn], axis=0)

        avg = np.trapezoid(field_data, t_pts, axis=0)/t_int

        std = np.trapezoid((field_data - avg)**2, t_pts, axis=0)/t_int
        std = np.sqrt(std)

        out.point_data[fn+'_avg'] = avg
        out.point_data[fn+'_std'] = std

    return out
