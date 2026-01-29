"""Class to open a .pvtu file or equivalent and evaluate quantities relevant to
the 1-D governing equations."""

from typing import Callable

import numpy as np

import h5py

import pyvista as pv

# Torch length from 2-D inlet condition to end of nozzle [m]
TORCH_LENGTH = 0.34

# Default number of points for radial sampling
N_POINTS = 100

# Operator acting on UnstructuredGrid
UnstructuredGridOperator = Callable[[pv.UnstructuredGrid], pv.UnstructuredGrid]


def _make_line(start: np.ndarray, end: np.ndarray, parameterized: np.ndarray) \
     -> pv.PolyData:
    """Forms a line with non-uniform sampling.

    args:
        start: Position of line start
        end: Position of line end
        parameterized: Sampling parameter in [0, 1]

    returns:
        The line with sample points
    """

    if start.ndim != 1:
        raise ValueError("start must be 1-D")

    if start.size != 3:
        raise ValueError("start must have 3 entries")

    if end.ndim != 1:
        raise ValueError("end must be 1-D")

    if end.size != 3:
        raise ValueError("end must have 3 entries")

    if parameterized.ndim != 1:
        raise ValueError("parameterized must be 1-D")

    if np.any(parameterized != np.unique(parameterized)):
        raise ValueError("parameterized must be increasing without repeats")

    return pv.PolyData(start + np.outer(parameterized, end - start))


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

    @property
    def z_min(self) -> float:
        """Minimum axial position in mesh. May not be zero since the inlet is
        truncated."""
        return self.mesh.bounds[2]

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

        z = np.linspace(self.z_min, z_max, n_points)
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

    def radial_profile(self, field: str, z: float, n_points: int = N_POINTS) \
            -> tuple[np.ndarray, float]:
        """Evaluates the radial profile of a field at the given axial position.
        The profile, f, of a quantity, q, is the normalized, dimensionless
        function

            f(r, z) = pi * R(z)^2 * q(r, z) / ⟨q⟩(z).

        The cross-sectional integral

            ⟨q⟩(z) = 2 * pi * int_0^R(z) q(r, z) * r dr

        is evaluated via trapezoidal rule.

        args:
            field: Field name
            z: Axial position
            n_points: Number of uniform sample points along the line, optional.

        returns:
            The profile values,
            the cross-sectional integral
        """

        self._field_check(field)

        mesh = self._mesh

        torch_r = self.torch_radius(z)

        r_hat = np.linspace(0.0, 1.0, n_points)

        r_pts = r_hat*torch_r

        # (NOTE): Line is formed based on normalized radius coordinates
        #         for consistency

        line = _make_line(np.array([0.0, z, 0.0]),
                          np.array([torch_r, z, 0.0]),
                          r_hat)

        field_values = line.sample(mesh).point_data[field]

        integral = 2*np.pi*np.trapezoid(r_pts*field_values, r_pts)

        prof = np.pi*torch_r**2*field_values/integral

        return prof, integral

    def cs_integral(self, field: str, z: float, n_points: int = N_POINTS) \
            -> float | np.ndarray:
        """Evaluates the cross-sectional integral of a field at the given axial
        position. See `radial_profile` for details.

        args:
            field: Field name
            z: Axial position
            n_points: Number of sample points along the line, optional.

        returns:
            The radial integral
        """
        return self.radial_profile(field, z, n_points)[1]


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


def step_finder(tdi: TwoDInterface, z_l: float, z_r: float,
                verbose: bool = True) -> float:
    """Locates the step by finding the axial location with the maximum
    torch radius derivative within a given window.

    args:
        tdi: Interface to 2-D dataset
        z_l: Left bound of step region
        z_r: Right bound of step region
        verbose: Indicator to print the result, optional. Default is True.

    returns:
        The location of maximum torch radius derivative
    """

    if z_l >= z_r:
        raise ValueError("z_l < z_r required")

    # Coarse search
    z = np.linspace(z_l, z_r, 100)

    r = np.array([tdi.torch_radius(z_) for z_ in z])

    dr_dz = np.gradient(r, z, edge_order=2)

    loc = np.argmax(np.abs(dr_dz))

    # Refined search
    z_ref = np.linspace(z[loc-1], z[loc+1], 100)

    r = np.array([tdi.torch_radius(z_) for z_ in z_ref])

    dr_dz = np.gradient(r, z_ref, edge_order=2)

    loc = np.argmax(np.abs(dr_dz))

    z_step = z_ref[loc]

    if verbose:
        print(f"Location of step: {100*z_step:.4f} [cm]")

    return z_ref[loc]
