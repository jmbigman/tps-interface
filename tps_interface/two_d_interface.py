"""Class to open a .pvtu file or equivalent and evaluate quantities relevant to
the 1-D governing equations."""

from os import makedirs
from os.path import exists, join

from typing import Callable, Literal

import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

import h5py

import pyvista as pv

# This is a generic array annotation library, not just for JAX
from jaxtyping import Real

from .models import Model
from ._utils import TORCH_LENGTH

Array = np.ndarray


# --------------------------------------------------------------------------- #
# This class is the main driver and contains all basic functionality
# --------------------------------------------------------------------------- #

# Default number of points for radial sampling
N_POINTS = 100


def _make_line(start: Real[Array, "3"], end: Real[Array, "3"],
               parameter: Real[Array, " n_pts"]) -> pv.PolyData:
    """Forms a line with non-uniform sampling.

    args:
        start: Position of line start (x, y, z)
        end: Position of line end (x, y, z)
        parameter: Sampling parameter in [0, 1]. Determines where the line
                   segment between the start and end points is sampled.

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

    if parameter.ndim != 1:
        raise ValueError("parameter must be 1-D")

    if np.any(parameter != np.unique(parameter)):
        raise ValueError("parameter must be increasing without repeats")

    return pv.PolyData(start + np.outer(parameter, end - start))


# (NOTE): The `TwoDInterface` methods are vectorized using NumPy vectorize,
#         which is convenient but not particularly efficient


class TwoDInterface:
    """Calculates radial integrals and profiles from 2-D data in an
    UnstructuredGrid. For generality, this works with data in memory; it does
    not load files.

    args:
        mesh: The 2-D data (for a single time point)
        rtol: Relative tolerance convergence criterion for torch radius
        n_points: Number of points to use in radial sampling of fields,
                  optional. Default is 100.
        norm_rtol: Non-negative cancellation tolerance for profile
                   normalization, relative to the integral of the absolute
                   field. Default is 1e-12.
    """

    # Cache torch radii
    _radius_cache: dict[float, float]

    n_points: int

    _r_norm: Real[Array, "n_points"]

    def __init__(self, mesh: pv.UnstructuredGrid = None, rtol: float = 1e-7,
                 n_points: int = N_POINTS, norm_rtol: float = 1e-12):

        if not np.isfinite(norm_rtol) or norm_rtol < 0:
            raise ValueError("norm_rtol must be finite and nonnegative")

        self._mesh = mesh
        self._rtol = rtol
        self._norm_rtol = norm_rtol
        self._radius_cache = {}

        if rtol < 1e-7:
            print('rtol of 1e-7 generally works well. Smaller tolerances do '
                  + 'not always converge.')

        self.n_points = n_points

        if not isinstance(n_points, int):
            raise TypeError("n_points must be an integer")

        if n_points < 3:
            raise ValueError("n_points must be at least 3")

        self._r_norm = np.linspace(0.0, 1.0, n_points)

        # Vectorize methods
        self.torch_radius = np.vectorize(self.torch_radius)
        self.wall_value = np.vectorize(self.wall_value, excluded=[0])
        self.radial_sample = np.vectorize(self.radial_sample, excluded=[0],
                                          signature='()->(n)')
        self.radial_profile = np.vectorize(self.radial_profile, excluded=[0],
                                           signature='()->(n),()')
        self.cs_integral = np.vectorize(self.cs_integral, excluded=[0])

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
        self.clear_radius_cache()

    def clear_radius_cache(self) -> None:
        """Resets the torch radius cache."""
        self._radius_cache = {}

    @property
    def field_names(self) -> list[str]:
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

        field_data = self._mesh.point_data[field]

        if field_data.ndim != 1:
            raise NotImplementedError("Only scalar fields are currently"
                                      + f" supported; field {field} has shape"
                                      + f" {field_data.shape}.")

    def torch_radius(self, z: Real[Array, "..."]) -> Real[Array, "..."]:
        """Finds the torch radius at one axial position by bisection. Returns
        a point guaranteed to be inside the domain, but it is possible it is
        not exactly at the torch wall.

        args:
            z: Axial position

        returns:
            the torch radius
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

    def __call__(self, field: str, r: float, z: float) -> Real[Array, "..."]:
        """Evaluates a field at the given position.

        args:
            field: Field name
            r: Radial position
            z: Axial position

        returns:
            the evaluated field
        """

        mesh = self._mesh

        point = [r, z, 0.0]

        if mesh.find_containing_cell(point) == -1:
            msg = "Specified point is not in the domain"
            raise RuntimeError(msg)

        self._field_check(field)

        probe = pv.PolyData(point)

        sampled = probe.sample(mesh)
        valid = np.asarray(sampled.point_data['vtkValidPointMask'])

        if not np.all(valid == 1):
            raise RuntimeError(f"Sampling field {field} failed at"
                               + f" (r, z) = ({r}, {z})")

        field_value = sampled.point_data[field][0]

        return field_value

    def wall_value(self, field: str, z: Real[Array, "..."]
                   ) -> Real[Array, "..."]:
        """Evaluates a field at the wall for a given axial position.

        args:
            field: Field name
            z: Axial position

        returns:
            the wall evaluation
        """

        r = self.torch_radius(z)

        wall_value = self(field, r, z)

        return wall_value

    @property
    def r_norm(self) -> Real[Array, "n_points"]:
        """Uniformly sampled normalized radius."""
        return self._r_norm

    def radial_sample(self, field: str, z: float
                      ) -> Real[Array, "n_points ..."]:
        """Samples a given field over the torch radius at the given axial
        position. The field is not normalized or processed in any way.

        args:
            field: Field name
            z: Axial position

        returns:
            the field values
        """

        self._field_check(field)

        mesh = self._mesh

        torch_r = self.torch_radius(z)

        # (NOTE): Line is formed based on normalized radius coordinates
        #         for consistency

        line = _make_line(np.array([0.0, z, 0.0]),
                          np.array([torch_r, z, 0.0]),
                          self.r_norm)

        sampled = line.sample(mesh)
        valid = np.asarray(sampled.point_data['vtkValidPointMask'])

        if not np.all(valid == 1):
            invalid_r = (torch_r*self.r_norm)[valid != 1]

            raise RuntimeError(f"Sampling field {field} failed at"
                               + f" (r, z) = ({invalid_r}, {z})")

        field_values = sampled.point_data[field]

        return np.asarray(field_values)

    def radial_profile(self, field: str, z: float
                       ) -> tuple[Real[Array, "n_points"], float]:
        """Evaluates the radial profile of a field at the given axial position.
        The profile, f, of a quantity, q, is the normalized, dimensionless
        function

            f(r, z) = π * R(z)^2 * q(r, z) / ⟨q⟩(z).

        The cross-sectional integral

            ⟨q⟩(z) = 2π * int_0^R(z) q(r, z) * r dr

        is evaluated via trapezoidal rule.

        args:
            field: Field name
            z: Axial position

        returns:
            the profile values,
            the cross-sectional integral

        raises:
            ValueError: The integral is nonfinite, zero, or too small relative
                        to the integral of the absolute field, as controlled
                        by norm_rtol.
        """

        field_values = self.radial_sample(field, z)

        # The torch radius is cached, so this isn't an additional computation
        torch_r = self.torch_radius(z)

        r_pts = torch_r*self.r_norm

        # Integrate and normalize
        integral = 2*np.pi*np.trapezoid(r_pts*field_values, r_pts)

        abs_integral = 2*np.pi*np.trapezoid(r_pts*np.abs(field_values), r_pts)

        if ((not np.isfinite(integral)
             or not np.isfinite(abs_integral)
             or abs_integral == 0
             or abs(integral) <= self._norm_rtol*abs_integral)):

            raise ValueError("Normalized profile is undefined or"
                             + f" ill-conditioned for field {field}"
                             + f" at z={z}: integral={integral},"
                             + f" absolute-field integral={abs_integral},"
                             + f" norm_rtol={self._norm_rtol}")

        prof = np.pi*torch_r**2*field_values/integral

        return prof, integral

    def cs_integral(self, field: str, z: float) -> float:
        """Evaluates the cross-sectional integral of a field at the given axial
        position.

        args:
            field: Field name
            z: Axial position

        returns:
            the radial integral
        """

        field_values = self.radial_sample(field, z)
        r_pts = self.torch_radius(z)*self.r_norm

        return 2*np.pi*np.trapezoid(r_pts*field_values, r_pts)


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #

# Operator acting on UnstructuredGrid
UnstructuredGridOperator = Callable[[pv.UnstructuredGrid], pv.UnstructuredGrid]


def time_statistics(reader: pv.PVDReader, t1: int = 0, t2: int | None = None,
                    field_names: list[str] = None,
                    operator: UnstructuredGridOperator = lambda x: x) \
                        -> pv.UnstructuredGrid:
    """Calculates the average and standard deviation of fields in a .pvd
    dataset inside the given time range. At least two time points with finite,
    strictly increasing timestamps are required.

    args:
        reader: .pvd file reader
        t1: Inclusive index of first time point, optional. Default is 0.
        t2: Inclusive index of last time point, optional. Default is None,
            meaning the last available time point.
        field_names: List of field names to use, optional. Default is all.
        operator: Operator to apply at each time point before calculating the
                  statistics, optional. Default is identity.

    returns:
        Averages and standard deviations of the fields. Field names are
        appended with '_avg' or '_std'
    """

    if isinstance(t1, bool) or not isinstance(t1, (int, np.integer)):
        raise TypeError("t1 must be an integer index")

    if t2 is not None and (isinstance(t2, bool)
                           or not isinstance(t2, (int, np.integer))):
        raise TypeError("t2 must be an integer index or None")

    time_values = np.asarray(reader.time_values)
    n_times = len(time_values)

    if t2 is None:
        t2 = n_times - 1

    if not (0 <= t1 < t2 < n_times):
        raise ValueError("Time statistics require at least two time points"
                         + f" with 0 <= t1 < t2 < {n_times};"
                         + f" got t1={t1}, t2={t2}")

    t_pts = time_values[t1:t2+1]

    if not np.all(np.isfinite(t_pts)) or not np.all(np.diff(t_pts) > 0):
        raise ValueError("Selected timestamps must be finite and strictly "
                         "increasing")

    t_int = t_pts[-1] - t_pts[0]

    # ----------------------------------------------------------------------- #
    # First time point
    # ----------------------------------------------------------------------- #

    reader.set_active_time_point(t1)

    mesh = operator(reader.read()[0])

    # Copy the reference geometry so subsequent reads cannot modify it
    reference_points = mesh.points.copy()
    reference_cells = mesh.cells.copy()
    reference_celltypes = mesh.celltypes.copy()

    if field_names is None:
        field_names = mesh.point_data.keys()

    # Initialize empty
    out = pv.UnstructuredGrid(reference_cells, reference_celltypes,
                              reference_points)

    data = {}

    for fn in field_names:

        data[fn] = [mesh.point_data[fn]]

    # ----------------------------------------------------------------------- #
    # Iterate over remaining time points
    # ----------------------------------------------------------------------- #

    for t in range(t1+1, t2+1):

        reader.set_active_time_point(t)

        mesh = operator(reader.read()[0])

        same_mesh = (np.array_equal(mesh.points, reference_points)
                     and np.array_equal(mesh.cells, reference_cells)
                     and np.array_equal(mesh.celltypes, reference_celltypes))

        if not same_mesh:
            raise ValueError(f"Mesh changed at time-point index {t} relative"
                             + f" to first index {t1}. Time statistics require"
                             + " a fixed mesh.")

        for fn in field_names:

            data[fn].append(mesh.point_data[fn])

    # ----------------------------------------------------------------------- #
    # Calculate statistics
    # ----------------------------------------------------------------------- #

    for fn in field_names:

        field_data = np.stack(data[fn], axis=0)

        avg = np.trapezoid(field_data, t_pts, axis=0)/t_int

        std = np.trapezoid((field_data - avg)**2, t_pts, axis=0)/t_int
        std = np.sqrt(std)

        out.point_data[fn+'_avg'] = avg
        out.point_data[fn+'_std'] = std

    return out


# --------------------------------------------------------------------------- #
# Torch radius analysis
# --------------------------------------------------------------------------- #


def step_finder(tdi: TwoDInterface, z_l: float, z_r: float,
                rtol: float, verbose: bool = True) -> float:
    """Locates the step using a binary search-like algorithm within a given
    window.

    args:
        tdi: Interface to 2-D dataset
        z_l: Left bound of step region
        z_r: Right bound of step region
        rtol: Relative tolerance for step location, compared to original window
        verbose: Indicator to print the result, optional. Default is True.

    returns:
        the location of maximum torch radius derivative
    """

    if z_l >= z_r:
        raise ValueError("z_l < z_r required")

    tr_l = tdi.torch_radius(z_l)
    tr_r = tdi.torch_radius(z_r)

    if np.allclose(tr_l, tr_r, 0.0, 1e-12):
        raise ValueError("R(z_l) != R(z_r) required")

    dz = z_r - z_l

    z_avg = 0.5*(z_l + z_r)
    tr_avg = tdi.torch_radius(z_avg)

    while z_r - z_l > rtol*dz:

        # Move boundary to whichever side the radius is closer to
        if np.abs(tr_avg - tr_l) < np.abs(tr_avg - tr_r):
            z_l = z_avg
        else:
            z_r = z_avg

        z_avg = 0.5*(z_l + z_r)
        tr_avg = tdi.torch_radius(z_avg)

    if verbose:
        print(f"Location of step: {100*z_avg:.4f} [cm]")

    return z_avg


def save_torch_radius(tdi: TwoDInterface, filename: str, n_points: int,
                      *options, folder: str = 'output',
                      z_max: float = TORCH_LENGTH,
                      mode: Literal[None, 'no_step', 'smooth'] = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Saves the torch radius as an HDF5 file, with options for
    post-processing.

    args:
        tdi: Interface to 2-D dataset
        filename: Name of HDF5 file
        n_points: Number of axial points, at least 3
        *options: Additional argument(s) for post-processing mode.
        folder: Output directory, optional. Default is 'output'.
        z_max: Maximum axial position, greater than `tdi.z_min` and no greater
               than the mesh's maximum axial position
        mode: Post-processing mode, optional. `None` saves the torch radius as
              is. 'no_step' ignores the step region and requires the left and
              right boundaries, exclusive, of the step region to be passed as a
              2 element iterable of floats to *options. 'smooth' applies a
              Gaussian smoothing filter to the torch radius and requires the
              FWHM to be passed as a float to *options. The smoothing filter
              truncates the Gaussian at 3 standard deviations. Default is
              `None`.

    returns:
        the axial positions [m]
        the torch radius [m]
    """

    if (isinstance(n_points, bool)
            or not isinstance(n_points, (int, np.integer))):

        raise TypeError("n_points must be an integer")

    if n_points < 3:
        raise ValueError("n_points must be at least 3")

    z_min = tdi.z_min
    mesh_z_max = tdi.mesh.bounds[3]

    if not np.isfinite(z_max) or not (z_min < z_max <= mesh_z_max):
        raise ValueError(f"z_max must be finite and satisfy {z_min} < z_max <="
                         + f"{mesh_z_max}; got {z_max}")

    z = np.linspace(z_min, z_max, n_points)

    if mode == 'no_step':

        if len(options) != 2:
            raise ValueError("options must have 2 elements for \'no_step\'")

        z_l, z_r = options

        if not isinstance(z_l, float) or not isinstance(z_r, float):
            raise TypeError("options must be floats for \'no_step\'")

        if not np.isfinite(z_l) or not np.isfinite(z_r) or z_l >= z_r:
            raise ValueError("'no_step' boundaries must be finite with"
                             + " z_l < z_r")

        inside_step = (z > z_l) & (z < z_r)

        if np.count_nonzero(inside_step) == 0:
            raise ValueError("No axial samples lie inside the 'no_step'"
                             + " interval.")

    r = tdi.torch_radius(z)

    if mode is None:

        r_d = np.gradient(r, z, edge_order=2)

        extra_attrs = {}
        pass

    elif mode == 'no_step':

        # Use straight line over the region
        r[inside_step] = r[inside_step][0]

        r_d = np.gradient(r, z, edge_order=2)

        extra_attrs = {'mode': 'no_step',
                       'left boundary': z_l,
                       'right boundary': z_r}

    elif mode == 'smooth':

        if len(options) != 1:
            raise ValueError("options must have 1 element for \'smooth\'")

        fwhm = options[0]

        if not isinstance(fwhm, float):
            raise TypeError("FWHM must be a float")

        if fwhm <= 0:
            raise ValueError("FWHM must be positive")

        dz = z[1] - z[0]

        # Convert from FWHM in physical units to standard deviation in data
        # units
        sig_d = fwhm/(2*np.sqrt(2*np.log(2))*dz)

        r_sm = gaussian_filter1d(r, sig_d, mode='nearest', truncate=3)

        # Differentiate the original radius at the same smoothing width.
        r_d = gaussian_filter1d(r, sig_d, order=1, mode='nearest', truncate=3)
        r_d /= dz

        extra_attrs = {'mode': 'smooth',
                       'fwhm': fwhm}

    else:
        raise KeyError('Post-processing mode must be None, \'no-step\', or '
                       + '\'smooth\'')

    r_out = r_sm if mode == 'smooth' else r
    if not exists(folder):
        makedirs(folder)

    filename = join(folder, filename)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('axial position', data=z)
        f.create_dataset('torch radius', data=r_out)
        f.create_dataset('torch radius derivative', data=r_d)
        f.attrs['units'] = 'm'

        f.attrs.update(extra_attrs)

    return z, r_out


def load_torch_radius(filename: str, folder: str = 'output'
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Loads axial positions and torch radii from a saved HDF5 file.

    args:
        filename: Name of the HDF5 file created by `save_torch_radius`
        folder: Input directory, optional. Default is 'output'.

    returns:
        The axial positions [m] and torch radii [m].
    """

    with h5py.File(join(folder, filename), 'r') as f:
        return f['axial position'][:], f['torch radius'][:]


# --------------------------------------------------------------------------- #
# Sampling and curve-fitting along axial direction
# --------------------------------------------------------------------------- #


def relative_error(r_norm: Real[Array, " _"], data: Real[Array, " _"],
                   model: Real[Array, " _"]) -> float:
    """Calculates the L2 relative error of the model fit. A radial factor is
    included so the comparison is over the whole cross-section. The trapezoidal
    rule is used for integration.

    args:
        r_norm: Normalized radius evaluation points
        data: Data values
        model: Model profile values

    returns:
        the L2 relative error
    """

    # Norm squared difference
    abs_diff = np.trapezoid((data - model)**2*r_norm, r_norm)
    # Norm squared of data
    scale = np.trapezoid(data**2*r_norm, r_norm)

    return np.sqrt(abs_diff/scale)


def fit_profile(tdi: TwoDInterface, field: str, z: Real[Array, " _"],
                model: Model, **kwargs) -> tuple[Real[Array, " _"],
                                                 Real[Array, " _"],
                                                 Real[Array, " _"],
                                                 Real[Array, " _"]]:
    """Samples the dimensionless radial profiles and curve fits the
    parameterized model at the axial positions.

    args:
        tdi: Interface to 2-D dataset
        field: Field name (to fit against)
        z: Axial positions [m]
        model: Model to curve fit
        **kwargs: Passed through to SciPy `curve_fit`, overriding default
                  optimizer settings when supplied (e.g. p0 and bounds).

    returns:
        the radial profiles,
        the optimal parameters at the axial positions
        the relative errors,
        the cross-sectional integrals
    """

    if not model.profile:
        raise ValueError("model must be a dimensionless radial profile")

    profs, integrals = tdi.radial_profile(field, z)

    r_norm = tdi.r_norm

    params_list = []
    rel_err_list = []

    fit_options = dict(method='dogbox', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    fit_options.update(kwargs)

    for prof in profs:

        # Error norm should be weighted by the radius for cross-sectional
        # integral comparison
        # `curve_fit` squares residuals divided by sigma, so use 1/sqrt(r)
        # to weight squared residuals by r. Infinite sigma excludes the axis.
        with np.errstate(divide='ignore'):
            weights = 1.0/np.sqrt(r_norm)

        cf_result = curve_fit(model, r_norm, prof, sigma=weights,
                              **fit_options)

        model_eval = model(r_norm, *cf_result[0])

        params_list.append(cf_result[0])
        rel_err_list.append(relative_error(r_norm, prof, model_eval))

    return profs, np.array(params_list), np.array(rel_err_list), integrals


def fit_deviation(tdi: TwoDInterface, field: str, z: Real[Array, " _"],
                  model: Model, **kwargs) -> tuple[Real[Array, " _"],
                                                   Real[Array, " _"],
                                                   Real[Array, " _"]]:
    """Samples the deviation of the quantity radially from its wall value and
    curve fits the normalized parameterized model at the axial positions.

    args:
        tdi: Interface to 2-D dataset
        field: Field name (to fit against)
        z: Axial positions [m]
        model: Model to curve fit
        **kwargs: Passed through to SciPy `curve_fit`

    returns:
        the radially sampled deviation,
        the optimal parameters at the axial positions,
        the relative errors,
        the maximum sampled values
    """

    if model.profile:
        raise ValueError("model must be a radial quantity")

    values = tdi.radial_sample(field, z)

    wall_val = values[:, -1]
    max_val = np.max(values, 1)

    # Rescale
    values -= wall_val[:, np.newaxis]
    values /= max_val[:, np.newaxis]

    r_norm = tdi.r_norm

    params_list = []
    rel_err_list = []

    for val in values:

        # Error norm should be weighted by the radius for cross-sectional
        # integral comparison
        # `curve_fit` squares residuals divided by sigma, so use 1/sqrt(r)
        # to weight squared residuals by r. Infinite sigma excludes the axis.
        with np.errstate(divide='ignore'):
            weights = 1.0/np.sqrt(r_norm)

        cf_result = curve_fit(model, r_norm, val, sigma=weights, **kwargs)

        model_eval = model(r_norm, *cf_result[0])

        params_list.append(cf_result[0])
        rel_err_list.append(relative_error(r_norm, val, model_eval))

    return values, np.array(params_list), np.array(rel_err_list), max_val


def fit_quantity(tdi: TwoDInterface, field: str, z: Real[Array, " _"],
                 model: Model, **kwargs) -> tuple[Real[Array, " _"],
                                                  Real[Array, " _"],
                                                  Real[Array, " _"]]:
    """Samples the quantity radially and curve fits the parameterized
    model at the axial positions.

    args:
        tdi: Interface to 2-D dataset
        field: Field name (to fit against)
        z: Axial positions [m]
        model: Model to curve fit
        **kwargs: Passed through to SciPy `curve_fit`

    returns:
        the radially sampled quantity,
        the optimal parameters at the axial positions,
        the relative errors
    """

    if model.profile:
        raise ValueError("model must be a radial quantity")

    values = tdi.radial_sample(field, z)

    r_norm = tdi.r_norm

    params_list = []
    rel_err_list = []

    for val in values:

        # Error norm should be weighted by the radius for cross-sectional
        # integral comparison
        # `curve_fit` squares residuals divided by sigma, so use 1/sqrt(r)
        # to weight squared residuals by r. Infinite sigma excludes the axis.
        with np.errstate(divide='ignore'):
            weights = 1.0/np.sqrt(r_norm)

        cf_result = curve_fit(model, r_norm, val, sigma=weights, **kwargs)

        model_eval = model(r_norm, *cf_result[0])

        params_list.append(cf_result[0])
        rel_err_list.append(relative_error(r_norm, val, model_eval))

    return values, np.array(params_list), np.array(rel_err_list)


def save_axial(z: Real[Array, " _"], quantities: Real[Array, " _ _"],
               names: list[str], fname: str,
               descs: list[str | None] | None = None,
               units: list[str | None] | None = None) -> None:
    """Saves quantities along the axial coordinate in HDF5 format.

    args:
        z: Axial coordinates [m]
        quantities: Quantities to save along axial coordinate. Shape should be
                    (n_qty, n_z)
        names: Quantity names
        fname: Output file name
        descs: Quantity descriptions, optional. Default is None. If passed,
               must be a list of strings and/or None.
        units: Quantity units, optional. Default is None. If passed, must be a
               list of strings and/or None.
    """

    if z.ndim != 1:
        raise TypeError("z must be 1-D")

    n_z = z.size

    if quantities.ndim != 2:
        raise TypeError("quantities must be 2-D")

    if quantities.shape[1] != n_z:
        raise ValueError("quantities must be equal to length of z in second"
                         + " dimension")

    n_qty = quantities.shape[0]

    # Replace None with list of None
    descs = [None]*n_qty if descs is None else descs
    units = [None]*n_qty if units is None else units

    for label, metadata in (('names', names), ('descs', descs),
                            ('units', units)):
        if len(metadata) != n_qty:
            raise ValueError(f"{label} must have {n_qty} entries, "
                             f"got {len(metadata)}")

    with h5py.File(fname, 'w') as f:
        z_pos = f.create_dataset('axial position', data=z)
        z_pos.attrs['units'] = 'm'

        for qty, name, desc, unit in zip(quantities, names, descs, units):

            if not isinstance(name, str):
                raise TypeError("names must be strings")

            d_set = f.create_dataset(name, data=qty)

            if desc is not None:

                if not isinstance(desc, str):
                    raise TypeError("descs must be strings")

                d_set.attrs['description'] = desc

            if unit is not None:

                if not isinstance(unit, str):
                    raise TypeError("units must be strings")

                d_set.attrs['units'] = unit
