"""Proposed radial profile models with curve fitting."""

from typing import Protocol

import numpy as np

from scipy.special import erfi
from scipy.optimize import curve_fit

import h5py

from .two_d_interface import TwoDInterface, N_POINTS

###############################################################################
# General utilities
###############################################################################


class ModelProfile(Protocol):
    def __call__(self, x: np.ndarray, *params: float) -> float | np.ndarray:
        """Model radial profile.

        args:
            x: Evaluation points (normalized radius)
            *params: Parameters to control the shape

        returns:
            The model profile evaluation(s)
        """
        ...


def relative_error(r_hat: np.ndarray, data: np.ndarray, model: np.ndarray) \
     -> float:
    """Calculates the L2 relative error of the model fit. A radial factor is
    included so the comparison is over the whole cross-section. The trapezoidal
    rule is used for integration.

    args:
        r_hat: Normalized radius evaluation points
        data: Data values
        model: Model profile values

    returns:
        The L2 relative error
    """

    # Norm squared difference
    abs_diff = np.trapezoid((data - model)**2*r_hat, r_hat)
    # Norm squared of data
    scale = np.trapezoid(data**2*r_hat, r_hat)

    return np.sqrt(abs_diff/scale)


def sample_profile(tdi: TwoDInterface, field: str, z: np.ndarray,
                   n_points: int = N_POINTS, field_std: str = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Samples the quantity profiles at the axial positions.

    args:
        tdi: Interface to 2-D dataset
        field: Field name (to be sampled)
        z: Axial positions [m]
        n_points: Number of sample points in radial profile, optional.
        field_std: Name of standard deviation of the sampled field, optional.
                   If passed, the uncertainties in the cross-sectional
                   integrals are returned too. Default is None.

    returns:
        The radial profiles,
        the cross-sectional integrals,
        the uncertainties in the cross-sectional integrals if specified,
            otherwise None is returned
    """

    prof_list = []
    integral_list = []
    u_list = []

    for z_ in z:

        prof, integral = tdi.radial_profile(field, z_, n_points)

        prof_list.append(prof)
        integral_list.append(integral)

        if field_std is not None:
            uncertainty = tdi.cs_uncertainty(field_std, z_, n_points)
            u_list.append(uncertainty)

    u_list = None if field_std is None else np.array(u_list)

    return np.array(prof_list), np.array(integral_list), u_list


def fit_profile(tdi: TwoDInterface, field: str, z: np.ndarray,
                n_points: int = N_POINTS, model: ModelProfile = None,
                field_std: str = None) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray | None]:
    """Samples the quantity profiles at the axial positions and fits the
    model profile parameters.

    args:
        tdi: Interface to 2-D dataset
        field: Field name (to be fit against)
        z: Axial positions [m]
        n_points: Number of sample points in radial profile
        model: Model profile to curve fit
        field_std: Name of standard deviation of the sampled field, optional.
                If passed, the uncertainties in the cross-sectional integrals
                are returned too. Default is None.

    returns:
        the radial profiles,
        the optimal parameters at the axial positions,
        the uncertainty of the parameters at the axial positions,
        the relative errors,
        the cross-sectional integrals,
        the uncertainties in the cross-sectional integrals if specified,
            otherwise None is returned
    """

    (profs,
     integrals,
     integrals_u) = sample_profile(tdi, field, z, n_points, field_std)

    r_hat = np.linspace(0.0, 1.0, n_points)

    params_list = []
    params_u_list = []
    rel_err_list = []

    for prof in profs:

        # Error norm should be weighted by the radius for cross-sectional
        # integral comparison
        # `curve_fit` takes reciprocal of weights like classical WLS
        with np.errstate(divide='ignore'):
            weights = 1.0/r_hat

        cf_result = curve_fit(model, r_hat, prof, sigma=weights,
                              method='dogbox')

        model_eval = model(r_hat, *cf_result[0])

        params_list.append(cf_result[0])
        params_u_list.append(np.sqrt(np.diag(cf_result[1])))
        rel_err_list.append(relative_error(r_hat, prof, model_eval))

    return profs, np.array(params_list), np.array(params_u_list), \
        np.array(rel_err_list), integrals, integrals_u


def save_parameters(z: np.ndarray, params: np.ndarray, names: list[str],
                    descs: list[str], fname: str = 'params.h5') -> None:
    """Saves the parameters in HDF5 format.

    args:
        z: Axial coordinates
        params: Parameters to save. Must have shape (# params, # z pos.)
        names: Parameter names
        descs: Parameter descriptions
        fname: Output file name, optional. Default is 'params.h5'
    """

    with h5py.File(fname, 'w') as f:
        z_pos = f.create_dataset('axial position', data=z)
        z_pos.attrs['units'] = 'm'

        for param, name, desc in zip(params, names, descs):
            d_set = f.create_dataset(name, data=param)
            d_set.attrs['desc'] = desc


def save_cs_integrals(z: np.ndarray, cs_integrals: np.ndarray,
                      names: list[str], units: list[str],
                      fname: str = 'cs_integral.h5') -> None:
    """Saves the cross-sectional integrals in HDF5 format.

    args:
        z: Axial coordinates
        cs_integrals: Cross-sectional integrals to save. Must have shape
                      (# quantities, # z pos.)
        names: Quantity names
        units: Quantity units
        fname: Output file name, optional. Default is 'cs_integral.h5'
    """

    with h5py.File(fname, 'w') as f:
        z_pos = f.create_dataset('axial position', data=z)
        z_pos.attrs['units'] = 'm'

        for cs_integral, name, u_ in zip(cs_integrals, names, units):
            d_set = f.create_dataset(name, data=cs_integral)
            d_set.attrs['units'] = u_


###############################################################################
# Specific profiles
###############################################################################


def _angular_coeff(a: float) -> float:
    """Normalization coefficient for the angular momentum radial profile.

    args:
        a: Parameter

    returns:
        The normalization coefficient
    """

    term1 = np.exp(a) + 2
    term1 /= 2*a**2

    term2 = 3*np.sqrt(np.pi)*erfi(np.sqrt(a))
    term2 /= 4*a**2.5

    recip = term1 - term2

    return 1.0/recip


def angular(x: float | np.ndarray, a: float) -> float | np.ndarray:
    """Angular momentum radial profile.

        f(x) = A * exp(a*x^2) * x^2 * (1 - x)

    The normalization coefficient, A, has an analytical expression in a.

    args:
        x: Evaluation points (normalized radius)
        a: Parameter

    returns:
        The evaluated profile
    """

    x_sq = x**2

    model_eval = x_sq*(1.0 - x)*np.exp(a*x_sq)

    try:
        with np.errstate(divide='raise', invalid='raise'):
            coeff = _angular_coeff(a)
    except FloatingPointError:
        recip_coeff = 2*np.trapezoid(model_eval*x, x)
        coeff = 1.0/recip_coeff

    return coeff*model_eval


def _axial_coeff(a: float, b: float) -> float:
    """Normalization coefficient for the axial momentum radial profile.

    args:
        a: Exponential parameter
        b: Shift parameter

    returns:
        The normalization coefficient
    """

    coeff = -2*a**2
    coeff /= a**2*b + 2*a - 2*np.exp(a) + 2

    return coeff


def axial(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """Axial momentum radial profile.

        f(x) = A * (exp(a*x^2) - b) * (1 - x^2)

    The normalization coefficient, A, has an analytical expression in a, b.

    args:
        x: Evaluation points (normalized radius)
        a: Exponential parameter
        b: Shift parameter

    returns:
        The evaluated profile
    """

    x_sq = x**2

    model_eval = (1.0 - x_sq)*(np.exp(a*x_sq) - b)

    try:
        with np.errstate(divide='raise', invalid='raise'):
            coeff = _axial_coeff(a, b)
    except FloatingPointError:
        recip_coeff = 2*np.trapezoid(model_eval*x, x)
        coeff = 1.0/recip_coeff

    return coeff*model_eval
