"""Proposed radial profile models with curve fitting."""

from typing import Callable

import numpy as np

from scipy.special import erfi

# Should take point(s), then additional arguments
ModelProfile = Callable[..., float | np.ndarray]


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
