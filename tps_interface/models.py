"""Proposed parameterized radial models."""

from typing import Any

from abc import ABC, abstractmethod

import numpy as np

from scipy.special import erfi

from jaxtyping import Real

from ._utils import TEMP_W

Array = np.ndarray


class Model(ABC):
    """A parameterized radial function for a quantity. Letting the quantity be
    q(r, z), the radial function can be a dimensionless radial profile

        f(r, z) = π R^2 q(r, z)/ ⟨q⟩(z)

    or the actual quantity q(r, z).
    """

    @property
    @abstractmethod
    def profile(self) -> bool:
        """Indicator the function is a dimensionless radial profile."""
        pass

    @abstractmethod
    def __call__(self, x: Real[Array, "..."], *params: Any):
        """Evaluate the radial function.

        args:
            x: Evaluation points in [0, 1] (normalized radius)
            *params: Parameters to control the shape

        returns:
            the model function evaluations
        """
        pass


# --------------------------------------------------------------------------- #
# Momentum
#
# These are dimensionless radial profiles
# --------------------------------------------------------------------------- #


class Angular(Model):
    """The parameterized dimensionless radial profile for the angular momentum.
    This is a one parameter model

        f(x) = A * exp(a*x^2) * x^2 * (1 - x)

    where the normalization coefficient, A, has an analytical expression.
    """

    @property
    def profile(self) -> bool:
        """Indicator the function is a dimensionless radial profile."""
        return True

    @staticmethod
    def _coeff(a: float) -> float:
        """Normalization coefficient.

        args:
            a: Parameter

        returns:
            the normalization coefficient
        """

        term1 = np.exp(a) + 2
        term1 /= 2*a**2

        term2 = 3*np.sqrt(np.pi)*erfi(np.sqrt(a))
        term2 /= 4*a**2.5

        recip = term1 - term2

        return 1.0/recip

    def __call__(self, x: Real[Array, " _"], a: float) -> Real[Array, " _"]:
        """Evaluates the angular momentum radial profile.

        args:
            x: Evaluation points in [0, 1] (normalized radius)
            a: Parameter

        returns:
            the evaluated profile
        """

        x_sq = x**2

        model_eval = x_sq*(1.0 - x)*np.exp(a*x_sq)

        try:
            with np.errstate(divide='raise', invalid='raise'):
                coeff = self._coeff(a)
        except FloatingPointError:
            recip_coeff = 2*np.trapezoid(model_eval*x, x)
            coeff = 1.0/recip_coeff

        return coeff*model_eval


class Axial(Model):
    """The parameterized dimensionless radial profile for the axial momentum.
    This is a two parameter model

        f(x) = A * (exp(a*x^2) - b) * (1 - x^2)

    where the normalization coefficient, A, has an analytical expression.
    """

    @property
    def profile(self) -> bool:
        """Indicator the function is a dimensionless radial profile."""
        return True

    @staticmethod
    def _coeff(a: float, b: float) -> float:
        """Normalization coefficient.

        args:
            a: Exponential parameter
            b: Shift parameter

        returns:
            the normalization coefficient
        """

        coeff = -2*a**2
        coeff /= a**2*b + 2*a - 2*np.exp(a) + 2

        return coeff

    def __call__(self, x: Real[Array, " _"], a: float, b: float
                 ) -> Real[Array, " _"]:
        """Evaluates the axial momentum radial profile.

        args:
            x: Evaluation points in [0, 1] (normalized radius)
            a: Exponential parameter
            b: Shift parameter

        returns:
            the evaluated profile
        """

        x_sq = x**2

        model_eval = (1.0 - x_sq)*(np.exp(a*x_sq) - b)

        try:
            with np.errstate(divide='raise', invalid='raise'):
                coeff = self._coeff(a, b)
        except FloatingPointError:
            recip_coeff = 2*np.trapezoid(model_eval*x, x)
            coeff = 1.0/recip_coeff

        return coeff*model_eval


# --------------------------------------------------------------------------- #
# Temperature
#
# These represent the actual temperature, not the dimensionless profile
# --------------------------------------------------------------------------- #


class TempSupEll(Model):
    """Super ellipse temperature radial function. This is a two parameter model

        T(x) = (a - T_w) sqrt(1 - x^b) + T_w

    where T_w is the wall temperature, a is the maximum temperature, and b >= 2
    determines the shape. b = 2 corresponds to a circle and as b -> infinity,
    the profile approaches a square.
    """

    @property
    def profile(self):
        """Indicator the function is a dimensionless radial profile."""
        return False

    def __call__(self, x: Real[Array, " _"], a: float, b: float
                 ) -> Real[Array, " _"]:
        """Evaluates the temperature radial function.

        args:
            x: Evaluation points in [0, 1] (normalized radius)
            a: Maximum temperature
            b: Shape parameter, b > 1

        returns:
            the evaluated profile
        """

        if b < 2:
            raise ValueError("shape parameter (b) must be greater than or"
                             + " equal to 2")

        return (a - TEMP_W)*np.sqrt(1 - x**b) + TEMP_W


class TempCubic(Model):
    """Cubic polynomial temperature radial function. This is a two parameter
    model

        T(x) = (a - T_w) [3b (x^2 - 1) - 2 (x^3 - 1)]
               / (b^3 - 3b + 2) + T_w

    where T_w is the wall temperature, a is the maximum temperature, and
    b < 2/3 is the location of the peak.
    """

    @property
    def profile(self):
        """Indicator the function is a dimensionless radial profile."""
        return False

    def __call__(self, x: Real[Array, " _"], a: float, b: float
                 ) -> Real[Array, " _"]:
        """Evaluates the temperature radial function.

        args:
            x: Evaluation points in [0, 1] (normalized radius)
            a: Maximum temperature
            b: Location of peak, 0 <= b < 2/3

        returns:
            the evaluated profile
        """

        if b < 0 or b >= 2/3:
            raise ValueError("peak location (b) must be in [0, 2/3)")

        out = 3*b*(x**2 - 1) - 2*(x**3 - 1)
        out /= b**3 - 3*b + 2
        out *= a - TEMP_W

        out += TEMP_W

        return out
