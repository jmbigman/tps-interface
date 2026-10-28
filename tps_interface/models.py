"""Proposed parameterized radial models."""

from typing import Any

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from scipy.special import erfi

from jaxtyping import Real

Array = np.ndarray


@dataclass(frozen=True)
class Parameter:
    """Describes a model parameter and its default fitting settings.

    args:
        name: Parameter name
        description: Description for metadata and statistics
        label: Plot legend label, optionally containing LaTeX
        initial: Initial fitting value, optional. Default is 1.0.
        lower: Inclusive lower fitting bound, optional. Default is -np.inf.
        upper: Inclusive upper fitting bound, optional. Default is np.inf.
    """

    name: str
    description: str
    label: str
    initial: float = 1.0
    lower: float = -np.inf
    upper: float = np.inf


class Model(ABC):
    """A parameterized radial function for a quantity. Letting the quantity be
    q(r, z), the radial function can be a dimensionless radial profile

        f(r, z) = π R^2 q(r, z)/ ⟨q⟩(z)

    or the actual quantity q(r, z).
    """

    @property
    @abstractmethod
    def parameters(self) -> tuple[Parameter, ...]:
        """Describes parameters in the order accepted by __call__.

        returns:
            The ordered parameter specifications, excluding x.
        """
        pass

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
    def parameters(self) -> tuple[Parameter, ...]:
        """Describes the angular exponential parameter.

        returns:
            The specification for a.
        """

        prms = []

        prms.append(Parameter(name='exponential',
                              description='Specific angular momentum'
                              + ' exponential a',
                              label=r'$a_\theta$'))

        return tuple(prms)

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


class CounterFlow(Model):
    """This is a two parameter radial profile for the axial momentum

        f_z(s; n, a) = ((n + 2) / n) * (1 - s^n)
                       * (1 - a * ((n+2)/(2(n+4)) - s^2)),

    where n > 1 and a is unbounded. This is a power law and polynomial
    modification of the fully-developed profile, which it recovers for n = 2,
    a = 0. The polynomial portion is of minimal non-trivial degree (quadratic).
    """

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Describes the axial power law and polynomial parameters.

        returns:
            The specifications for n and a, with n constrained above one.
        """

        prms = []

        prms.append(Parameter(name='exponent',
                              description='Axial velocity power law'
                              + ' exponent n',
                              label=r'$n$', initial=2.0, lower=1.0 + 1e-8))

        prms.append(Parameter(name='polynomial',
                              description='Axial velocity polynomial'
                              + ' parameter a',
                              label=r'$a_z$', initial=0.0))

        return tuple(prms)

    @property
    def profile(self) -> bool:
        """Indicator the function is a dimensionless radial profile."""
        return True

    def __call__(self, x: Real[Array, " _"], n: float, a: float
                 ) -> Real[Array, " _"]:
        """Evaluates the axial momentum radial profile.

        args:
            x: Evaluation points in [0, 1] (normalized radius)
            n: Power law exponent
            a: Polynomial parameter

        returns:
            the evaluated profile
        """

        out = (n+2)/n
        out *= (1 - x**n)
        out *= 1 - a*((n+2)/(2*(n+4)) - x**2)

        return out


class Axial(Model):
    """The parameterized dimensionless radial profile for the axial momentum.
    This is a two parameter model

        f(x) = A * (exp(a*x^2) - b) * (1 - x^2)

    where the normalization coefficient, A, has an analytical expression.
    """

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Describes the axial exponential and shift parameters.

        returns:
            The specifications for a and b.
        """

        prms = []

        prms.append(Parameter(name='exponential',
                              description='Axial velocity exponential'
                              + ' parameter a',
                              label=r'$a_z$'))

        prms.append(Parameter(name='shift',
                              description='Axial velocity shift parameter b',
                              label=r'$b_z$'))

        return tuple(prms)

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
