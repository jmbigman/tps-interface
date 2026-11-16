"""Proposed parameterized radial models."""

from typing import Any

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from scipy.integrate import quad

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

        f(r, z) = π R^2 q(r, z) / ⟨q⟩(z)

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


class AngularRational(Model):
    """A rational dimensionless radial profile for angular momentum.

        f(x; p) = A * x^2 * (1 - x) * [1 + (3/2 + 2*p)*(1 - x)]
                  / [1 + p*(1 - x)]^2,

    where p > -1. The normalization coefficient is computed by adaptive
    quadrature.
    """

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Describes the rational parameter, constrained above minus one."""
        return (Parameter(name='rational',
                          description='Specific angular momentum rational'
                          + ' parameter p',
                          label=r'$p_\theta$', initial=0.0,
                          lower=np.nextafter(-1.0, np.inf)),)

    @property
    def profile(self) -> bool:
        """Indicator the function is a dimensionless radial profile."""
        return True

    @staticmethod
    def _shape(x: Real[Array, "..."] | float, p: float
               ) -> Real[Array, "..."] | float:
        """Evaluates the unnormalized profile for a valid parameter."""

        # Flipped coordinates
        wall = 1.0 - x
        # Equivalent to 1 + p*(1 - x), without cancellation near p = -1.
        denom = x + (1.0 + p)*wall

        # Divide the numerator by denom before multiplying, avoiding its
        # square and the potentially large factor 2*p.
        out = x**2*(wall/denom)*(2.0+(1.5*wall-1.0)/denom)

        return out

    @staticmethod
    def _coeff(p: float) -> float:
        """Computes A using the cross-sectional weight 2*x."""

        if not np.isfinite(p) or p <= -1.0:
            raise ValueError("p must be finite and greater than -1")

        # Relative tolerance also controls normalization error when the
        # integral becomes small at large positive p.
        integral, _ = quad(lambda x: 2.0*x*AngularRational._shape(x, p),
                           0.0, 1.0, epsabs=0.0, epsrel=1e-12)

        return 1.0/integral

    def __call__(self, x: Real[Array, " _"], p: float) -> Real[Array, " _"]:
        """Evaluates the angular momentum radial profile.

        args:
            x: Evaluation points in [0, 1] (normalized radius)
            p: Rational parameter, > -1

        returns:
            the evaluated profile
        """
        return self._coeff(p)*self._shape(x, p)


class AxialPowerPoly(Model):
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
