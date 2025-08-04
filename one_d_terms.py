"""Evaluates the terms in the 1-D governing equations. All plots are saved to
the `images` folder."""

from os import makedirs
from os.path import join, exists

import numpy as np
import matplotlib.pyplot as plt

from two_d_interface import TwoDInterface

IMAGES_FOLDER = 'images'


def _pderiv(var: str, axial: bool = True, area: bool = False,
            neg: bool = False) -> str:
    """Generates legend labels for partial derivatives.

    args:
        var: Name of variable/quantity being differentiated
        axial: Indicator if it is an axial derivative. False means temporal
               derivative. Default is True.
        area: Indicator if it is an area integral. Adds angled brackets if so.
              Default is False.
        neg: Indicator for negative sign in front. Default is False.

    returns:
        The partial derivative
    """

    if not isinstance(var, str):
        raise TypeError('var must be a string')

    # Argument being differentiated
    if area:
        diff_arg = r'\left \langle ' + var + r'\right \rangle'
    else:
        diff_arg = var

    # Differentiation variable
    if axial:
        diff_var = 'z'
    else:
        diff_var = 't'

    if neg:
        premult = '-'
    else:
        premult = ''

    # Form inline partial derivative
    out = r'$' + premult + r'\partial_{' + diff_var + r'}' + diff_arg + r'$'

    return out


def _wall_term(var: str, deriv: bool = False, lin_avg: bool = False,
               neg: bool = False) -> str:
    """Generates legend labels for wall terms.

    args:
        var: Name of variable being evaluated.
        deriv: Indicator for torch radius derivative factor. Default is False.
        lin_avg: Indicator for linear average. Default is False.
        neg: Indicator for negative sign in front. Default is False.

    returns:
        The wall term
    """

    if not isinstance(var, str):
        raise TypeError('var must be a string')

    if neg:
        premult = '-'
    else:
        premult = ''

    # Circumference
    if deriv:
        factor = r"2 \pi R R'"
    else:
        factor = r'2 \pi R'

    if lin_avg:
        var_ = r'\overline{' + var + r'}'
    else:
        var_ = var + r'^b'

    # Form wall evaluation
    out = r'$' + premult + factor + var_ + r'$'

    return out


def _plot_profiles(tdi: TwoDInterface, field: str, z: np.ndarray, var: str,
                   n_points: int = None) -> None:
    """Plots the radial profiles of the given field and axial positions.

    args:
        tdi: Interface to 2-D dataset
        field: Field name
        z: Axial positions
        var: LaTeX for variable/quantity being plotted
        n_points: Number of sample points in radial profile
    """

    folder = join(IMAGES_FOLDER, field)

    if not exists(folder):
        makedirs(folder)

    for i, z_ in enumerate(z):

        if n_points is None:
            prof = tdi.radial_profile(field, z_, )
        else:
            prof = tdi.radial_profile(field, z_, n_points)

        radius = prof[-1, 0]

        plt.figure(figsize=(4, 3))
        plt.plot(prof[:, 0]/radius, prof[:, 1])
        plt.title(f"$z = {z_:.2f}$")

        plt.xlabel("$r$")
        plt.xlim((-0.05, 1.05))
        plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0],
                ["$0$", "$R/4$", "$R/2$", "$3R/4$", "$R$"])

        ylabel = r'$' + var + r'/\left \langle ' + var + r'\right \rangle$'
        plt.ylabel(ylabel)

        plt.grid()
        plt.tight_layout()

        plt.savefig(join(folder, str(i)+'.pdf'))
        plt.close()


def _plot_terms(terms: list[np.ndarray], z: np.ndarray, labels: list[str],
                field: str, var: str) \
        -> None:
    """Plots the terms in a 1-D governing equation (besides the time
    derivative).

    args:
        terms: List of 1-D NumPy arrays containing the terms
        z: Axial positions
        labels: List of term labels
        state_var: State variable of governing equation
        field: Field name
        var: LaTeX for state variable
    """

    folder = join(IMAGES_FOLDER, field)

    plt.figure(figsize=(8, 6))

    for term, label_ in zip(terms, labels):
        plt.plot(z, term, label=label_)

    plt.xlabel("$z$")

    plt.ylabel(_pderiv(var, False, True))

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(join(folder, "governing.pdf"))
    plt.close()


if __name__ == "__main__":


    PRMS_DICT = {"font.size": 12, "text.usetex": True, "font.family": 'serif',
                "font.serif": 'Computer Modern', "lines.linewidth": 2,
                "text.latex.preamble": r'\usepackage{amsfonts}'}

    plt.rcParams.update(PRMS_DICT)

    z_grid = np.linspace(0, 0.33, 100)

    # =============================================================================
    # Angular momentum
    # =============================================================================

    two_d = TwoDInterface("results/angular_momentum.pvtu")

    print(f"Angular momentum fields: {two_d.field_names}")

    _plot_profiles(two_d, "angular_momentum", z_grid, r"\ell_z")

    adv_flux = np.array([two_d.area_integral("advective_flux", z_) for z_ in z_grid])
    visc_flux = np.array([two_d.area_integral("viscous_flux", z_) for z_ in z_grid])
    wall_stress = np.array([two_d.wall_value("wall_stress", z_) for z_ in z_grid])
    r = np.array([two_d.torch_radius(z_) for z_ in z_grid])

    trms = [-np.gradient(adv_flux),
            np.gradient(visc_flux),
            wall_stress*2*np.pi*r**2]

    lbls = [_pderiv(r'\ell_z u_z', area=True, neg=True),
            _pderiv(r'r \tau_{\theta z}', area=True),
            r"$2 \pi R^2 \tau_{r \theta}^b$"]

    _plot_terms(trms, z_grid, lbls, "angular_momentum", r"\ell_z")

    # =============================================================================
    # Density
    # =============================================================================

    two_d = TwoDInterface("results/density.pvtu")

    print(f"Density fields: {two_d.field_names}")

    _plot_profiles(two_d, "density", z_grid, r"\rho")

    adv_flux = np.array([two_d.area_integral("advective_flux", z_) for z_ in z_grid])

    trms = [-np.gradient(adv_flux)]

    lbls = [_pderiv(r'\rho u_z', area=True, neg=True)]

    _plot_terms(trms, z_grid, lbls, "density", r'\rho')

    # =============================================================================
    # Axial momentum
    # =============================================================================

    two_d = TwoDInterface("results/axial_momentum.pvtu")

    print(f"Axial momentum fields: {two_d.field_names}")

    _plot_profiles(two_d, "axial_momentum", z_grid, r"\rho u_z")

    _plot_profiles(two_d, "pressure", z_grid, r"p")

    adv_flux = np.array([two_d.area_integral("advective_flux", z_) for z_ in z_grid])
    pressure = np.array([two_d.area_integral("pressure", z_) for z_ in z_grid])
    visc_flux = np.array([two_d.area_integral("viscous_flux", z_) for z_ in z_grid])
    wall_stress_rz = np.array([two_d.wall_value("wall_stress_rz", z_) for z_ in z_grid])
    wall_stress_zz = np.array([two_d.wall_value("wall_stress_zz", z_) for z_ in z_grid])
    wall_pressure = np.array([two_d.wall_value("pressure", z_) for z_ in z_grid])
    body_force = np.array([two_d.area_integral("body_force", z_) for z_ in z_grid])
    r = np.array([two_d.torch_radius(z_) for z_ in z_grid])
    r_d = np.gradient(r)

    trms = [-np.gradient(adv_flux),
            -np.gradient(pressure),
            np.gradient(visc_flux),
            2*np.pi*r*wall_stress_rz,
            -2*np.pi*r*r_d*wall_stress_zz,
            2*np.pi*r*r_d*wall_pressure,
            -body_force]

    lbls = [_pderiv(r'\rho u_z^2', area=True, neg=True),
            _pderiv(r'p', area=True, neg=True),
            _pderiv(r'\tau_{zz}', area=True),
            _wall_term(r'\tau_{rz}'),
            _wall_term(r'\tau_{zz}', deriv=True, neg=True),
            _wall_term(r'p', deriv=True),
            r'$\left \langle \rho \right \rangle g$']

    _plot_terms(trms, z_grid, lbls, "axial_momentum", r'\rho u_z')

    # =============================================================================
    # Radial momentum
    # =============================================================================

    two_d = TwoDInterface("results/radial_momentum.pvtu")

    print(f"Radial momentum fields: {two_d.field_names}")

    _plot_profiles(two_d, "radial_momentum", z_grid, r"\rho u_r")

    adv_flux = np.array([two_d.area_integral("advective_flux", z_) for z_ in z_grid])
    visc_flux = np.array([two_d.area_integral("viscous_flux", z_) for z_ in z_grid])
    wall_stress_rr = np.array([two_d.wall_value("wall_stress_rr", z_) for z_ in z_grid])
    wall_stress_rz = np.array([two_d.wall_value("wall_stress_rz", z_) for z_ in z_grid])
    wall_pressure = np.array([two_d.wall_value("pressure", z_) for z_ in z_grid])
    centrifugal = np.array([two_d.linear_average("centrifugal", z_) for z_ in z_grid])
    pressure = np.array([two_d.linear_average("pressure", z_) for z_ in z_grid])
    stress_tt = np.array([two_d.linear_average("stress_tt", z_) for z_ in z_grid])
    r = np.array([two_d.torch_radius(z_) for z_ in z_grid])
    r_d = np.gradient(r)

    trms = [-np.gradient(adv_flux),
            np.gradient(visc_flux),
            2*np.pi*r*wall_stress_rr,
            -2*np.pi*r*r_d*wall_stress_rz,
            2*np.pi*r*wall_pressure,
            2*np.pi*r*centrifugal,
            -2*np.pi*r*pressure,
            -2*np.pi*r*stress_tt]

    lbls = [_pderiv(r'\rho u_r u_z', area=True, neg=True),
            _pderiv(r'\tau_{rz}', area=True),
            _wall_term(r'\tau_{rr}'),
            _wall_term(r'\tau_{r z}', deriv=True, neg=True),
            _wall_term(r'p'),
            _wall_term(r'\rho u_\theta^2', lin_avg=True),
            _wall_term(r'p', lin_avg=True, neg=True),
            _wall_term(r'\tau_{\theta \theta}', lin_avg=True, neg=True)]

    _plot_terms(trms, z_grid, lbls, "radial_momentum", r'\rho u_r')
