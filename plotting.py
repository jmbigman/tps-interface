"""Plotting functions for 1-D relevant quantities from 2-D data."""

from os import makedirs
from os.path import join, exists

import numpy as np

from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from two_d_interface import TwoDInterface
from model_profiles import ModelProfile, relative_error

IMAGES_FOLDER = 'images'

if not exists(IMAGES_FOLDER):
    makedirs(IMAGES_FOLDER)

plt.rcParams.update({"font.size": 12,
                     "text.usetex": True,
                     "font.family": 'serif',
                     "font.serif": 'Computer Modern',
                     "lines.linewidth": 2,
                     "text.latex.preamble": r'\usepackage{amsfonts}',
                     'xtick.minor.size': 0, 'xtick.minor.width': 0,
                     'ytick.minor.size': 0, 'ytick.minor.width': 0})


def plot_radius(tdi: TwoDInterface) -> None:
    """Plots the torch radius.

    args:
        tdi: Interface to 2-D dataset
    """

    z_min, z_max = tdi.mesh.bounds[2:4]

    z = np.linspace(z_min, z_max, 250)

    r_grid = [tdi.torch_radius(z_) for z_ in z]

    tdi.clear_radius_cache()

    plt.figure(figsize=(5, 3))
    plt.plot(z, r_grid)

    plt.xlabel('$z$')
    plt.ylabel('$R(z)$')

    _, y_max = plt.gca().get_ylim()
    plt.ylim((0.0, y_max))

    plt.grid()
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig(join(IMAGES_FOLDER, "radius.pdf"))
    plt.close()


def plot_cs_integral(tdi: TwoDInterface, field: str, z: np.ndarray,
                     var: str, n_points: int = None) -> None:
    """Plots the axial development of the cross-sectional integral of the given
    field.

    args:
        tdi: Interface to 2-D dataset
        field: Field name
        z: Axial positions
        var: LaTeX for variable/quantity being plotted
        n_points: Number of sample points in radial profile
    """

    plt.figure(figsize=(5, 3))

    if n_points is None:
        cs_integral = [tdi.cs_integral(field, z_) for z_ in z]
    else:
        cs_integral = [tdi.cs_integral(field, z_, n_points) for z_ in z]

    plt.plot(z, cs_integral)

    ylbl = r'$\left \langle ' + var + r'\right \rangle$'
    plt.ylabel(ylbl)
    plt.xlabel("$z$")

    plt.grid()
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig(join(IMAGES_FOLDER, field+'.pdf'))
    plt.close()


def plot_profiles(tdi: TwoDInterface, field: str, z: np.ndarray, var: str,
                  n_points: int = None, model_profile: ModelProfile = None) \
                    -> None:
    """Plots the radial profiles of the given field and axial positions.

    args:
        tdi: Interface to 2-D dataset
        field: Field name
        z: Axial positions
        var: LaTeX for variable/quantity being plotted
        n_points: Number of sample points in radial profile
        model_profile: Model profile to curve fit and plot
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

        # Normalized radial coordinate
        r_hat = prof[:, 0]/radius

        plt.figure(figsize=(4, 3))
        plt.plot(r_hat, prof[:, 1], label="Data")
        plt.title(f"$z = {z_:.4f}$")

        if model_profile is not None:

            # Error norm should be weighted by the radius for cross-sectional
            # integral
            # curve_fit takes reciprocal of weights like classical WLS
            with np.errstate(divide='ignore'):
                weights = 1.0/r_hat

            cf_result = curve_fit(model_profile, r_hat, prof[:, 1],
                                  sigma=weights, method='dogbox')

            model_eval = model_profile(r_hat, *cf_result[0])

            plt.plot(r_hat, model_eval, ls='--', label='Model')
            plt.legend()

            rel_err = relative_error(r_hat, prof[:, 1], model_eval)

            print(f"z: {z_:.3f}, rel_error: {rel_err:.3e}")

        plt.xlabel(r"$\hat{r}$")
        plt.xlim((-0.05, 1.05))
        plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0],
                   ["$0$", "$0.25$", "$0.5$", "$0.75$", "$1$"])

        y_min, y_max = plt.gca().get_ylim()
        if y_min > 0.0:
            # Change viewing window to start at 0.0 for positive profiles
            y_min = 0.0
            y_max = 1.05*y_max
            plt.ylim((y_min, y_max))

        ylabel = r'$f_{' + var + r'}$'

        plt.ylabel(ylabel, fontsize=14)

        plt.grid()
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which='minor', linestyle='-', alpha=0.3)

        plt.tight_layout()

        plt.savefig(join(folder, str(i)+'.pdf'))
        plt.close()
