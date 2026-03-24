"""Plotting functions for 1-D relevant quantities from 2-D data."""

from os import makedirs
from os.path import join, exists

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Ellipse

from .two_d_interface import TwoDInterface, TORCH_LENGTH
from .model_profiles import ModelProfile

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


def plot_radius(tdi: TwoDInterface, z_points: list[float]) -> None:
    """Plots the torch radius in units of [cm].

    args:
        tdi: Interface to 2-D dataset
        z_points: Axial positions to mark with vertical lines [m]
    """

    plt.figure(figsize=(5, 2.5))

    for z_ in z_points:
        plt.axvline(100*z_, ls=':', lw=1.5, color='black')

    z_min = tdi.mesh.bounds[2]
    z_max = TORCH_LENGTH

    z = np.linspace(z_min, z_max, 500)

    r_grid = np.array([tdi.torch_radius(z_) for z_ in z])

    tdi.clear_radius_cache()

    plt.plot(100*z, 100*r_grid)

    circle = Ellipse((13, 2.625),
                     width=3.25,
                     height=0.58,
                     fill=False, edgecolor='red', lw=1.5)
    plt.gca().add_patch(circle)

    circle = Ellipse((31.5, 1.9),
                     width=1.75,
                     height=1.0,
                     fill=False, edgecolor='red', lw=1.5)
    plt.gca().add_patch(circle)

    plt.ylim((0.0, 3.0))

    plt.xlabel(r'$z \,[\mathrm{cm}]$')
    plt.ylabel(r'$R(z) \,[\mathrm{cm}]$')

    plt.grid()
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig(join(IMAGES_FOLDER, "radius.pdf"))
    plt.close()


def plot_cs_integral(z: np.ndarray, cs_integral: np.ndarray, var: str,
                     field: str) -> None:
    """Plots the axial development of the cross-sectional integral of the given
    field.

    args:
        z: Axial positions [m]
        cs_integral: Cross-sectional integral of quantity
        var: LaTeX for quantity being plotted
        field: Field name. Used to name output file.
    """

    plt.figure(figsize=(5, 3))

    plt.plot(100*z, cs_integral)

    ylbl = r'$\left \langle ' + var + r'\right \rangle$'
    plt.ylabel(ylbl)
    plt.xlabel(r'$z \,[\mathrm{cm}]$')

    plt.grid()

    bottom, top = plt.ylim()

    if bottom > 0:
        plt.ylim(0.0, top)

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig(join(IMAGES_FOLDER, field+'.pdf'))
    plt.close()


def plot_profiles(z: np.ndarray, profs: np.ndarray, var: str, field: str,
                  model: ModelProfile = None, params: np.ndarray = None) \
                    -> None:
    """Plots the radial profiles of the given field and axial positions.
    Can optionally include the optimized model profile.

    args:
        z: Axial positions [m]
        profs: Radial profiles from data
        var: LaTeX for quantity being plotted
        field: Field name. Used to name output folder.
        mode: Model profile to plot, optional.
        params: Model profile parameters, optional. Required if a model is
                passed.
    """

    folder = join(IMAGES_FOLDER, field)

    if not exists(folder):
        makedirs(folder)

    r_hat = np.linspace(0.0, 1.0, profs.shape[1])

    for i, z_ in enumerate(z):

        prof = profs[i]

        plt.figure(figsize=(4, 3))
        plt.plot(r_hat, prof, label="Data")
        plt.title(f"$z = {100*z_:.2f}" + r'\, [\mathrm{cm}]$')

        if model is not None:

            if params is None:
                raise ValueError("Parameters must be passed to evaluate the"
                                 + " model profiles.")

            model_eval = model(r_hat, *params[i])

            plt.plot(r_hat, model_eval, ls='--', label='Model')
            plt.legend()

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
