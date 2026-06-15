"""Plotting functions for 1-D relevant quantities from 2-D data."""

from os import makedirs
from os.path import join, exists, splitext

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

from jaxtyping import Real

from .two_d_interface import Model

Array = np.ndarray


plt.rcParams.update({"font.size": 12,
                     "text.usetex": True,
                     "font.family": 'serif',
                     "font.serif": 'Computer Modern',
                     "lines.linewidth": 2,
                     "text.latex.preamble": r'\usepackage{amsfonts}',
                     'xtick.minor.size': 0, 'xtick.minor.width': 0,
                     'ytick.minor.size': 0, 'ytick.minor.width': 0})


def plot_radius(z: Real[Array, " _"], r: Real[Array, " _"],
                fname: str = 'radius.pdf', folder: str = 'output') -> None:
    """Plots the torch radius in units of [cm].

    args:
        z: Axial positions [m]
        r: Torch radius at the axial positions [m]
        fname: Output file name, optional. Default is 'radius.pdf'.
        folder: Output directory, optional. Default is 'output'.
    """

    if not exists(folder):
        makedirs(folder)

    plt.figure(figsize=(5, 2.5))

    plt.plot(100*z, 100*r)

    plt.ylim((0.0, 3.0))

    plt.xlabel(r'$z \,[\mathrm{cm}]$')
    plt.ylabel(r'$R(z) \,[\mathrm{cm}]$')

    plt.grid()
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    head, ext = splitext(fname)

    if ext == '':
        fname = head + '.pdf'

    plt.savefig(join(folder, fname))
    plt.close()


def plot_axial(z: Real[Array, " _"], f: Real[Array, " _"], var: str,
               fname: str, cs: bool = True, folder: str = 'output') -> None:
    """Plots the axial development of a quantity, typically the cross-sectional
    integral of a conserved quantity.

    args:
        z: Axial positions [m]
        f: Axial quantity to plot
        var: LaTeX for quantity being plotted, without surrounding `$`s. Used
             in the y-axis label.
        fname: Output file name
        cs: Indicator quantity is a cross-sectional to automatically add angle
            brackets, optional. Default is True.
        folder: Output directory, optional. Default is 'output'.
    """

    if not exists(folder):
        makedirs(folder)

    plt.figure(figsize=(5, 3))

    if cs:
        ylbl = r'$\left \langle ' + var + r'\right \rangle$'
    else:
        ylbl = '$' + var + '$'

    plt.ylabel(ylbl)
    plt.xlabel(r'$z \,[\mathrm{cm}]$')

    plt.grid()

    # Automatically use log-spaced y if the data span many orders of magnitude
    if (np.max(f) - np.min(f))/np.mean(f) > 100:
        plt.semilogy(100*z, f)
    else:
        plt.plot(100*z, f)

        bottom, top = plt.ylim()

        if bottom > 0:
            plt.ylim(0.0, top)

        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    head, ext = splitext(fname)

    if ext == '':
        fname = head + '.pdf'

    plt.savefig(join(folder, fname))
    plt.close()


def plot_radial(z: Real[Array, " n_z"], profs: Real[Array, " n_z n_r"],
                var: str, folder: str = 'output/radial', model: Model = None,
                params: Real[Array, " n_z _"] = None,
                dimless: bool = True) -> None:
    """Plots the radial variation of a quantity, typically a dimensionless
    radial profile. Can optionally include an optimized parameterized model.

    args:
        z: Axial positions [m]
        profs: Radial quantities to plot
        var: LaTeX for quantity being plotted, without surrounding `$`s. Used
             in the y-axis label.
        folder: Output directory, optional. Default is 'output/radial'.
        mode: Model function to plot, optional.
        params: Model function parameters. Required if a model is passed.
        dimless: Indicator quantity is a dimensionless radial profile to
                 automatically subscript `f`, optional. Default is True.
    """

    if not exists(folder):
        makedirs(folder)

    r_norm = np.linspace(0.0, 1.0, profs.shape[1])

    for i, z_ in enumerate(z):

        prof = profs[i]

        plt.figure(figsize=(4, 3))
        plt.plot(r_norm, prof, label="Data")
        plt.title(f"$z = {100*z_:.2f}" + r'\, [\mathrm{cm}]$')

        if model is not None:
            if params is None:
                raise ValueError("Parameters must be passed to evaluate the"
                                 + " model profiles.")

            model_eval = model(r_norm, *params[i])

            plt.plot(r_norm, model_eval, ls='--', label='Model')
            plt.legend()

        plt.xlabel(r"$\tau$")
        plt.xlim((-0.05, 1.05))
        plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0],
                   ["$0$", "$0.25$", "$0.5$", "$0.75$", "$1$"])

        y_min, y_max = plt.ylim()
        if y_min > 0.0:
            # Change viewing window to start at 0.0 for positive profiles
            y_min = 0.0
            y_max = 1.05*y_max
            plt.ylim((y_min, y_max))

        if dimless:
            ylbl = r'$f_{' + var + r'}$'
        else:
            ylbl = '$' + var + '$'

        plt.ylabel(ylbl, fontsize=14)

        plt.grid()
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which='minor', linestyle='-', alpha=0.3)

        plt.tight_layout()

        plt.savefig(join(folder, str(i)+'.pdf'))
        plt.close()


def plot_relative_error(z: np.ndarray, rel: np.ndarray,
                        var_list: list[str], fname: str = 'error.pdf',
                        folder: str = 'output') -> None:
    """Plots relative errors along the axial coordinate.

    args:
        z: Axial positions [m]
        rel: Relative errors
        var_list: LaTeX for quantities whose relative errors are being plotted
        fname: Output file name, optional. Default is 'error.pdf'.
        folder: Output directory, optional. Default is 'output'.
    """

    if not exists(folder):
        makedirs(folder)

    plt.figure(figsize=(5, 3))

    for rel_, var in zip(rel, var_list):
        plt.semilogy(100*z, rel_, label=var)

    plt.xlabel(r'$z \,[\mathrm{cm}]$')
    plt.ylabel('Relative Error')

    y_min, _ = plt.ylim()
    plt.ylim(min(y_min, 1e-2), 1.05)

    plt.legend()

    plt.grid()

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    head, ext = splitext(fname)

    if ext == '':
        fname = head + '.pdf'

    plt.savefig(join(folder, fname))
    plt.close()
