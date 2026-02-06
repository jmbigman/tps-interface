"""Analysis of 2-D cold flow data."""

from os.path import join

from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np

import pyvista as pv

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

from tps_interface import TwoDInterface, time_statistics, TORCH_LENGTH, \
    step_finder, N_POINTS, plot_radius, plot_cs_integral, plot_profiles, \
    IMAGES_FOLDER, fit_profile, sample_profile
from tps_interface.model_profiles import angular, axial, save_parameters, \
    save_cs_integrals


def _pre_process(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Calculates angular momentum, separates velocity components into scalars.

    args:
        mesh: 2-D cold flow data

    returns:
        The update data
    """

    mesh.point_data['ang_m'] = -mesh.points[:, 0]*mesh.point_data['swirl']

    mesh.point_data['vel_r'] = mesh.point_data['velocity'][:, 0]
    mesh.point_data['vel_z'] = mesh.point_data['velocity'][:, 1]

    return mesh


def _plot_parameters(z: np.ndarray, ang_params: np.ndarray,
                     ax_params: np.ndarray) -> None:
    """Plots the momentum parameters along the axial coordinate.

    args:
        z: Axial positions [m]
        ang_params: Angular momentum model profile parameters
        ax_params: Axial momentum model profile parameters
    """

    plt.figure(figsize=(5, 3))

    plt.semilogy(100*z, ang_params, label='$a$')
    plt.semilogy(100*z, ax_params[:, 0], label='$b$')
    plt.semilogy(100*z, ax_params[:, 1], label='$c$')

    plt.xlabel(r'$z \,[\mathrm{cm}]$')

    plt.legend()

    plt.grid()

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig(join(IMAGES_FOLDER, 'momentum_params.pdf'))
    plt.close()


def _plot_relative_error(z: np.ndarray, ang_rel: np.ndarray,
                         ax_rel: np.ndarray) -> None:
    """Plots the momentum relative errors along the axial coordinate.

    args:
        z: Axial positions [m]
        ang_rel: Angular momentum model profile relative error
        ax_rel: Axial momentum model profile relative error
    """

    plt.figure(figsize=(5, 3))

    plt.semilogy(100*z, ang_rel, label=r'$r u_\theta$')
    plt.semilogy(100*z, ax_rel, label='$u_z$')

    plt.xlabel(r'$z \,[\mathrm{cm}]$')
    plt.ylabel('Relative Error')

    plt.ylim(1e-2, 1e0)

    plt.legend()

    plt.grid()

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig(join(IMAGES_FOLDER, 'momentum_error.pdf'))
    plt.close()


def _plot_profile_grid(z_list: list[float], tdi: TwoDInterface) -> None:
    """Plots the angular and axial momenta and their optimally fitted profiles
    at four relevant axial positions.

    args:
        z_list: List of relevant axial positions [m]
        tdi: Interface to 2-D data
    """

    fig, axs = plt.subplots(2, 4, sharex=True, figsize=(6, 3))

    for ax in axs.flat:
        ax.set_box_aspect(1)

    axs[0, 0].set_xticks([0.0, 0.5, 1.0], [r'$0$', r'$0.5$', r'$1$'])

    r_hat = np.linspace(0.0, 1.0, N_POINTS)

    ang_profs, ang_params, _, _ = fit_profile(tdi, 'ang_m_avg',
                                              np.array(z_list), model=angular)

    ax_profs, ax_params, _, _ = fit_profile(tdi, 'vel_z_avg',
                                            np.array(z_list), model=axial)

    for i in range(4):

        axs[0, i].set_title(f'${100*z_list[i]:.0f}' + r'\, \mathrm{cm}$')

        axs[0, i].plot(r_hat, ang_profs[i], ls='-', color='black')
        axs[0, i].plot(r_hat, angular(r_hat, *ang_params[i]), ls='--',
                       color='tab:red')
        axs[0, i].grid()

        axs[1, i].plot(r_hat, ax_profs[i], ls='-', color='black')
        axs[1, i].plot(r_hat, axial(r_hat, *ax_params[i]), ls='--',
                       color='tab:red')
        axs[1, i].grid()

    axs[0, 0].set_ylabel(r'$\rho u_z$')
    axs[1, 0].set_ylabel(r'$\rho l_z$')

    fig.supxlabel(r'$\hat{r}$')

    plt.tight_layout()

    fig.subplots_adjust(bottom=0.13,  left=0.1, top=0.95,
                        hspace=0.05, wspace=0.5)

    plt.savefig(join(IMAGES_FOLDER, 'profile_grid.pdf'))
    plt.close()


parser = ArgumentParser(description="Analysis of 2-D cold flow data")
parser.add_argument('-f', type=str, metavar="\b",
                    dest="filename",
                    help="Name of file with 2-D TPS data, typically .pvd")
parser.add_argument('--pre-process', action=BooleanOptionalAction,
                    metavar="\b", default=True, dest='pre_process',
                    help="Pre-process the data for time statistics")
parser.add_argument('-o', metavar="\b", default="momentum_statistics.vtu",
                    dest='output', help="Output filename for time statistics")
parser.add_argument('-t1', type=int, metavar="\b", dest="t1", default=0,
                    help="First time point to include in statistics")
parser.add_argument('-t2', type=int, metavar="\b", dest="t2", default=-1,
                    help="Last time point to include in statistics")


if __name__ == '__main__':

    args = parser.parse_args()

    if args.pre_process:
        reader = pv.PVDReader(args.filename)
        mesh = time_statistics(reader, args.t1, args.t2,
                               ['ang_m', 'vel_r', 'vel_z'], _pre_process)
        mesh.save(args.output)
    else:
        mesh = pv.read(args.filename)

    tdi = TwoDInterface(mesh)

    tdi.save_torch_radius('1d_geometry.h5', 500)

    # Example z locations
    z_list = [0.05, 0.13, 0.22, 0.32]

    _plot_profile_grid(z_list, tdi)

    _ = step_finder(tdi, 0.12, 0.13, verbose=True)

    plot_radius(tdi, z_list)

    z = np.linspace(tdi.z_min, TORCH_LENGTH, 100)

    # Angular momentum

    ang_profs, ang_params, ang_rel, ang_cs = fit_profile(tdi, 'ang_m_avg', z,
                                                         model=angular)

    plot_cs_integral(z, ang_cs, r'r u_\theta', 'ang_m_avg')

    plot_profiles(z, ang_profs, r'r u_\theta', 'ang_m_avg', angular,
                  ang_params)

    # Axial momentum

    ax_profs, ax_params, ax_rel, ax_cs = fit_profile(tdi, 'vel_z_avg', z,
                                                     model=axial)

    plot_cs_integral(z, ax_cs, 'u_z', 'vel_z_avg')

    plot_profiles(z, ax_profs, 'u_z', 'vel_z_avg', axial, ax_params)

    # Axial development of parameters and relative error

    _plot_parameters(z, ang_params, ax_params)

    _plot_relative_error(z, ang_rel, ax_rel)

    # Radial momentum

    rad_profs, rad_cs = sample_profile(tdi, 'vel_r_avg', z)

    plot_cs_integral(z, rad_cs, 'u_r', 'vel_r_avg')

    plot_profiles(z, rad_profs, 'u_r', 'vel_r_avg')

    save_cs_integrals(z, np.vstack((ax_cs, ang_cs, rad_cs)),
                      ['axial momentum',
                       'angular momentum',
                       'radial momentum'],
                       ['m^3/s', 'm^4/s', 'm^3/s'])

    save_parameters(z, np.hstack((ang_params, ax_params)).T,
                    ['angular exponential',
                     'axial exponential',
                     'axial shift'],
                    ['Angular momentum exponential parameter',
                     'Axial momentum exponential parameter',
                     'Axial momentum shift parameter'])
