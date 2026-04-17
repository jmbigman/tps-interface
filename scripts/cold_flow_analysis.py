"""Analysis of 2-D cold flow data."""

from os.path import join

from argparse import ArgumentParser, BooleanOptionalAction

from dataclasses import dataclass

import numpy as np

import pyvista as pv

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

from tps_interface import TwoDInterface, time_statistics, TORCH_LENGTH, \
    step_finder, N_POINTS, plot_radius, plot_cs_integral, plot_profiles, \
    IMAGES_FOLDER, fit_profile, sample_profile, save_torch_radius
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


@dataclass(frozen=True)
class AngularJump:
    """Stores information on the jump in angular momentum due to the torch
    radius discontinuity.

    args:
        adv_l: Advective flux of angular momentum on the left side
        adv_r: Advective flux of angular momentum on the right side
        diff_l: Diffusive flux of angular momentum on the left side
        diff_r: Diffusive flux of angular momentum on the right side
    """

    adv_l: float
    adv_r: float
    diff_l: float
    diff_r: float

    def adv_jump(self) -> float:
        """Jump in advective flux across the discontinuity."""
        return self.adv_r - self.adv_l

    def diff_jump(self) -> float:
        """Jump in diffusive flux across the discontinuity."""
        return self.diff_r - self.diff_l

    def jump(self) -> float:
        """Jump in total flux across the discontinuity."""
        return self.adv_jump() + self.diff_jump()


def _angular_jump(z_l: float, z_r: float, tdi: TwoDInterface, nu: float
                  ) -> AngularJump:
    """Calculates the difference in the angular momentum flux across the torch
    radius discontinuity.

    args:
        z_l: Axial position left of the torch radius discontinuity [m]
        z_r: Axial position right of the torch radius discontinuity [m]
        tdi: Interface to 2-D data
        nu: Kinematic viscosity [m^2/s]

    returns:
        an `AngularJump` instance describing the angular momentum flux at the
        discontinuity
    """

    adv_flux = []
    diff_flux = []

    # Infinitesimal for first order derivative approximation
    dz = 1e-8
    sign = {z_l: -1.0, z_r: 1.0}

    for z in [z_l, z_r]:

        r = tdi.torch_radius(z)

        ang_prof, ang = tdi.radial_profile('ang_m_avg', z)
        ax_prof, ax = tdi.radial_profile('vel_z_avg', z)

        r_pts = np.linspace(0.0, 1.0, ang_prof.size)

        adv = 2*np.pi*np.trapezoid(r_pts*ang_prof*ax_prof, r_pts)
        adv *= ang
        adv *= ax
        adv /= (np.pi*r**2)**2

        adv_flux.append(adv)

        # First order derivative approximation
        ang_d = tdi.cs_integral('ang_m_avg', z + sign[z]*dz)
        diff_flux.append(-nu*sign[z]*(ang - ang_d)/dz)

    return AngularJump(*adv_flux, *diff_flux)


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

    axs[0, 0].set_ylabel(r'$\rho l_z$')
    axs[1, 0].set_ylabel(r'$\rho u_z$')

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

    z, r = save_torch_radius(tdi, '1d_geometry.h5', 500)
    plot_radius(z, r, filename='original.pdf')
    z, r = save_torch_radius(tdi, '1d_geometry_no_step.h5', 500, 0.11, 0.15,
                             mode='no_step')
    plot_radius(z, r, filename='no_step.pdf')
    z, r = save_torch_radius(tdi, '1d_geometry_smooth.h5', 500, 2.5e-3,
                             mode='smooth')
    plot_radius(z, r, filename='smooth.pdf')

    # Example z locations
    z_list = [0.05, 0.13, 0.22, 0.32]

    _plot_profile_grid(z_list, tdi)

    z_step = step_finder(tdi, 0.12, 0.13, 1e-12, verbose=True)

    ang_j = _angular_jump(z_step - 1e-12, z_step + 1e-12, tdi,
                          3.77e-5/1.62277)

    print(f"Jump in advective flux: {ang_j.adv_jump()}")
    print(f"Jump in diffusive flux: {ang_j.diff_jump()}")
    print(f"Jump in total flux: {ang_j.jump()}")

    # Mesh refined in inlet and step regions
    z = np.linspace(tdi.z_min, 0.02, 10)
    z = np.append(z, np.linspace(z.max(), 0.11, 20))
    z = np.append(z, np.linspace(z.max(), 0.14, 40))
    z = np.append(z, np.linspace(z.max(), TORCH_LENGTH, 50))
    z = np.unique(z)

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
