"""Analysis of 2-D cold flow data."""

from os.path import join

from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np

import pyvista as pv

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

from tps_interface import TwoDInterface, time_statistics, TORCH_LENGTH, \
    step_finder, plot_radius, plot_cs_integral, plot_profiles, \
    IMAGES_FOLDER, fit_profile, sample_profile, save_torch_radius
from tps_interface.model_profiles import angular, axial, save_parameters, \
    save_cs_integrals

# (TODO):
# Plot cross-sectional integrals with error, need to propagate the error


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


def _mean_std(z: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the mean and standard deviation of the quantity.

    args:
        z: Axial positions [m]
        f: Evaluated quantity at the axial positions

    returns:
        the mean, the standard deviation
    """

    length = z[-1] - z[0]

    f = f.flatten()

    mean = np.trapezoid(f, z)/length

    std = np.sqrt(np.trapezoid((f - mean)**2, z)/length)

    return mean, std


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

    print("Saving and plotting torch radius")

    z, r = save_torch_radius(tdi, '1d_geometry.h5', 2000)
    plot_radius(z, r, filename='original.pdf')
    z, r = save_torch_radius(tdi, '1d_geometry_no_step.h5', 2000, 0.11, 0.15,
                             mode='no_step')
    plot_radius(z, r, filename='no_step.pdf')
    z, r = save_torch_radius(tdi, '1d_geometry_smooth.h5', 2000, 2.5e-3,
                             mode='smooth')
    plot_radius(z, r, filename='smooth.pdf')

    z_step = step_finder(tdi, 0.12, 0.13, 1e-12, verbose=True)

    # Mesh refined in inlet and step regions
    z = np.linspace(tdi.z_min, 0.02, 10)
    z = np.append(z, np.linspace(z.max(), 0.11, 10))
    z = np.append(z, np.linspace(z.max(), 0.14, 20))
    z = np.append(z, np.linspace(z.max(), TORCH_LENGTH, 40))
    z = np.unique(z)

    # Angular momentum
    print("Evaluating angular momentum cross-sectional integral and"
          + " parameters")

    (ang_profs,
     ang_params,
     ang_params_u,
     ang_rel,
     ang_cs,
     ang_cs_u) = fit_profile(tdi, 'ang_m_avg', z, model=angular,
                             field_std='ang_m_std')

    sample_profile(tdi, "ang_m_std", z)

    plot_cs_integral(z, ang_cs, r'r u_\theta', 'ang_m_avg', ang_cs_u)

    plot_profiles(z, ang_profs, r'r u_\theta', 'ang_m_avg', angular,
                  ang_params)

    print("Angular parameter a")
    print(f"\tMinimum: {ang_params.min()}")
    print(f"\tMaximum: {ang_params.max()}")

    mean, std = _mean_std(z, ang_params)

    print(f"\tMean: {mean}")
    print(f"\tStandard deviation: {std}")

    # Axial momentum
    print("Evaluating axial momentum cross-sectional integral and parameters")

    (ax_profs,
     ax_params,
     ax_params_u,
     ax_rel,
     ax_cs,
     ax_cs_u) = fit_profile(tdi, 'vel_z_avg', z, model=axial,
                            field_std='vel_z_std')

    plot_cs_integral(z, ax_cs, 'u_z', 'vel_z_avg', ax_cs_u)

    plot_profiles(z, ax_profs, 'u_z', 'vel_z_avg', axial, ax_params)

    print("Axial parameter b")
    print(f"\tMinimum: {ax_params[:, 0].min()}")
    print(f"\tMaximum: {ax_params[:, 0].max()}")

    mean, std = _mean_std(z, ax_params[:, 0])

    print(f"\tMean: {mean}")
    print(f"\tStandard deviation: {std}")

    print("Axial parameter log(c)")
    log_param = np.log(ax_params[:, 1])

    print(f"\tMinimum: {log_param.min()}")
    print(f"\tMaximum: {log_param.max()}")

    mean, std = _mean_std(z, log_param)

    print(f"\tMean: {mean}")
    print(f"\tStandard deviation: {std}")

    # Axial development of parameters and relative error

    _plot_parameters(z, ang_params, ax_params)

    _plot_relative_error(z, ang_rel, ax_rel)

    # Radial momentum

    rad_profs, rad_cs, rad_cs_u = sample_profile(tdi, 'vel_r_avg', z,
                                                 field_std='vel_r_std')

    plot_cs_integral(z, rad_cs, 'u_r', 'vel_r_avg', rad_cs_u)

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
