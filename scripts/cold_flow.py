"""Analyzes 2-D cold flow data."""

from os import makedirs
from os.path import basename, join

import numpy as np

import pyvista as pv

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

from tps_interface import TwoDInterface, time_statistics, TORCH_LENGTH, \
    step_finder, plot_radius, plot_axial, plot_radial, fit_profile, \
    save_torch_radius, plot_relative_error, mean_std, save_axial, \
    create_analysis_parser
from tps_interface.models import Angular, Axial

FOLDER = 'output/cold_flow'


def _pre_process(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Calculates specific angular momentum, separates velocity components
    into scalars.

    args:
        mesh: 2-D cold flow data

    returns:
        the updated data
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

    plt.savefig(join(FOLDER, 'momentum_params.pdf'))
    plt.close()


parser = create_analysis_parser("Analysis of 2-D cold flow data",
                                "momentum_statistics.vtu")

if __name__ == '__main__':

    args = parser.parse_args()

    print(f"Input file: {args.filename}")

    print(f"Output directory: {FOLDER}")

    makedirs(FOLDER, exist_ok=True)
    # ----------------------------------------------------------------------- #
    if args.pre_process:

        reader = pv.PVDReader(args.filename)

        last_index = (len(reader.time_values) - 1
                      if args.t2 is None else args.t2)

        print(f"Computing time statistics for indices {args.t1}"
              + f" through {last_index}")

        mesh = time_statistics(reader, args.t1, args.t2,
                               ['ang_m', 'vel_r', 'vel_z'], _pre_process)

        statistics_file = join(FOLDER, basename(args.output))

        mesh.save(statistics_file)

        print(f"Saved time statistics to {statistics_file}")

    else:

        print(f"Loading existing statistics from {args.filename}")

        mesh = pv.read(args.filename)

    print(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    tdi = TwoDInterface(mesh, n_points=200)
    # ----------------------------------------------------------------------- #
    print("Saving and plotting original, no-step, and smoothed torch geometry")

    z, r = save_torch_radius(tdi, '1d_geometry.h5', 2000, folder=FOLDER)
    plot_radius(z, r, 'original.pdf', folder=FOLDER)
    z, r = save_torch_radius(tdi, '1d_geometry_no_step.h5', 2000,
                             0.11, 0.15, mode='no_step', folder=FOLDER)
    plot_radius(z, r, 'no_step.pdf', folder=FOLDER)
    z, r = save_torch_radius(tdi, '1d_geometry_smooth.h5', 2000,
                             2.5e-3, mode='smooth', folder=FOLDER)
    plot_radius(z, r, 'smooth.pdf', folder=FOLDER)
    # ----------------------------------------------------------------------- #
    print("Locating the torch geometry step")

    z_step = step_finder(tdi, 0.12, 0.13, 1e-12, verbose=True)

    tdi.clear_radius_cache()

    # Mesh refined in inlet and step regions
    z = np.linspace(tdi.z_min, 0.02, 10)
    z = np.append(z, np.linspace(z.max(), 0.11, 10))
    z = np.append(z, np.linspace(z.max(), 0.14, 20))
    z = np.append(z, np.linspace(z.max(), TORCH_LENGTH, 40))
    z = np.unique(z)
    # ----------------------------------------------------------------------- #
    # Angular momentum
    # ----------------------------------------------------------------------- #
    print("Evaluating angular momentum cross-sectional integral and"
          + " parameters")

    ang_model = Angular()

    (ang_profs,
     ang_params,
     ang_rel,
     ang_cs) = fit_profile(tdi, 'ang_m_avg', z, ang_model)

    plot_axial(z, ang_cs, r'r u_\theta', 'ang_m.pdf', folder=FOLDER)

    plot_radial(z, ang_profs, r'r u_\theta', join(FOLDER, 'ang_m'), ang_model,
                ang_params)

    print("Angular fit parameter statistics")
    _ = mean_std(z, ang_params)
    # ----------------------------------------------------------------------- #
    # Axial momentum
    # ----------------------------------------------------------------------- #
    print("Evaluating axial momentum cross-sectional integral and parameters")

    axi_model = Axial()

    (axi_profs,
     axi_params,
     axi_rel,
     axi_cs) = fit_profile(tdi, 'vel_z_avg', z, axi_model)

    plot_axial(z, axi_cs, 'u_z', 'axi_m.pdf', folder=FOLDER)

    plot_radial(z, axi_profs, 'u_z', join(FOLDER, 'axi_m'), axi_model,
                axi_params)

    print("Axial exponential parameter statistics")
    _ = mean_std(z, axi_params[:, 0])

    print("Logarithm of axial shift parameter statistics")
    _ = mean_std(z, np.log(axi_params[:, 1]))

    # Axial development of parameters and relative error
    print("Plotting fitted parameters and relative errors")
    _plot_parameters(z, ang_params, axi_params)

    plot_relative_error(z, np.vstack((ang_rel, axi_rel)),
                        [r'$f_\theta$', '$f_z$'],
                        'momentum_error.pdf', FOLDER)

    # ----------------------------------------------------------------------- #
    # Radial momentum
    # ----------------------------------------------------------------------- #
    print("Computing radial velocity cross-sectional integrals")
    rad_cs = tdi.cs_integral('vel_r_avg', z)

    plot_axial(z, rad_cs, 'u_r', 'rad_m', folder=FOLDER)

    # ----------------------------------------------------------------------- #
    # Save cross-sectional integrals, parameters
    # ----------------------------------------------------------------------- #
    print(f"Saving cross-sectional integrals and parameters to {FOLDER}")
    save_axial(z, np.vstack((axi_cs, ang_cs, rad_cs)),
               ['axial momentum', 'angular momentum', 'radial momentum'],
               join(FOLDER, 'momentum.h5'),
               units=['m^3/s', 'm^4/s', 'm^3/s'])

    save_axial(z, np.hstack((ang_params, axi_params)).T,
               ['angular exponential',
                'axial exponential',
                'axial shift'],
               join(FOLDER, 'parameters.h5'),
               ['Angular momentum exponential parameter',
                'Axial momentum exponential parameter',
                'Axial momentum shift parameter'])

    print(f"Analysis complete. Results saved in {FOLDER}/")
