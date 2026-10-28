"""Analyzes 2-D cold flow data."""

from os import makedirs
from os.path import basename, isfile, join

import numpy as np

import pyvista as pv

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from tps_interface import TwoDInterface, time_statistics, TORCH_LENGTH, \
    step_finder, plot_radius, plot_axial, plot_radial, fit_profile, \
    save_torch_radius, plot_relative_error, mean_std, save_axial, \
    create_analysis_parser, load_torch_radius
from tps_interface.models import Angular, CounterFlow, Parameter

FOLDER = 'output/cold_flow'


def _pre_process(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Calculates specific angular momentum, separates velocity components
    into scalars.

    args:
        mesh: 2-D cold flow data

    returns:
        the updated data
    """

    mesh.point_data['ang_sm'] = -mesh.points[:, 0]*mesh.point_data['swirl']

    mesh.point_data['vel_r'] = mesh.point_data['velocity'][:, 0]
    mesh.point_data['vel_z'] = mesh.point_data['velocity'][:, 1]

    return mesh


def _plot_parameters(z: np.ndarray, params: np.ndarray,
                     parameters: tuple[Parameter, ...]) -> None:
    """Plots the momentum parameters along the axial coordinate.

    args:
        z: Axial positions [m]
        params: Fitted parameters with shape (n_z, n_params)
        parameters: Ordered specifications for the fitted parameters
    """

    plt.figure(figsize=(5, 3))

    for parameter, values in zip(parameters, params.T, strict=True):
        plt.plot(100*z, values, label=parameter.label)

    plt.xlabel(r'$z \,[\mathrm{cm}]$')

    plt.legend()

    plt.grid()

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which='minor', linestyle='-', alpha=0.3)

    plt.tight_layout()

    plt.savefig(join(FOLDER, 'momentum_params.pdf'))
    plt.close()


parser = create_analysis_parser("Analysis of 2-D cold flow data",
                                "statistics.vtu")

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
                               ['ang_sm', 'vel_r', 'vel_z'], _pre_process)

        statistics_file = join(FOLDER, basename(args.output))

        mesh.save(statistics_file)

        print(f"Saved time statistics to {statistics_file}")

    else:

        print(f"Loading existing statistics from {args.filename}")

        mesh = pv.read(args.filename)

    print(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    tdi = TwoDInterface(mesh, n_points=200)
    # ----------------------------------------------------------------------- #
    print("Loading or saving and plotting torch geometry")

    if args.overwrite_radius or not isfile(join(FOLDER, '1d_geometry.h5')):
        z, r = save_torch_radius(tdi, '1d_geometry.h5', 2000, folder=FOLDER)
    else:
        z, r = load_torch_radius('1d_geometry.h5', folder=FOLDER)

    plot_radius(z, r, 'original.pdf', folder=FOLDER)

    if (args.overwrite_radius
            or not isfile(join(FOLDER, '1d_geometry_no_step.h5'))):
        z, r = save_torch_radius(tdi, '1d_geometry_no_step.h5', 2000,
                                 0.11, 0.15, mode='no_step', folder=FOLDER)
    else:
        z, r = load_torch_radius('1d_geometry_no_step.h5', folder=FOLDER)

    plot_radius(z, r, 'no_step.pdf', folder=FOLDER)

    if (args.overwrite_radius
            or not isfile(join(FOLDER, '1d_geometry_smooth.h5'))):
        z, r = save_torch_radius(tdi, '1d_geometry_smooth.h5', 2000,
                                 2.5e-3, mode='smooth', folder=FOLDER)
    else:
        z, r = load_torch_radius('1d_geometry_smooth.h5', folder=FOLDER)

    plot_radius(z, r, 'smooth.pdf', folder=FOLDER)
    # ----------------------------------------------------------------------- #
    print("Locating the torch geometry step")

    z_step = step_finder(tdi, 0.12, 0.13, 1e-12, verbose=True)

    tdi.clear_radius_cache()
    # ----------------------------------------------------------------------- #
    # Mesh refined in inlet and step regions
    z = np.linspace(tdi.z_min, 0.02, 10)
    z = np.append(z, np.linspace(z.max(), 0.11, 10))
    z = np.append(z, np.linspace(z.max(), 0.14, 20))
    z = np.append(z, np.linspace(z.max(), TORCH_LENGTH, 40))
    z = np.unique(z)
    # ----------------------------------------------------------------------- #
    # Angular momentum
    # ----------------------------------------------------------------------- #
    print("Evaluating specific angular momentum cross-sectional integral and"
          + " parameters")

    ang_model = Angular()
    ang_specs = ang_model.parameters

    (ang_profs,
     ang_params,
     ang_rel,
     ang_cs) = fit_profile(tdi, 'ang_sm_avg', z, ang_model,
                           p0=tuple(p.initial for p in ang_specs),
                           bounds=(tuple(p.lower for p in ang_specs),
                                   tuple(p.upper for p in ang_specs)))

    plot_axial(z, ang_cs, r'r u_\theta', 'ang_sm.pdf', folder=FOLDER)

    plot_radial(z, ang_profs, r'\theta', join(FOLDER, 'ang_sm'), ang_model,
                ang_params)

    for parameter, values in zip(ang_specs, ang_params.T, strict=True):
        print(f"Angular {parameter.name} statistics")
        _ = mean_std(z, values)
    # ----------------------------------------------------------------------- #
    # Axial velocity
    # ----------------------------------------------------------------------- #
    print("Evaluating axial velocity cross-sectional integral and parameters")

    axi_model = CounterFlow()
    axi_specs = axi_model.parameters

    (axi_profs,
     axi_params,
     axi_rel,
     axi_cs) = fit_profile(tdi, 'vel_z_avg', z, axi_model,
                           p0=tuple(p.initial for p in axi_specs),
                           bounds=(tuple(p.lower for p in axi_specs),
                                   tuple(p.upper for p in axi_specs)))

    plot_axial(z, axi_cs, 'u_z', 'axi_vel.pdf', folder=FOLDER)

    plot_radial(z, axi_profs, 'z', join(FOLDER, 'axi_vel'), axi_model,
                axi_params)

    for parameter, values in zip(axi_specs, axi_params.T, strict=True):
        print(f"Axial {parameter.name} statistics")
        _ = mean_std(z, values)

    params = np.hstack((ang_params, axi_params))
    parameter_specs = ang_specs + axi_specs

    # Axial development of parameters and relative error
    print("Plotting fitted parameters and relative errors")
    _plot_parameters(z, params, parameter_specs)

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
               ['axial velocity', 'specific angular momentum',
                'radial velocity'],
               join(FOLDER, 'momentum.h5'),
               units=['m^3/s', 'm^4/s', 'm^3/s'])

    parameter_names = ([f'angular {p.name}' for p in ang_specs]
                       + [f'axial {p.name}' for p in axi_specs])

    save_axial(z, params.T, parameter_names,
               join(FOLDER, 'parameters.h5'),
               [p.description for p in parameter_specs])

    print(f"Analysis complete. Results saved in {FOLDER}/")
