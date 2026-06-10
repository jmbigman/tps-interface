"""Analyzes 2-D plasma data."""

from argparse import ArgumentParser, BooleanOptionalAction

from os.path import join

import numpy as np

import pyvista as pv

from tps_interface import TwoDInterface, time_statistics, TORCH_LENGTH, \
    plot_axial, plot_radial, fit_profile, mean_std, fit_quantity
from tps_interface.models import Angular, Axial, TempSupEll

FOLDER = 'output/plasma'


def _pre_process(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Calculates the axial and angular momenta.

    args:
        mesh: 2-D plasma data

    returns:
        the updated data
    """

    density = mesh.point_data['density']
    vel_z = mesh.point_data['velocity'][:, 1]
    vel_theta = mesh.point_data['swirl']
    radius = mesh.points[:, 0]

    mesh.point_data['axi_m'] = density*vel_z
    mesh.point_data['ang_m'] = -density*radius*vel_theta

    return mesh


parser = ArgumentParser(description="Analysis of 2-D plasma data")
parser.add_argument('-f', type=str, metavar="\b",
                    dest="filename",
                    help="Name of file with 2-D TPS data, typically .pvd")
parser.add_argument('--pre-process', action=BooleanOptionalAction,
                    metavar="\b", default=True, dest='pre_process',
                    help="Pre-process the data for time statistics")
parser.add_argument('-o', metavar="\b", default="plasma_statistics.vtu",
                    dest='output', help="Output filename for time statistics")
parser.add_argument('-t1', type=int, metavar="\b", dest="t1", default=0,
                    help="First time point (integer) to include in statistics")
parser.add_argument('-t2', type=int, metavar="\b", dest="t2", default=-1,
                    help="Last time point (integer) to include in statistics")

if __name__ == '__main__':

    args = parser.parse_args()

    if args.pre_process:
        reader = pv.PVDReader(args.filename)
        mesh = time_statistics(reader, args.t1, args.t2,
                               ['density', 'temperature', 'Yn_Ar', 'Yn_Ar.+1',
                                'Yn_E', 'Yn_Ar_m', 'Yn_Ar_r', 'Yn_Ar_h',
                                'Yn_Ar_p', 'axi_m', 'ang_m'], _pre_process)
        mesh.save(args.output)
    else:
        mesh = pv.read(args.filename)

    tdi = TwoDInterface(mesh)

    # Mesh refined in inlet and step regions
    z = np.linspace(tdi.z_min, 0.02, 10)
    z = np.append(z, np.linspace(z.max(), 0.11, 10))
    z = np.append(z, np.linspace(z.max(), 0.14, 20))
    z = np.append(z, np.linspace(z.max(), TORCH_LENGTH, 40))
    z = np.unique(z)

    # Temperature
    print("Evaluating temperature parameters")

    temp_model = TempSupEll()

    (temp_vals,
     temp_params,
     temp_rel) = fit_quantity(tdi, "temperature_avg", z, temp_model,
                              bounds=([0.0, 2.0], [np.inf, 100.0]))

    plot_axial(z, temp_params[:, 0], r'T_{max}', 'temp_max', False, FOLDER)

    plot_axial(z, temp_params[:, 1], r'\alpha_T', 'temp_shape', False, FOLDER)

    plot_radial(z, temp_vals, 'T', join(FOLDER, 'temp'), temp_model,
                temp_params, False)

    print("Maximum temperature")
    _ = mean_std(z, temp_params[:, 0])

    print("Temperature shape parameter")
    _ = mean_std(z, temp_params[:, 1])

    # Angular momentum
    print("Evaluating angular momentum cross-sectional integral and"
          + " parameters")

    ang_model = Angular()

    (ang_profs,
     ang_params,
     ang_rel,
     ang_cs) = fit_profile(tdi, 'ang_m_avg', z, ang_model)

    plot_axial(z, ang_cs, r'\rho l_z', 'ang_m', folder=FOLDER)

    plot_radial(z, ang_profs, r'\rho l_z', join(FOLDER, 'ang_m'), ang_model,
                ang_params)

    print("Angular parameter")
    _ = mean_std(z, ang_params)

    # Axial momentum
    print("Evaluating axial momentum cross-sectional integral and parameters")

    axi_model = Axial()

    (axi_profs,
     axi_params,
     axi_rel,
     axi_cs) = fit_profile(tdi, 'axi_m_avg', z, axi_model)

    plot_axial(z, axi_cs, r'\rho u_z', 'axi_m', folder=FOLDER)

    plot_radial(z, axi_profs, r'\rho u_z', join(FOLDER, 'axi_m'), axi_model,
                axi_params)

    print("Axial parameter b")
    _ = mean_std(z, axi_params[:, 0])

    print("Axial parameter log(c)")
    _ = mean_std(z, np.log(axi_params[:, 1]))
