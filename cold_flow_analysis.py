"""Analysis of 2-D cold flow data."""

from argparse import ArgumentParser

import numpy as np

import pyvista as pv

from two_d_interface import time_statistics, TwoDInterface, TORCH_LENGTH
from plotting import plot_radius, plot_cs_integral, plot_profiles


def _pre_process(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Calculates angular momentum, separates velocity components into scalars.

    args:
        mesh: 2-D cold flow data

    returns:
        The update data
    """

    mesh.point_data['ang_m'] = mesh.points[:, 0]*mesh.point_data['swirl']

    mesh.point_data['vel_r'] = mesh.point_data['velocity'][:, 0]
    mesh.point_data['vel_z'] = mesh.point_data['velocity'][:, 1]

    return mesh


parser = ArgumentParser(description="Analysis of 2-D cold flow data")
parser.add_argument('-f', '--filename', type=str, metavar="\b",
                    dest="filename",
                    help="Name of .pvtu file with 2-D TPS data")
parser.add_argument('-t1', type=int, metavar="\b", dest="t1",
                    help="First time point to include in statistics")
parser.add_argument('-t2', type=int, metavar="\b", dest="t2", default=-1,
                    help="Last time point to include in statistics")

if __name__ == '__main__':

    args = parser.parse_args()

    reader = pv.PVDReader(args.filename)

    mesh = time_statistics(reader, args.t1, args.t2,
                           ['ang_m', 'vel_r', 'vel_z'], _pre_process)

    mesh.save('momentum_statistics.vtu')

    tdi = TwoDInterface(mesh)

    plot_radius(tdi)

    plot_cs_integral(tdi, 'ang_m_avg',
                     np.linspace(0.01, TORCH_LENGTH, 500),
                     r'r u_\theta', 100)

    plot_cs_integral(tdi, 'vel_r_avg',
                     np.linspace(0.01, TORCH_LENGTH, 500),
                     r'u_r', 100)

    plot_cs_integral(tdi, 'vel_z_avg',
                     np.linspace(0.01, TORCH_LENGTH, 500),
                     r'u_z', 100)

    plot_profiles(tdi, 'ang_m_avg', np.linspace(0.01, TORCH_LENGTH, 50),
                  r'r u_\theta', 100)

    plot_profiles(tdi, 'vel_r_avg', np.linspace(0.01, TORCH_LENGTH, 50),
                  r'u_r', 100)

    plot_profiles(tdi, 'vel_z_avg', np.linspace(0.01, TORCH_LENGTH, 50),
                  r'u_z', 100)
