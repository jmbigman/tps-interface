"""Miscellaneous utility functions."""

from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np

# Torch length from 2-D inlet condition to end of nozzle [m]
TORCH_LENGTH = 0.34

# Wall temperature [K]
TEMP_W = 300.0


def create_analysis_parser(description: str, default_output: str
                           ) -> ArgumentParser:
    """Creates the shared argument parser for cold flow and plasma analysis.

    args:
        description: Description of the analysis shown in the help text
        default_output: Default output filename for time statistics

    returns:
        the argument parser
    """

    parser = ArgumentParser(description=description)

    parser.add_argument('-f', type=str, metavar="\b", dest='filename',
                        help="Name of file with 2-D TPS data, typically .pvd")

    parser.add_argument('--pre-process', action=BooleanOptionalAction,
                        metavar="\b", default=True, dest='pre_process',
                        help="Pre-process the data for time statistics")

    parser.add_argument('-o', metavar="\b", default=default_output,
                        dest='output',
                        help="Output filename for time statistics")

    parser.add_argument('-t1', type=int, metavar="\b", dest='t1', default=0,
                        help="First time-point index to include")

    parser.add_argument('-t2', type=int, metavar="\b", dest='t2', default=None,
                        help="Last time-point index to include"
                        + " (default: last)")

    return parser


def mean_std(z: np.ndarray, f: np.ndarray, verbose: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the mean and standard deviation of the quantity along the
    axial coordinate using the trapezoidal rule.

    args:
        z: Axial positions [m]
        f: Evaluated quantity at the axial positions
        verbose: Indicator to print results, optional. Default is True.

    returns:
        the mean,
        the standard deviation
    """

    length = z[-1] - z[0]

    f = f.flatten()

    mean = np.trapezoid(f, z)/length

    std = np.sqrt(np.trapezoid((f - mean)**2, z)/length)

    if verbose:
        print(f"\tMinimum: {f.min()}")
        print(f"\tMaximum: {f.max()}")
        print(f"\tMean: {mean}")
        print(f"\tStandard deviation: {std}")

    return mean, std
