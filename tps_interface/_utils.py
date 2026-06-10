"""Miscellaneous utility functions."""

import numpy as np

# Torch length from 2-D inlet condition to end of nozzle [m]
TORCH_LENGTH = 0.34

# Wall temperature [K]
TEMP_W = 300.0


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
