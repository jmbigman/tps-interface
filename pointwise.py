"""Uses ParaView for Python to evaluate derived fields necessary for 1-D
governing equations. All outputs are saved to a `results` folder."""

from os.path import join

from scipy.constants import g

from paraview.simple import XMLPartitionedUnstructuredGridReader, \
    Calculator, Gradient, PassArrays, SaveData


def _start_print(filename: str, data: XMLPartitionedUnstructuredGridReader,
                 verbose: bool) -> None:
    """Prints information about the loaded file.

    args:
        filename: Name of 2-D TPS data in .pvtu file or equivalent.
        data: Loaded data set.
        verbose: Indicator to print information. Default is True.
    """

    if verbose:
        print('-'*80)
        print(f"Loaded data from {filename}")

        fields_str = 'Input file contains point data fields:\n'
        for k in data.PointData.keys():
            fields_str += k + ', '
        print(fields_str[:-2])


def _calculate(data: XMLPartitionedUnstructuredGridReader,
               field_name: str, function: str) \
                -> XMLPartitionedUnstructuredGridReader:
    """Performs a calculation and returns the updated pipeline

    args:
        data: Data at current processing stage
        field_name: Name of newly calculated field
        function: Functional definition of new field

    returns:
        Data set with new field
    """

    calc = Calculator(Input=data)
    calc.ResultArrayName = field_name
    calc.Function = function
    calc.UpdatePipeline()
    return calc


def _extract(data: XMLPartitionedUnstructuredGridReader,
             field_names: list[str]) \
                -> XMLPartitionedUnstructuredGridReader:
    """Extracts the listed fields

    args:
        data: Data at current processing stage
        field_names: Names of fields to keep

    returns:
        Data set with selected fields
    """

    extract = PassArrays(Input=data)
    extract.PointDataArrays = field_names
    extract.UpdatePipeline()
    return extract


def _end_print(filename: str, data: XMLPartitionedUnstructuredGridReader,
               verbose: bool) -> None:
    """Prints information about the saved file.

    args:
        filename: Name of output .pvtu file or equivalent.
        data: Saved data set.
        verbose: Indicator to print information. Default is True.
    """

    if verbose:
        print(f"\nSaved to {filename}")

        fields_str = 'Output file contains point data fields:\n'
        for k in data.PointData.keys():
            fields_str += k + ', '
        print(fields_str[:-2])
        print('-'*80)


OUTPUT_DIR = 'results'

# (NOTE): The wall stress terms are the portions of the viscous tensor that are
#         non-zero at the wall. Usually it is no-slip and no wall turbulence
#         that simplify the expressions.


def angular_momentum(filename: str, verbose: bool = True) -> None:
    """Evaluates the pointwise quantities necessary for the 1-D angular
    momentum governing equation. A new .pvtu file is saved named
    `results/angular_momentum.pvtu` containing the following fields:

    `angular_momentum`: l_z = rho * r * u_z
    `advective_flux`: l_z * u_z
    `viscous_flux`: r * tau_{theta z} = r * (mu + mu_T) * d u_theta / dz
    `wall_stress`: tau_{r theta}^b = mu * d u_theta / dr

    args:
        filename: Name of 2-D TPS data in .pvtu file or equivalent.
        verbose: Indicator to print information. Default is True.
    """

    output_filename = join(OUTPUT_DIR, 'angular_momentum.pvtu')
    output_quantities = ['angular_momentum', 'advective_flux', 'viscous_flux',
                         'wall_stress']

    # Load data
    data = XMLPartitionedUnstructuredGridReader(FileName=filename)
    data.UpdatePipeline()

    _start_print(filename, data, verbose)

    # Gradient of azimuthal velocity
    gradient = Gradient(Input=data)
    gradient.ScalarArray = ['POINT_DATA', 'swirl']
    gradient.ResultArrayName = 'swirl_grad'
    gradient.UpdatePipeline()

    # Angular momentum
    calc = _calculate(gradient, output_quantities[0],
                      'density * coords X * swirl')

    # Advective flux
    calc = _calculate(calc, output_quantities[1],
                      'angular_momentum * velocity_Y')

    # Viscous flux
    calc = _calculate(calc, output_quantities[2],
                      'coordsX * (mu + muT) * swirl_grad_Y')

    # Wall stress
    calc = _calculate(calc, output_quantities[3],
                      'mu * swirl_grad_X')

    extr = _extract(calc, output_quantities)

    SaveData(output_filename, proxy=extr)

    _end_print(output_filename, extr, verbose)


def axial_momentum(filename: str, verbose: bool = True) -> None:
    """Evaluates the pointwise quantities necessary for the 1-D axial
    momentum governing equation. A new .pvtu file is saved named
    `results/axial_momentum.pvtu` containing the following fields:

    `pressure`: p
    `axial_momentum`: rho * u_z
    `advective_flux`: rho * u_z^2
    `viscous_flux`: tau_{z z} = 2/3 * (mu + mu_T)
                              * (- d u_r / dr - u_r / r + 2 * d u_z / dz)
    `wall_stress_rz`: tau_{r z}^b = mu * d u_z / dr
    `wall_stress_zz`: tau_{z z}^b = -2/3 * mu d u_r / dr
    `body_force`: -rho*g
    
    args:
        filename: Name of 2-D TPS data in .pvtu file or equivalent.
        verbose: Indicator to print information. Default is True.
    """

    output_filename = join(OUTPUT_DIR, 'axial_momentum.pvtu')
    output_quantities = ['pressure', 'axial_momentum', 'advective_flux',
                         'viscous_flux', 'wall_stress_rz', 'wall_stress_zz',
                         'body_force']

    # Load data
    data = XMLPartitionedUnstructuredGridReader(FileName=filename)
    data.UpdatePipeline()

    _start_print(filename, data, verbose)

    # Gradient of axial velocity
    gradient = Gradient(Input=data)
    gradient.ScalarArray = ['POINT_DATA', 'velocity_Y']
    gradient.ResultArrayName = 'axial_grad'
    gradient.UpdatePipeline()

    # Gradient of radial velocity
    gradient = Gradient(Input=gradient)
    gradient.ScalarArray = ['POINT_DATA', 'velocity_X']
    gradient.ResultArrayName = 'radial_grad'
    gradient.UpdatePipeline()

    # Pressure is already in the dataset, so can be skipped

    # Axial momentum
    calc = _calculate(gradient, output_quantities[1],
                      'density * velocity_Y')

    # Advective flux
    calc = _calculate(calc, output_quantities[2],
                      'density * velocity_Y * velocity_Y')

    # Viscous flux
    calc = _calculate(calc, output_quantities[3],
                      '(2/3) * (mu + muT) * ' +
                      '(-radial_grad_X - velocity_X/coordsX + 2*axial_grad_Y)')

    # Wall stress rz
    calc = _calculate(calc, output_quantities[4],
                      'mu * axial_grad_X')

    # Wall stress zz
    calc = _calculate(calc, output_quantities[5],
                      '-(2/3) * mu * radial_grad_X')

    # Body force
    calc = _calculate(calc, output_quantities[6],
                      '-' + str(g) + '* density')

    # Extract fields for simple output
    extr = _extract(calc, output_quantities)

    SaveData(output_filename, proxy=extr)

    _end_print(output_filename, extr, verbose)


def radial_momentum(filename: str, verbose: bool = True) -> None:
    """Evaluates the pointwise quantities necessary for the 1-D radial
    momentum governing equation. A new .pvtu file is saved named
    `results/radial_momentum.pvtu` containing the following fields:

    `pressure`: p
    `radial_momentum`: rho * u_r
    `advective_flux`: rho * u_r *u_z
    `viscous_flux`: tau_{r z} = (mu + mu_T) * (d u_r / dz + d u_z / dr)
    `wall_stress_rr`: 4/3 * mu * d u_r / dr
    `wall_stress_rz`: mu * d u_z / dr
    `centrifugal`: rho * u_theta^2
    `stress_tt`: tau_{theta theta} =  2/3 * (mu + mu_T)
                                   * (- d u_r / dr + 2*u_r / r - d u_z / dz)

    args:
        filename: Name of 2-D TPS data in .pvtu file or equivalent.
        verbose: Indicator to print information. Default is True.
    """

    output_filename = join(OUTPUT_DIR, 'radial_momentum.pvtu')
    output_quantities = ['pressure', 'radial_momentum', 'advective_flux',
                         'viscous_flux', 'wall_stress_rr', 'wall_stress_rz',
                         'centrifugal', 'stress_tt']

    # Load data
    data = XMLPartitionedUnstructuredGridReader(FileName=filename)
    data.UpdatePipeline()

    _start_print(filename, data, verbose)

    # Gradient of axial velocity
    gradient = Gradient(Input=data)
    gradient.ScalarArray = ['POINT_DATA', 'velocity_Y']
    gradient.ResultArrayName = 'axial_grad'
    gradient.UpdatePipeline()

    # Gradient of radial velocity
    gradient = Gradient(Input=gradient)
    gradient.ScalarArray = ['POINT_DATA', 'velocity_X']
    gradient.ResultArrayName = 'radial_grad'
    gradient.UpdatePipeline()

    # Pressure is already in the dataset, so can be skipped

    # Radial momentum
    calc = _calculate(gradient, output_quantities[1],
                      'density * velocity_X')

    # Advective flux
    calc = _calculate(calc, output_quantities[2],
                      'density * velocity_X * velocity_Y')

    # Viscous flux
    calc = _calculate(calc, output_quantities[3],
                      '(mu + muT) * (radial_grad_Y + axial_grad_X)')

    # Wall stress rr
    calc = _calculate(calc, output_quantities[4],
                      '(4/3) * mu * radial_grad_X')

    # Wall stress rz
    calc = _calculate(calc, output_quantities[5],
                      'mu * axial_grad_X')

    # Centrifugal force
    calc = _calculate(calc, output_quantities[6],
                      'density * swirl * swirl')

    # Viscous stress theta theta component
    calc = _calculate(calc, output_quantities[7],
                      '(2/3) * (mu + muT) * ' +
                      '(-radial_grad_X + 2*velocity_X/coordsX - axial_grad_Y)')

    # Extract fields for simple output
    extr = _extract(calc, output_quantities)

    SaveData(output_filename, proxy=extr)

    _end_print(output_filename, extr, verbose)


def density(filename: str, verbose: bool = True) -> None:
    """Evaluates the pointwise quantities necessary for the 1-D density
    governing equation. A new .pvtu file is saved named
    `results/density.pvtu` containing the following fields:

    `density`: rho
    `advective_flux`: rho * u_z

    args:
        filename: Name of 2-D TPS data in .pvtu file or equivalent.
        verbose: Indicator to print information. Default is True.
    """

    output_filename = join(OUTPUT_DIR, 'density.pvtu')
    output_quantities = ['density', 'advective_flux']

    # Load data
    data = XMLPartitionedUnstructuredGridReader(FileName=filename)
    data.UpdatePipeline()

    _start_print(filename, data, verbose)

    # Density is already in the dataset, so can be skipped

    # Advective flux
    calc = _calculate(data, output_quantities[1],
                      'density * velocity_Y')

    # Extract fields for simple output
    extr = _extract(calc, output_quantities)

    SaveData(output_filename, proxy=extr)

    _end_print(output_filename, extr, verbose)

if __name__ == "__main__":

    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(description="Evaluates pointwise quantities for"
                            + " the 1-D governing equations from 2-D data.")
    parser.add_argument('-f', '--filename', type=str, metavar="\b",
                        dest="filename",
                        help="Name of .pvtu file with 2-D TPS data")
    parser.add_argument('-v', '--verbose', type=bool,
                        action=BooleanOptionalAction, default=True,
                        dest="verbose", help="Verbose output")
    args = parser.parse_args()

    angular_momentum(args.filename, args.verbose)
    axial_momentum(args.filename, args.verbose)
    radial_momentum(args.filename, args.verbose)
    density(args.filename, args.verbose)
