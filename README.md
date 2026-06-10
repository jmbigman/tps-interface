# TPS-radial-profiles
Reads and analyzes 2- and 3-D `Torch Plasma Simulator` (`TPS`) simulation results for 1-D modeling.
`TPS` can be found here: [https://github.com/pecos/tps](https://github.com/pecos/tps).

## Installation
The latest Python version with support for the `vtk` package is 3.13: [https://vtk.org/download/](https://vtk.org/download/). To install (on Mac):
```
brew install python@3.13
python3.13 -m venv venv-interface
source venv-interface/bin/activate
pip3 install -e .
```

## Modules
`two_d_interface`: Evaluates quantities needed for 1-D modeling from 2-D `TPS` simulation results

`plotting`: Plots the torch radius, cross-sectionally integrated quantities, radial profiles, etc.

`model_profiles`: Samples radial profiles from `TPS` data and fits parameters in model profiles

## Scripts
`cold_flow.py`: Analyzes time-dependent 2-D cold flow data
`plasma.py`: Analyzes time-dependent 2-D plasma data

## Development
To lint the code, run:
```
flake8 tps_interface scripts
```
