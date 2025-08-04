# TPS-radial-profiles
The analysis code is all written in `python` but is split into two parts. The first script, `pointwise.py`,
uses ParaView's `pvpython` to evaluate certain derived quantities, like stresses, pointwise. To run it:
```
pvpython pointwise.py -f (--filename) FILENAME -v (--verbosity) VERBOSITY
```
where `FILENAME` is the 2-D `.pvtu` output from `TPS`.

The second script, `one_d_terms.py`, evaluates the various terms in the 1-D evolution equations. This class
relies on the package `pyvista`, which is not included with `pvpython`. `pvpython` does not come with `pip`,
but it seems possible to external packages through a virtual environment [^1] [^2]. Otherwise the script can
be run from any virtual environment with `pyvista` installed:
```
python one_d_terms.py
```

[^1]: https://www.kitware.com/using-python-virtual-environments-in-paraview-5-13-0/
[^2]: https://mbarzegary.github.io/2022/01/03/use-python-packages-modules-in-paraview/