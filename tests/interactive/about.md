This directory contains notebooks showcasing the use of all algorithms contained within `QuTiP-QOC`.
The examples are chosen as simple as possible, for the purpose of demonstrating how to use `QuTiP-QOC` in all of these scenarios, and for the purpose of automatically testing `QuTiP-QOC`'s basic functionality in all of these scenarios.
The included algorithms are:
- GRAPE
- CRAB
- GOAT
- JOPT

For each algorithm, we have:
- a closed-system state transfer example
- an open-system state transfer example
- a closed-system gate synthesis example
- an open-system gate synthesis example

The notebooks are included automatically in runs of the test suite (see `test_interactive.py`).

To view and run the notebooks manually, the `jupytext` package is required.
The notebooks can then either be opened from within Jupyter Notebook using "Open With" -> "Jupytext Notebook", or by converting them first to the `ipynb` format using
```
jupytext --to ipynb [notebook name].nb
```