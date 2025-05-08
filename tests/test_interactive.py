"""
This file contains the test suite for running the interactive test notebooks
in the 'tests/interactive' directory.
Taken, modified from https://github.com/SeldonIO/alibi/blob/master/testing/test_notebooks.py
"""

import glob
import pytest

from pathlib import Path
from jupytext.cli import jupytext

# Set of all example notebooks
NOTEBOOK_DIR = 'tests/interactive'
ALL_NOTEBOOKS = {
    Path(x).name for x in glob.glob(str(Path(NOTEBOOK_DIR).joinpath('*.md')))
}

@pytest.mark.timeout(600)
@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS)
def test_notebook(notebook):
    notebook = Path(NOTEBOOK_DIR, notebook)
    jupytext(args=[str(notebook), "--execute"])