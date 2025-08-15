"""
This file contains the test suite for running the interactive test notebooks
in the 'tests/interactive' directory.
Taken, modified from https://github.com/SeldonIO/alibi/blob/master/testing/test_notebooks.py
"""

import glob
import nbclient.exceptions
import pytest

from pathlib import Path
from jupytext.cli import jupytext
import nbclient

# Set of all example notebooks
NOTEBOOK_DIR = 'tests/interactive'
ALL_NOTEBOOKS = {
    Path(x).name
    for x in glob.glob(str(Path(NOTEBOOK_DIR).joinpath('*.md')))
    if Path(x).name != 'about.md'
}

@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS)
def test_notebook(notebook):
    notebook = Path(NOTEBOOK_DIR, notebook)
    try:
        jupytext(args=[str(notebook), "--execute"])
    except nbclient.exceptions.CellExecutionError as e:
        if e.ename == "Skipped":
            pytest.skip(e.evalue)
        else:
            raise e