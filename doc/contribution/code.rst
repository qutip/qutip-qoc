.. _contribute_code:

*******************************
Contributing to the source code
*******************************

Build up an development environment
===================================

Please follow the instruction on the `QuTiP contribution guide <https://qutip.readthedocs.io/en/latest/development/contributing.html>`_ to
build a conda environment.

You don't need to build ``qutip`` in the editable mode unless you also want to contribute to `qutip`.
Instead, you need to install ``qutip-qoc`` by downloading the source code and running

.. code-block:: bash

    pip install -e .

Docstrings for the code
=======================

Each class and function should be accompanied with a docstring
explaining the functionality, including input parameters and returned values.
The docstring should follow
`NumPy Style Python Docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html>`_.

Checking Code Style and Format
==============================

The Python code is formatted and linted using a collection of `pre-commit hooks <https://pre-commit.com/>`_.
This collection includes:

- `ruff <https://docs.astral.sh/ruff/>`_ -- an extremely fast Python linter and formatter, written in Rust.
- `mypy <http://mypy-lang.org/>`_ -- a static type checker for Python code


You can install the hooks manually by running :code:`pre-commit install` in the project root directory.
The hooks will then be executed automatically when committing changes.

    .. code-block:: console

        (venv) $ pre-commit run -a

In order to check if your code in ``some_file.py`` follows `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_
style guidelines, `Black <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_
can be installed.

.. code-block::

  pip install black

In the directory that contains ``some_file.py``, use

.. code-block::

  black some_file.py --check
  black some_file.py --diff --color
  black some_file.py

Using ``--check`` will show if any of the file will be reformatted or not.

  * `Code 0 <https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#the-basics>`_ means nothing will be reformatted.
  * Code 1 means one or more files could be reformatted. More than one files could
    be reformatted if ``black some_directory --check`` is used.

Using ``--diff --color`` will show a `difference <https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#diffs>`_ of
the changes that will be made by ``Black``. If you would prefer these changes to be made, use the last line of above code block.

Checking tests locally
=======================

You can run tests and generate code coverage report locally. First make sure
required packages have been installed.

.. code-block::

  pip install pytest pytest-cov

``pytest`` is used to test files containing tests. If you would like to test all the
files contained in a directory then specify the path to this directory. In order to run
tests in ``test_something.py`` then specify the exact path to this file for ``pytest``
or navigate to the file before running the tests.

.. code-block::

  pytest path_to_some_directory
  pytest /path_to_test_something/test_something.py
  ~/path_to_test_something$ pytest test_something.py

A code coverage report in ``html`` format  can be generated locally for
``qutip-qoc`` using the code line given below. By default the coverage report
is generated in a temporary directory ``htmlcov``. The report can be output
in `other formats <https://pytest-cov.readthedocs.io/en/latest/reporting.html>`_
besides ``html``.

.. code-block::

  pytest --cov-report html --cov=qutip_qoc tests/

If you would prefer to check the code coverage of one specific file, specify
the location of this file. Same as above the report can be accessed in ``htmlcov``.

.. code-block::

  pytest --cov-report html --cov=qutip_qoc tests/test_something.py
