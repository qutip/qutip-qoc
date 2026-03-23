Changelog
=========

Version 0.2.0 (Mar 23, 2026)
----------------------------

Bug Fixes
^^^^^^^^^

Reinforcement learning module (``src/qutip_qoc/_rl.py``):

- Corrections in algorithm execution time #31

JOPT

- Fix: Enable JOPT to support open-system optimization with TRACEDIFF fidelity #49
        - This PR resolves an issue where JOPT failed to optimize open quantum systems using TRACEDIFF fidelity due to incompatibilities between JAX autodiff and Qobj data structures.

GRAPE

- Match grape infidelity with manually computed one #51 (fixes #46):
        - With this PR the GRAPE-reported infidelity matches the manually computed one by evolving the system using the optimized control pulses.

GitHub Workflows

- Update versions of action tools #35

Pulse optimisation and objective modules

- Fix state transfer not working for GRAPE and CRAB #36 (fixes #34)

Documentation
^^^^^^^^^^^^^

- Fixing broken links in README #39
- Fix: Load jQuery explicitly to resolve broken search panel on deployed docs #42

Miscellaneous
^^^^^^^^^^^^^

- Add interactive test notebooks for closed systems #43
- Added requirements for interactive tests to setup.cfg #53
- Make it possible to display qutip-qoc as a family package #56

Dependencies management
^^^^^^^^^^^^^^^^^^^^^^^

- Make all JAX and machine learning related dependencies optional #32

Dependabot dependencies upgrades

- #25
- #26
- #27
- #28
- #45

Version 0.1.1 (Oct 04, 2024)
----------------------------

This is an update to the beta release of ``qutip-qoc``.

It mainly introduces the new reinforcement learning algorithm ``qutip_qoc._rl``.

- Non-public facing functions have been renamed to start with an underscore.
- As with other QuTiP functions, ``optimize_pulses`` now takes a ``tlist`` argument instead of ``_TimeInterval``.
- The structure for the control guess and bounds has changed and now takes in an optional ``__time__`` keyword.
- The ``result`` does no longer return ``optimized_objectives`` but instead ``optimized_H``.

Features
^^^^^^^^

- New reinforcement learning algorithm, developed during GSOC24 (#19, #18, by LegionAtol)
- Automatic transfromation of initial and target operator to superoperator (#23, by flowerthrower)

Bug Fixes
^^^^^^^^^

- Prevent loss of ``__time__`` keyword in ``optimize_pulses`` (#22, by flowerthrower)


Version 0.1.0b1 (July, 2024)
----------------------------

This is the beta release of ``qutip-qoc``, the extended quantum control package in QuTiP.

It has undergone major refactoring and restructuring of the codebase.

- Non-public facing functions have been renamed to start with an underscore.
- As with other QuTiP functions, ``optimize_pulses`` now takes a ``tlist`` argument instead of ``_TimeInterval``.
- The structure for the control guess and bounds has changed and now takes in an optional ``__time__`` keyword.
- The ``result`` does no longer return ``optimized_objectives`` but instead ``optimized_H``.

Bug Fixes
^^^^^^^^^

- basinhopping result does not contain minimizer message
- boundary issues with CRAB


Version 0.0.0 (December 26, 2023)
---------------------------------

This is the alpha version of ``qutip-qoc``, the extended quantum control package in QuTiP.

The ``qutip-qoc`` package builds up on the ``qutip-qtrl`` `package <https://github.com/qutip/qutip-qtrl>`_.
It enhances it by providing two additional algorithms to optimize analytically defined control functions.
The package also aims for a more general way of defining control problems with QuTiP and makes switching between the four control algorithms very easy.

Features
^^^^^^^^

- ``qutip_qoc.GOAT`` is an extension to the Gradient Optimization of Analytic conTrols (GOAT) :cite:`GOAT` algorithm.
  It encoporates an additional time parameterization to allow for optimization over the total evolution time.
- ``qutip_qoc.JOPT`` is an JAX automatic differentiation Optimization of Analytic conTrols (JOPT) algorithm.
- Both algorithms can be addressed by the ``qutip_qoc.optimize_pulses`` function, which consists of a two-layer approach to find global optimal values for parameterized analytical control functions.
  The global optimization layer provides ``scipy.optimize.dual_annealing`` and ``scipy.optimize.basinhopping``, while the local minimization layer supports all gradient driven ``scipy.optimize.minimize`` methods.

Bug Fixes
^^^^^^^^^

- None
