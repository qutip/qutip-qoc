*********
Changelog
*********


Version 0.0.0 (December 26, 2023)
+++++++++++++++++++++++++++++++++

This is the first release of qutip-qoc, the extended quantum control package in QuTiP.

The qutip-qoc package builds up on the ``qutip-qtrl`` `package <https://github.com/qutip/qutip-qtrl>`_.
It enhances it by providing two additional algorithms to optimize analytically defined control functions.
The package also aims for a more general way of defining control problems with QuTiP and makes switching between the four control algorithms very easy.

Features
--------

- ``qutip_qoc.GOAT`` is an extension to the Gradient Optimization of Analytic conTrols (GOAT) :cite:`GOAT` algorithm.
    It encoporates an additional time parameterization to allow for optimization over the total evolution time.
- ``qutip_qoc.JOAT`` is an JAX automatic differentiation Optimization of Analytic conTrols (JOAT) algorithm.  
- Both algorithms can be addressed by the ``qutip_qoc.optimize_pulses`` function, which consists of a two-layer approach to find global optimal values for parameterized analytical control functions.
    The global optimization layer provides ``scipy.optimize.dual_annealing`` and ``scipy.optimize.basinhopping``, while the local minimization layer supports all gradient driven ``scipy.optimize.minimize`` methods.


Bug Fixes
---------

- None