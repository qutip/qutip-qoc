.. _introduction:

************
Introduction
************

QuTiP - Quantum Optimal Control
===============================

The qutip-qoc package builds up on the ``qutip-qtrl`` `package <https://github.com/qutip/qutip-qtrl>`_.

It enhances it by providing two additional algorithms to optimize analytically defined control functions.
The first one is an extension to Gradient Optimization of Analytic conTrols (GOAT) :cite:`GOAT`.
The second one (JOPT) leverages QuTiPs version 5 new diffrax abilities to directly calculate gradients of JAX defined control functions using automatic differentiation.

Both algorithms consist of a two-layer approach to find global optimal values for parameterized analytical control functions.
The global optimization layer provides ``scipy.optimize.dual_annealing`` and ``scipy.optimize.basinhopping``, while the local minimization layer supports all
gradient driven ``scipy.optimize.minimize`` methods.

The package also aims for a more general way of defining control problems with QuTiP and makes switching between the four control algorithms (GOAT, JOPT, and GRAPE and CRAB implemented in ``qutip-qtrl``) very easy.

As with qutip-qtrl, the qutip-qoc package aims at providing advanced tools for the optimal control of quantum devices.
Compared to other libraries for quantum optimal control, qutip-qoc puts additional emphasis on the physics layer and the interaction with the QuTiP package.
Along with the extended GOAT and JOPT algorithms the package offers support for both the CRAB and GRAPE methods defined in ``qutip-qtrl``.

Citing
======

If you use `qutip-qoc` in your research, please cite the original QuTiP papers that are available `here <https://dml.riken.jp/?s=QuTiP>`_.
