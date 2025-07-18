.. _control:

*********************************************
Quantum Optimal Control
*********************************************


Introduction
============

In quantum control we look to prepare some specific state, effect some state-to-state transfer, or effect some transformation (or gate) on a quantum system. For a given quantum system there will always be factors that affect the dynamics that are outside of our control. As examples, the interactions between elements of the system or a magnetic field required to trap the system. However, there may be methods of affecting the dynamics in a controlled way, such as the time varying amplitude of the electric component of an interacting laser field. And so this leads to some questions; given a specific quantum system with a known time-independent dynamics generator (referred to as the *drift* dynamics generator) and a set of externally controllable fields for which the interaction can be described by *control* dynamics generators:

1. What states or transformations can we achieve (if any)?

2. What is the shape of the control pulse required to achieve this?

These questions are addressed as *controllability* and *quantum optimal control* :cite:`dAless08`. The answer to question of *controllability* is determined by the commutability of the dynamics generators and is formalised as the *Lie Algebra Rank Criterion* and is discussed in detail in :cite:`dAless08`. The solutions to the second question can be determined through optimal control algorithms, or control pulse optimisation.

.. figure:: figures/quant_optim_ctrl.png
   :align: center
   :width: 2.5in

   Schematic showing the principle of quantum control.

Quantum Control has many applications including NMR, *quantum metrology*, *control of chemical reactions*, and *quantum information processing*.

To explain the physics behind these algorithms we will first consider only finite-dimensional, closed quantum systems.

Closed Quantum Systems
======================
In closed quantum systems the states can be represented by kets, and the transformations on these states are unitary operators. The dynamics generators are Hamiltonians. The combined Hamiltonian for the system is given by

.. math::

    H(t) = H_0 + \sum_{j=1} u_j(t) H_j

where :math:`H_0` is the drift Hamiltonian and the :math:`H_j` are the control Hamiltonians. The :math:`u_j` are time varying amplitude functions for the specific control.

The dynamics of the system are governed by *Schrödingers equation*.

.. math::

    \tfrac{d}{dt} \ket{\psi} = -i H(t)\ket{\psi}

Note that we use units where :math:`\hbar=1` throughout. The solutions to Schrödinger's equation are of the form:

.. math::

    \ket{\psi(t)} = U(t)\ket{\psi_0}

where :math:`\ket{\psi_0}` is the state of the system at :math:`t=0` and :math:`U(t)` is a unitary operator on the Hilbert space containing the states. :math:`U(t)` is a solution to the *Schrödinger operator equation*

.. math::

    \tfrac{d}{dt}U = -i H(t)U ,\quad U(0) = \mathbb{1}

We can use optimal control algorithms to determine a set of :math:`u_j` that will drive our system from :math:`\ket{\psi_0}` to :math:`\ket{\psi_1}`, this is state-to-state transfer, or drive the system from some arbitary state to a given state :math:`\ket{\psi_1}`, which is state preparation, or effect some unitary transformation :math:`U_{target}`, called gate synthesis. The latter of these is most important in quantum computation.

The GOAT Algorithm
===================
The GOAT method, like CRAB, operates with analytical control functions :cite:`GOAT`.
It constructs a system of coupled equations of motion, enabling the calculation of the derivative of the evolution operator (time-ordered) with respect to the control parameters, following numerical forward integration.
This implementation supports arbitrary control functions along with their corresponding derivatives.
To further accelerate convergence, our implementation enhances the original algorithm by providing the option to optimize controls with respect to the overall pulse duration.

The GRAPE algorithm
===================
The **GR**\ adient **A**\ scent **P**\ ulse **E**\ ngineering was first proposed in :cite:`NKanej`. Solutions to Schrödinger's equation for a time-dependent Hamiltonian are not generally possible to obtain analytically. Therefore, a piecewise constant approximation to the pulse amplitudes is made. Time allowed for the system to evolve :math:`T` is split into :math:`M` timeslots (typically these are of equal duration), during which the control amplitude is assumed to remain constant. The combined Hamiltonian can then be approximated as:

.. math::

    H(t) \approx H(t_k) = H_0 + \sum_{j=1}^N u_{jk} H_j\quad

where :math:`k` is a timeslot index, :math:`j` is the control index, and :math:`N` is the number of controls. Hence :math:`t_k` is the evolution time at the start of the timeslot, and :math:`u_{jk}` is the amplitude of control :math:`j` throughout timeslot :math:`k`. The time evolution operator, or propagator, within the timeslot can then be calculated as:

.. math::

    X_k:=e^{-iH(t_k)\Delta t_k}

where :math:`\Delta t_k` is the duration of the timeslot. The evolution up to (and including) any timeslot :math:`k` (including the full evolution :math:`k=M`) can the be calculated as

.. math::

    X(t_k):=X_k X_{k-1}\cdots X_1 X_0

If the objective is state-to-state transfer then :math:`X_0=\ket{\psi_0}` and the target :math:`X_{targ}=\ket{\psi_1}`, for gate synthesis :math:`X_0 = U(0) = \mathbb{1}` and the target :math:`X_{targ}=U_{targ}`.

A *figure of merit* or *fidelity* is some measure of how close the evolution is to the target, based on the  control amplitudes in the timeslots. The typical figure of merit for unitary systems is the normalised overlap of the evolution and the target.

.. math::
    \DeclareMathOperator{\tr}{tr}
    f_{PSU} = \tfrac{1}{d} \big| \tr \{X_{targ}^{\dagger} X(T)\} \big|

where :math:`d` is the system dimension. In this figure of merit the absolute value is taken to ignore any differences in global phase, and :math:`0 \le f \le 1`. Typically the fidelity error (or *infidelity*) is more useful, in this case defined as :math:`\varepsilon = 1 - f_{PSU}`.  There are many other possible objectives, and hence figures of merit.

As there are now :math:`N \times M` variables (the :math:`u_{jk}`) and one
parameter to minimise :math:`\varepsilon`, then the problem becomes a finite
multi-variable optimisation problem, for which there are many established
methods, often referred to as 'hill-climbing' methods. The simplest of these to
understand is that of steepest ascent (or descent). The gradient of the
fidelity with respect to all the variables is calculated (or approximated) and
a step is made in the variable space in the direction of steepest ascent (or
descent). This method is a first order gradient method. In two dimensions this
describes a method of climbing a hill by heading in the direction where the
ground rises fastest. This analogy also clearly illustrates one of the main
challenges in multi-variable optimisation, which is that all methods have a
tendency to get stuck in local maxima. It is hard to determine whether one has
found a global maximum or not - a local peak is likely not to be the highest
mountain in the region. In quantum optimal control we can typically define an
infidelity that has a lower bound of zero. We can then look to minimise the
infidelity (from here on we will only consider optimising for infidelity
minima). This means that we can terminate any pulse optimisation when the
infidelity reaches zero (to a sufficient precision). This is however only
possible for fully controllable systems; otherwise it is hard (if not
impossible) to know that the minimum possible infidelity has been achieved. In
the hill walking analogy the step size is roughly fixed to a stride, however,
in computations the step size must be chosen. Clearly there is a trade-off here
between the number of steps (or iterations) required to reach the minima and
the possibility that we might step over a minima. In practice it is difficult
to determine an efficient and effective step size.

The second order differentials of the infidelity with respect to the variables
can be used to approximate the local landscape to a parabola. This way a step
(or jump) can be made to where the minima would be if it were parabolic. This
typically vastly reduces the number of iterations, and removes the need to
guess a step size. The method where all the second differentials are calculated
explicitly is called the *Newton-Raphson* method. However, calculating the
second-order differentials (the Hessian matrix) can be computationally
expensive, and so there are a class of methods known as *quasi-Newton* that
approximate the Hessian based on successive iterations. The most popular of
these (in quantum optimal control) is the Broyden–Fletcher–Goldfarb–Shanno
algorithm (BFGS). The default method in the QuTiP QOC GRAPE implementation is
the L-BFGS-B method in Scipy, which is a wrapper to the implementation
described in :cite:`Byrd95`. This limited memory and bounded method does not need to
store the entire Hessian, which reduces the computer memory required, and
allows bounds to be set for variable values, which considering these are field
amplitudes is often physical.

The pulse optimisation is typically far more efficient if the gradients can be
calculated exactly, rather than approximated. For simple fidelity measures such
as :math:`f_{PSU}` this is possible. Firstly the propagator gradient for each
timeslot with respect to the control amplitudes is calculated. For closed
systems, with unitary dynamics, a method using the eigendecomposition is used,
which is efficient as it is also used in the propagator calculation (to
exponentiate the combined Hamiltonian). More generally (for example open
systems and symplectic dynamics) the Frechet derivative (or augmented matrix)
method is used, which is described in :cite:`Flo12`. For other optimisation goals it
may not be possible to calculate analytic gradients. In these cases it is
necessary to approximate the gradients, but this can be very expensive, and can
lead to other algorithms out-performing GRAPE.


The CRAB Algorithm
===================
It has been shown :cite:`Lloyd14`, the dimension of a quantum optimal control
problem is a polynomial function of the dimension of the manifold of the
time-polynomial reachable states, when allowing for a finite control precision
and evolution time. You can think of this as the information content of the
pulse (as being the only effective input) being very limited e.g. the pulse is
compressible to a few bytes without loosing the target.

This is where the **C**\ hopped **RA**\ ndom **B**\ asis (CRAB) algorithm
:cite:`Doria11`, :cite:`Caneva11` comes into play: Since the pulse complexity is usually
very low, it is sufficient to transform the optimal control problem to a few
parameter search by introducing a physically motivated function basis that
builds up the pulse. Compared to the number of time slices needed to accurately
simulate quantum dynamics (often equals basis dimension for Gradient based
algorithms), this number is lower by orders of magnitude, allowing CRAB to
efficiently optimize smooth pulses with realistic experimental constraints. It
is important to point out, that CRAB does not make any suggestion on the basis
function to be used. The basis must be chosen carefully considered, taking into
account a priori knowledge of the system (such as symmetries, magnitudes of
scales,...) and solution (e.g. sign, smoothness, bang-bang behavior,
singularities, maximum excursion or rate of change,....). By doing so, this
algorithm allows for native integration of experimental constraints such as
maximum frequencies allowed, maximum amplitude, smooth ramping up and down of
the pulse and many more. Moreover initial guesses, if they are available, can
(however not have to) be included to speed up convergence.

As mentioned in the GRAPE paragraph, for CRAB local minima arising from
algorithmic design can occur, too. However, for CRAB a 'dressed' version has
recently been introduced :cite:`Rach15` that allows to escape local minima.

For some control objectives and/or dynamical quantum descriptions, it is either
not possible to derive the gradient for the cost functional with respect to
each time slice or it is computationally expensive to do so. The same can apply
for the necessary (reverse) propagation of the co-state. All this trouble does
not occur within CRAB as those elements are not in use here. CRAB, instead,
takes the time evolution as a black-box where the pulse goes as an input and
the cost (e.g. infidelity) value will be returned as an output. This concept,
on top, allows for direct integration in a closed loop experimental environment
where both the preliminarily open loop optimization, as well as the final
adoption, and integration to the lab (to account for modeling errors,
experimental systematic noise, ...) can be done all in one, using this
algorithm.


The RL Algorithm
================
Reinforcement Learning (RL) represents a different approach compared to traditional
quantum control methods, such as GRAPE and CRAB. Instead of relying on gradients or
prior knowledge of the system, RL uses an agent that autonomously learns to optimize
control policies by interacting with the quantum environment.

The RL algorithm consists of three main components:

**Agent**: The RL agent is responsible for making decisions regarding control
parameters at each time step. The agent observes the current state of the quantum
system and chooses an action (i.e., a set of control parameters) based on the current policy.
**Environment**: The environment represents the quantum system that evolves over time.
The environment is defined by the system's dynamics, which include drift and control Hamiltonians.
Each action chosen by the agent induces a response in the environment, which manifests as an
evolution of the system's state. From this, a reward can be derived.
**Reward**: The reward is a measure of how much the action chosen by the agent brings the
quantum system closer to the desired objective. In this context, the objective could be the
preparation of a specific state, state-to-state transfer, or the synthesis of a quantum gate.

Each interaction between the agent and the environment defines a step.
A sequence of steps forms an episode. The episode ends when certain conditions, such as reaching
a specific fidelity, are met.
The reward function is a crucial component of the RL algorithm, carefully designed to
reflect the objective of the quantum control problem.
It guides the algorithm in updating its policy to maximize the reward obtained during the various
training episodes.
For example, in a state-to-state transfer problem, the reward is based on the fidelity
between the achieved final state and the desired target state.
In addition, a constant penalty term is subtracted in order to encourages the agent to reach the
objective in as few steps as possible.

In QuTiP, the RL environment is modeled as a custom class derived from the gymnasium library.
This class allows defining the quantum system's dynamics at each step, the actions the agent
can take, the observation space, and so on. The RL agent is trained using the Proximal Policy Optimization
(PPO) algorithm from the stable baselines3 library.


Optimal Quantum Control in QuTiP
================================
Defining a control problem with QuTiP is very easy.
The objective is to find a pulse that will drive some system from an initial state or operator represntation to a desired target representation.
Both initial and target can be specified through ``Qobj`` instances.

.. code-block:: bash

  import qutip as qt

  # state to state transfer
  initial = qt.basis(2, 0)
  target = qt.basis(2, 1)

  # gate synthesis
  initial = qt.qeye(2)
  target = qt.sigmax()


The system evovles under some drift Hamiltonian or Liouvillian, that can be expressed with a ``QobjEvo`` instance.
Instead of defining the full ``QobjEvo`` object, it is sufficient to only specify a list of Hamiltonians and possible control functions
to construct the objective (similar to initializing ``QobjEvo``).

.. code-block:: bash

  import qutip_qoc as qoc

  drift = qt.sigmaz()

  # discretized control
  control = [[qt.sigmax(), np.ones(100)],
             [qt.sigmay(), np.ones(100)]]

  # continuous control
  control = [[qt.sigmax(), lambda t, p: p[0] * t + p[1]],
             [qt.sigmay(), lambda t, q: p[0] * t + p[1]]]

  H = [drift, control]

  objective = qoc.Objective(initial, H, target)


The control problem is then fully defined by the ``qutip_qoc.Objective`` class.


Running the optimization
========================

After having defined the control problem, the ``qutip_qoc.optimize_pulses`` function can be used to find an optimal control pulse.
It requires some extra arguments to prepare the optimization.

.. code-block:: bash

  # initial parameters to be optimized
  p_guess = q_guess = [0., 0.]

  # boundaries for the parameters
  p_bounds = q_bounds = [(-1, 1), (-1, 1)]


Eventually, the optimization for a desired `fid_err_targ` can be started by calling the ``optimize`` function.

.. code-block:: bash

  result = qoc.optimize_pulses(
    objectives=[objective], # list of objectives
    control_parameters={
      "p": {"guess": p_guess, "bounds": p_bounds},
      "q": {"guess": q_guess, "bounds": q_bounds},
    },
    tlist=np.linspace(0, 1, 100),
    algorithm_kwargs={
      "fid_err_targ": 0.1,
      "alg": "GOAT",
    },
  )

The Genetic Algorithm (GA)
==========================

The Genetic Algorithm (GA) is a global optimization technique inspired by natural selection. 
Unlike gradient-based methods like GRAPE or CRAB, GA explores the solution space stochastically, 
making it robust against local minima and suitable for problems with non-differentiable or noisy objectives.

In QuTiP, the GA-based optimizer evolves a population of candidate control pulses across multiple 
generations to minimize the infidelity between the system's final and target states.

The GA optimization consists of the following components:

**Population**:
    A collection of candidate solutions (chromosomes), where each chromosome encodes a full set 
    of control amplitudes over time for the given control Hamiltonians.

**Fitness Function**:
    The fitness of each candidate is evaluated using a fidelity-based measure, such as:

    - **PSU (Projective State Overlap)**: 
      :math:`1 - |\langle \psi_{\text{target}} | \psi_{\text{final}} \rangle|`
    - **TRACEDIFF**: 
      Trace distance between final and target density matrices.

**Genetic Operations**:

- **Selection**:
  A subset of the best-performing candidates (based on fitness) are chosen to survive.
  
- **Crossover (Mating)**:
  New candidates are generated by combining genes from selected parents using arithmetic crossover.
  
- **Mutation**:
  Random perturbations are added to the new candidates' genes to maintain diversity and escape local minima.

This process continues until either a target fidelity error is reached or a fixed number of generations 
have passed without improvement (stagnation criterion).

Each generation represents a full evaluation of the population, making the method inherently parallelizable 
and effective in high-dimensional control landscapes.

In QuTiP, the GA optimization is implemented via the ``_Genetic`` class, and can be invoked using the 
standard ``optimize_pulses`` interface by setting the algorithm to ``"Genetic"``.

Optimal Quantum Control in QuTiP (Genetic Algorithm)
====================================================

Defining a control problem in QuTiP using the Genetic Algorithm follows the same pattern as with other algorithms.

.. code-block:: python

    import qutip as qt
    import qutip_qoc as qoc
    import numpy as np

    # state to state transfer
    initial = qt.basis(2, 0)
    target = qt.basis(2, 1)

    # define the drift and control Hamiltonians
    drift = qt.sigmaz()
    controls = [qt.sigmax(), qt.sigmay()]

    H = [drift] + controls

    # define the objective
    objective = qoc.Objective(initial, H, target)

    # discretized time grid (e.g., 100 steps over 10 units of time)
    tlist = np.linspace(0, 10, 100)

    # define control parameter bounds (same for all controls in GA)
    control_parameters = {
        "p": {"bounds": [(-1.0, 1.0)]}
    }

    # set genetic algorithm hyperparameters
    algorithm_kwargs = {
        "alg": "Genetic",
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.2,
        "fid_err_targ": 1e-3,
    }

    # run the optimization
    result = qoc.optimize_pulses(
        objectives=[objective],
        control_parameters=control_parameters,
        tlist=tlist,
        algorithm_kwargs=algorithm_kwargs,
    )

    print("Final infidelity:", result.infidelity)
    print("Best control parameters:", result.optimized_params)

.. TODO: add examples

Examples for Liouvillian dynamics and multi-objective optimization will follow soon.
