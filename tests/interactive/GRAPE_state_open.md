---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: qutip-dev
    language: python
    name: python3
---

# GRAPE algorithm for 2 level system

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, fidelity, liouvillian, ket2dm, Qobj, basis, sigmam)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

import logging
```

## Problem setup

```python
# Energy levels
E1, E2 = 1.0, 2.0  

hbar = 1
omega = 0.1  # energy splitting
delta = 1.0  # tunneling
gamma = 0.1  # amplitude damping
c_ops = [np.sqrt(gamma) * sigmam()]

Hd = Qobj(np.diag([E1, E2]))
Hd = liouvillian(H=Hd, c_ops=c_ops)
Hc = Qobj(np.array([
    [0, 1],
    [1, 0]
])) 
Hc = liouvillian(Hc)
H = [Hd, Hc]

initial_state = ket2dm(basis(2, 0))
target_state = ket2dm(basis(2, 1))  

times = np.linspace(0, 2 * np.pi, 250)
```

## Guess

```python
grape_guess = np.sin(times)

Hresult_guess = [Hd, [Hc, grape_guess]]
evolution_guess = qt.mesolve(Hresult_guess, initial_state, times)

print('Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_guess.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], '--', label="Fidelity")
plt.legend()
plt.title("Guess performance")
plt.xlabel("Time")
plt.show()
```

## GRAPE algorithm

```python
alg_args = {"alg": "GRAPE", "fid_err_targ": 0.001, "log_level": logging.DEBUG - 2}
control_params = {
    "ctrl_1": {"guess": grape_guess, "bounds": [-1, 1]},  # Control pulse for Hc1
}

res_grape = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_grape.infidelity)

plt.plot(times, grape_guess, label='initial guess')
plt.plot(times, res_grape.optimized_controls[0], label='optimized pulse')
plt.title('GRAPE pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd, [Hc, res_grape.optimized_controls[0]]]
evolution = qt.mesolve(H_result, initial_state, times, c_ops)

plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states])
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")
plt.title("CRAB performance")
plt.xlabel('Time')
plt.legend()
plt.ylim(0, 1)
plt.show()
```

## Validation

```python
assert res_grape.infidelity < 0.01
```

```python
qt.about()
```

```python

```
