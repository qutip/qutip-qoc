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

## CRAB algorithm

```python
n_params = 6 # adjust in steps of 3
control_params = {
    "ctrl_x": {
        "guess": [1 for _ in range(n_params)],
        "bounds": [(-1, 1)] * n_params,
    },
}
alg_args = {"alg": "CRAB", "fid_err_targ": 0.001, "fix_frequency": False} 

res_crab = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=control_params,
    tlist=times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_crab.infidelity)

plt.plot(times, res_crab.optimized_controls[0], label='optimized pulse')
plt.title('CRAB pulse')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd, [Hc, res_crab.optimized_controls[0]]]
evolution = qt.mesolve(H_result, initial_state, times, c_ops)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")
plt.title("CRAB performance")
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Validation

```python
assert res_crab.infidelity < 0.01
```

```python
qt.about()
```

```python

```
