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
from qutip import (about, fidelity, Qobj, basis)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
```

## Problem setup

```python
# Energy levels
E1, E2 = 1.0, 2.0  

Hd = Qobj(np.diag([E1, E2]))
Hc = Qobj(np.array([
    [0, 1],
    [1, 0]
]))
H = [Hd, Hc]

initial_state = basis(2, 0)  # |1>
target_state = basis(2, 1)   # |2>

times = np.linspace(0, 2*np.pi, 100)
```

## CRAB algorithm

```python
n_params = 6 # adjust in steps of 3
alg_args = {"alg": "CRAB", "fid_err_targ": 0.001, "fix_frequency": False} 

res_crab = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters={
        "ctrl_x": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
    },
    tlist=times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_crab.infidelity)


plt.plot(times, res_crab.optimized_controls[0], label='optimized pulse')
plt.title('CRAB pulse')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')

plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, res_crab.optimized_controls[0]]]
evolution = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")
plt.legend()
plt.title("CRAB performance")
plt.show()
```

## Validation

```python
assert res_crab.infidelity < 0.001
assert np.abs(evolution.states[-1].overlap(target_state)) > 1-0.001
```

```python
qt.about()
```

```python

```
