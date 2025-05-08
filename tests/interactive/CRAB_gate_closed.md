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

# CRAB algorithm for a closed system

```python
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, numpy
from qutip import (about, Qobj, gates, qeye, sigmam, sigmax, sigmay, sigmaz, tensor)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
```

## Problem setup

```python
hbar = 1
omega = 0.1  # energy splitting
delta = 1.0  # tunneling
gamma = 0.1  # amplitude damping
sx, sy, sz = sigmax(), sigmay(), sigmaz()

Hd = 1 / 2 * hbar * omega * sz
Hc = [sx, sy, sz]
H = [Hd, Hc[0], Hc[1], Hc[2]]

initial = qeye(2)
target = Qobj(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))

times = np.linspace(0, 2*np.pi / 2, 100)
```

## CRAB algotihm

```python
n_params = 3  # adjust in steps of 3
alg_args = {"alg": "CRAB", "fid_err_targ": 0.001}

res_crab = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters={
        "ctrl_x": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
        "ctrl_y": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
        "ctrl_z": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
    },
    tlist=times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_crab.infidelity)

plt.plot(times, res_crab.optimized_controls[0], label='optimized pulse sx')
plt.plot(times, res_crab.optimized_controls[1], label='optimized pulse sy')
plt.plot(times, res_crab.optimized_controls[2], label='optimized pulse sz')
plt.title('CRAB pulse')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc[0], res_crab.optimized_controls[0]], [Hc[1], res_crab.optimized_controls[1]], [Hc[2], res_crab.optimized_controls[2]]]
evolution = qt.sesolve(Hresult, initial, times)


plt.plot(times, [np.abs(state.overlap(target) / target.norm()) for state in evolution.states], '--', label="Fidelity")
plt.legend()
plt.title("GRAPE performance")
plt.show()
```

## Validation

```python
assert res_crab.infidelity < 0.001
assert np.abs(evolution.states[-1].overlap(target) / target.norm()) > 1-0.001
```

```python
qt.about()
```

```python

```
