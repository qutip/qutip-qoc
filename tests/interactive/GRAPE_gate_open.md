---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# GRAPE algorithm for 2 level system


```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, Qobj, gates, liouvillian, fidelity,basis, qeye, sigmam, sigmax, sigmay, sigmaz, tensor)
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
c_ops = [np.sqrt(gamma) * sigmam()]

Hd = 1 / 2 * hbar * omega * sz
Hd = liouvillian(H=Hd, c_ops=c_ops)
Hc = sx
Hc = [liouvillian(sx), liouvillian(sy), liouvillian(sz)]

H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial_gate = qeye(2)
target_gate = Qobj(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))

times = np.linspace(0, np.pi / 2, 250)
```

## Guess


```python
grape_guess = np.sin(times)

H_result_guess = [Hd,
            [Hc[0], grape_guess],
            [Hc[1], grape_guess],
            [Hc[2], grape_guess]]

identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution_guess = qt.mesolve(H_result_guess, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps_guess = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution_guess.states]
target_overlaps_guess = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution_guess.states]

plt.plot(times, initial_overlaps_guess, label="Overlap with initial gate")
plt.plot(times, target_overlaps_guess, label="Overlap with target gate")
plt.title("Guess performance")
plt.xlabel("Time")
plt.legend()
plt.show()
```
## Grape algorithm


```python
control_params = {
    "ctrl_x": {"guess": np.sin(times), "bounds": [-1, 1]},
    "ctrl_y": {"guess": np.cos(times), "bounds": [-1, 1]},
    "ctrl_z": {"guess": np.tanh(times), "bounds": [-1, 1]},
}
alg_args = {"alg": "GRAPE", "fid_err_targ": 0.01}

res_grape = optimize_pulses(
    objectives=Objective(initial_gate, H, target_gate),
    control_parameters=control_params,
    tlist=times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_grape.infidelity)

plt.plot(times, res_grape.optimized_controls[0], label='optimized pulse sx')
plt.plot(times, res_grape.optimized_controls[1], label='optimized pulse sy')
plt.plot(times, res_grape.optimized_controls[2], label='optimized pulse sz')
plt.title('GRAPE pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd,
            [Hc[0], res_grape.optimized_controls[0]],
            [Hc[1], res_grape.optimized_controls[1]],
            [Hc[2], res_grape.optimized_controls[2]]]


identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution = qt.mesolve(H_result, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution.states]
target_overlaps = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution.states]

plt.plot(times, initial_overlaps, label="Overlap with initial gate")
plt.plot(times, target_overlaps, label="Overlap with target gate")
plt.title("GRAPE performance")
plt.xlabel("Time")
plt.legend()
plt.show()
```
## Validation


```python
assert res_grape.infidelity < 0.01
```


```python
qt.about()
```

