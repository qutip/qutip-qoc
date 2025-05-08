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

# GRAPE algorithm for a closed system

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

Hd = 1 / 2 * hbar * omega * sz
Hc = [sx, sy, sz]
H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial_gate = qeye(2)
target_gate = Qobj(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))

times = np.linspace(0, np.pi / 2, 100)
```

## Guess

```python
guess = np.sin(times)

Hresult_guess = [Hd] + [[hc, guess] for hc in Hc]
evolution_guess = qt.sesolve(Hresult_guess, initial_gate, times)

print('Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_gate))

plt.plot(times, [np.abs(state.overlap(initial_gate) / initial_gate.norm())**2 for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_gate) / target_gate.norm())**2 for state in evolution_guess.states], label="Overlap with target state")
plt.legend()
plt.title("Guess performance")
plt.show()
```

## GRAPE algorithm

```python
control_params = {
    "ctrl_x": {"guess": np.sin(times), "bounds": [-1, 1]},
    "ctrl_y": {"guess": np.cos(times), "bounds": [-1, 1]},
    "ctrl_z": {"guess": np.tanh(times), "bounds": [-1, 1]},
}

res_grape = optimize_pulses(
    objectives=Objective(initial_gate, H, target_gate),
    control_parameters=control_params,
    tlist=times,
    algorithm_kwargs={"alg": "GRAPE", "fid_err_targ": 0.001},
)

print('Infidelity: ', res_grape.infidelity)


plt.plot(times, res_grape.optimized_controls[0], label='optimized pulse sx')
plt.plot(times, res_grape.optimized_controls[1], label='optimized pulse sy')
plt.plot(times, res_grape.optimized_controls[2], label='optimized pulse sz')
plt.title('GRAPE pulses')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc[0], res_grape.optimized_controls[0]], [Hc[1], res_grape.optimized_controls[1]], [Hc[2], res_grape.optimized_controls[2]]]
evolution = qt.sesolve(Hresult, initial_gate, times)


plt.plot(times, [np.abs(state.overlap(target_gate) / target_gate.norm())**2 for state in evolution.states], label="target overlap")
plt.plot(times, [np.abs(state.overlap(initial_gate) / initial_gate.norm())**2 for state in evolution.states], label="initial overlap")
plt.legend()
plt.title("GRAPE performance")
plt.show()
```

## Validation

```python
assert res_grape.infidelity < 0.001
assert np.abs(evolution.states[-1].overlap(target_gate)) > 1-0.001
```

```python
qt.about()
```

```python

```
