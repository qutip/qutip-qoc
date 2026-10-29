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

# CRAB algorithm for a closed system (gate synthesis)

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import gates, qeye, sigmax, sigmay, sigmaz
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

def fidelity(gate, target_gate):
    """
    Fidelity used for unitary gates in qutip-qtrl and qutip-qoc
    """
    return np.abs(gate.overlap(target_gate) / target_gate.norm())
```

## Problem setup

```python
omega = 0.1  # energy splitting
sx, sy, sz = sigmax(), sigmay(), sigmaz()

Hd = 1 / 2 * omega * sz
Hc = [sx, sy, sz]
H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial_gate = qeye(2)
target_gate = gates.hadamard_transform()

times = np.linspace(0, np.pi / 2, 250)
```

## CRAB algorithm

```python
n_params = 3  # adjust in steps of 3
control_params = {
    "ctrl_x": {"guess": [1 for _ in range(n_params)], "bounds": [(-1, 1)] * n_params},
    "ctrl_y": {"guess": [1 for _ in range(n_params)], "bounds": [(-1, 1)] * n_params},
    "ctrl_z": {"guess": [1 for _ in range(n_params)], "bounds": [(-1, 1)] * n_params},
}

res_crab = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "CRAB",
        "fid_err_targ": 0.001
    },
)

print('Infidelity: ', res_crab.infidelity)

plt.plot(times, res_crab.optimized_controls[0], 'b', label='optimized pulse sx')
plt.plot(times, res_crab.optimized_controls[1], 'g', label='optimized pulse sy')
plt.plot(times, res_crab.optimized_controls[2], 'r', label='optimized pulse sz')
plt.title('CRAB pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd, [Hc[0], res_crab.optimized_controls[0]], [Hc[1], res_crab.optimized_controls[1]], [Hc[2], res_crab.optimized_controls[2]]]
evolution = qt.sesolve(H_result, initial_gate, times)

plt.plot(times, [fidelity(gate, initial_gate) for gate in evolution.states], label="Overlap with initial gate")
plt.plot(times, [fidelity(gate, target_gate) for gate in evolution.states], label="Overlap with target gate")

plt.title('CRAB performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Validation

```python
assert res_crab.infidelity < 0.001
assert np.allclose(fidelity(evolution.states[-1], target_gate), 1 - res_crab.infidelity, atol=1e-3)
```

```python
qt.about()
```
