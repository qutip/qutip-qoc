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

# CRAB algorithm for an open system (gate synthesis)

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import gates, qeye, liouvillian, sigmam, sigmax, sigmay, sigmaz
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

def fidelity(gate_super, target_super):
    gate_oper = qt.Qobj(gate_super.data)
    target_oper = qt.Qobj(target_super.data)
    
    return np.abs(gate_oper.overlap(target_oper) / target_oper.norm())
```

## Problem setup


```python
omega = 0.1  # energy splitting
gamma = 0.1  # amplitude damping
sx, sy, sz = sigmax(), sigmay(), sigmaz()
c_ops = [np.sqrt(gamma) * sigmam()]

Hd = 1 / 2 * omega * sz
Hd = liouvillian(H=Hd, c_ops=c_ops)
Hc = [liouvillian(sx), liouvillian(sy), liouvillian(sz)]
H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial_gate = qeye(2)
target_gate = gates.hadamard_transform()

times = np.linspace(0, np.pi / 2, 250)
```

## Crab algorithm


```python
n_params = 6 # adjust in steps of 3
control_params = {
    "ctrl_x": {"guess": [1 for _ in range(n_params)], "bounds": [(-1, 1)] * n_params},
    "ctrl_y": {"guess": [1 for _ in range(n_params)], "bounds": [(-1, 1)] * n_params},
    "ctrl_z": {"guess": [1 for _ in range(n_params)], "bounds": [(-1, 1)] * n_params},
}
alg_args = {"alg": "CRAB", "fid_err_targ": 0.01}

res_crab = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "CRAB",
        "fid_err_targ": 0.01
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
initial_super = qt.to_super(initial_gate)
target_super = qt.to_super(target_gate)

H_result = [Hd,
            [Hc[0], np.array(res_crab.optimized_controls[0])],
            [Hc[1], np.array(res_crab.optimized_controls[1])],
            [Hc[2], np.array(res_crab.optimized_controls[2])]]
evolution = qt.mesolve(H_result, initial_super, times)

plt.plot(times, [fidelity(gate, initial_super) for gate in evolution.states], label="Overlap with initial gate")
plt.plot(times, [fidelity(gate, target_super) for gate in evolution.states], label="Overlap with target gate")

plt.title("CRAB performance")
plt.xlabel("Time")
plt.legend()
plt.show()
```
## Validation


```python
assert res_crab.infidelity < 0.01
assert np.allclose(fidelity(evolution.states[-1], target_super), 1 - res_crab.infidelity, atol=1e-3)
```


```python
qt.about()
```
