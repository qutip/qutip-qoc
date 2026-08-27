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

# CRAB algorithm for an open system (state transfer)

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, ket2dm, liouvillian, sigmam, Qobj
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

def fidelity(dm, target_dm):
    """
    Fidelity used for density matrices in qutip-qtrl and qutip-qoc
    """

    diff = dm - target_dm
    return 1 - np.real(diff.overlap(diff) / target_dm.norm()) / 2
```

## Problem setup


```python
# Energy levels
E1, E2 = 1.0, 2.0

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
    "ctrl_x": {"guess": [1 for _ in range(n_params)], "bounds": [(-1, 1)] * n_params},
}

res_crab = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "CRAB",
        "fid_err_targ": 0.02
    },
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
evolution = qt.mesolve(H_result, initial_state, times)

plt.plot(times, [fidelity(dm, initial_state) for dm in evolution.states], label="Overlap with initial state")
plt.plot(times, [fidelity(dm, target_state) for dm in evolution.states], label="Overlap with target state")

plt.title("CRAB performance")
plt.xlabel('Time')
plt.legend()
plt.show()
```
## Validation


```python
assert res_crab.infidelity < 0.02
assert np.allclose(fidelity(evolution.states[-1], target_state), 1 - res_crab.infidelity, atol=1e-3)
```


```python
qt.about()
```