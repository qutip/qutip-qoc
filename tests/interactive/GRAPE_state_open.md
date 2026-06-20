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

# GRAPE algorithm for an open system (state transfer)

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

## Guess


```python
guess_pulse = np.sin(times)

H_guess = [Hd, [Hc, guess_pulse]]
evolution_guess = qt.mesolve(H_guess, initial_state, times)

print('Fidelity: ', fidelity(evolution_guess.states[-1], target_state))

plt.plot(times, [fidelity(state, initial_state) for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [fidelity(state, target_state) for state in evolution_guess.states], label="Overlap with target state")
plt.title("Guess performance")
plt.xlabel("Time")
plt.legend()
plt.show()
```

## GRAPE algorithm


```python
control_params = {
    "ctrl_x": {"guess": guess_pulse, "bounds": [-1, 1]},  # Control pulse for Hc1
}

res_grape = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GRAPE",
        "fid_err_targ": 0.01,
    },
)

print('Infidelity: ', res_grape.infidelity)

plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_grape.optimized_controls[0], label='optimized pulse')
plt.title('GRAPE pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
control_step_function = qt.coefficient(res_grape.optimized_controls[0], tlist=times, order=0)
H_result = [Hd, [Hc, control_step_function]]

fine_times = np.linspace(0, times[-1], len(times) * 25)
evolution = qt.mesolve(H_result, initial_state, fine_times)

plt.plot(fine_times, [fidelity(state, initial_state) for state in evolution.states], label="Overlap with initial state")
plt.plot(fine_times, [fidelity(state, target_state) for state in evolution.states], label="Overlap with target state")
plt.title("GRAPE performance")
plt.xlabel('Time')
plt.legend()
plt.show()
```
## Validation


```python
assert res_grape.infidelity < 0.01
assert np.allclose(fidelity(evolution.states[-1], target_state), 1 - res_grape.infidelity, atol=1e-3)
```


```python
qt.about()
```
