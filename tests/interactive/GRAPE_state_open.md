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
from qutip import basis, ket2dm, liouvillian, sigmam, Qobj
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

def fidelity(state, target_state):
    """
    Fidelity used for density matrices in qutip-qtrl and qutip-qoc
    """
    diff = state - target_state
    return 1 - np.real(diff.overlap(diff)) / (2 * target_state.norm())
```

## Problem setup


```python
# Energy levels
E1, E2 = 1.0, 2.0
Hd = Qobj(np.diag([E1, E2]))

gamma = 0.1  # amplitude damping
c_ops = [np.sqrt(gamma) * sigmam()]

Hc = Qobj(np.array([
    [0, 1],
    [1, 0]
]))

Ld = liouvillian(H=Hd, c_ops=c_ops)
Lc = liouvillian(Hc)
L = [Ld, Lc]

initial_state = ket2dm(basis(2, 0))
target_state = ket2dm(basis(2, 1))  

times = np.linspace(0, 2 * np.pi, 250)
```

## Guess


```python
guess_pulse = np.sin(times)

L_guess = [Ld, [Lc, guess_pulse]]
evolution_guess = qt.mesolve(L_guess, initial_state, times)

print('Fidelity: ', fidelity(evolution_guess.states[-1], target_state))

plt.plot(times, [fidelity(state, initial_state) for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [fidelity(state, target_state) for state in evolution_guess.states], label="Overlap with target state")
plt.legend()
plt.title("Guess performance")
plt.xlabel("Time")
plt.show()
```

## GRAPE algorithm


```python
control_params = {
    "ctrl_x": {"guess": guess_pulse, "bounds": [-1, 1]},  # Control pulse for Hc1
}

res_grape = optimize_pulses(
    objectives = Objective(initial_state, L, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GRAPE",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_grape.infidelity)

control_step_function = qt.coefficient(res_grape.optimized_controls[0], tlist=times, order=0)
plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, [control_step_function(t) for t in times], label='optimized pulse')
plt.title('GRAPE pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
L_result = [Ld, [Lc, control_step_function]]
evolution = qt.mesolve(L_result, initial_state, times)

plt.plot(times, [fidelity(state, initial_state) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [fidelity(state, target_state) for state in evolution.states], label="Overlap with target state")
plt.title("GRAPE performance")
plt.xlabel('Time')
plt.legend()
plt.show()

print(1 - fidelity(evolution.states[-1], target_state))
```
```python
state = initial_state
dt = times[1] - times[0]

for c in res_grape.optimized_controls[0][:-1]:
    propagator = ((Ld + c * Lc) * dt).expm()
    state = propagator(state)

print(1 - fidelity(state, target_state))
```

## Validation


```python
assert res_grape.infidelity < 0.001
assert fidelity(evolution.states[-1], target_state) > 1-0.001
```


```python
qt.about()
```