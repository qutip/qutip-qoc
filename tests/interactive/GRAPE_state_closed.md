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

# GRAPE algorithm for a closed system (state transfer)

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, Qobj
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

times = np.linspace(0, 2 * np.pi, 250)
```

## Guess

```python
guess_pulse = np.sin(times)

H_guess = [Hd, [Hc, guess_pulse]]
evolution_guess = qt.sesolve(H_guess, initial_state, times)

print('Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_guess.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], '--', label="Fidelity")
plt.title("Guess performance")
plt.xlabel('Time')
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
        "fid_err_targ": 0.001
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
H_result = [Hd, [Hc, np.array(res_grape.optimized_controls[0])]]
evolution = qt.sesolve(H_result, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")

plt.title('GRAPE performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Validation

```python
assert res_grape.infidelity < 0.01
assert np.allclose(np.abs(evolution.states[-1].overlap(target_state)), 1 - res_grape.infidelity, atol=1e-3)
```

```python
qt.about()
```
