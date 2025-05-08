---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# GOAT algorithm for a 2 level system

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, Qobj)
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
goat_guess = [1, 0.5]
guess_pulse = goat_guess[0] * np.sin(goat_guess[1] * times)

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

## GOAT algorithm

```python
# control function
def sin(t, c):
    return c[0] * np.sin(c[1] * t)

# gradient
def grad_sin(t, c, idx):
    if idx == 0:  # w.r.t. c0
        return np.sin(c[1] * t)
    if idx == 1:  # w.r.t. c1
        return c[0] * np.cos(c[1] * t) * t
    if idx == 2:  # w.r.t. time
        return c[0] * np.cos(c[1] * t) * c[1]
    
H = [Hd] + [[Hc, sin, {"grad": grad_sin}]]
```

### a) not optimized over time

```python
control_params = {
    "ctrl_x": {"guess": goat_guess, "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
}

# run the optimization
res_goat = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.001
    },
)

print('Infidelity: ', res_goat.infidelity)

plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_goat.optimized_controls[0])]]
evolution = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")

plt.title('GOAT performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

Here, GOAT is stuck in a local minimum and does not reach the desired fidelity.


### b) optimized over time

```python
# treats time as optimization variable
control_params["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}

# run the optimization
res_goat_time = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.001
    },
)

print('Infidelity: ', res_goat_time.infidelity)
print('Optimized time: ', res_goat_time.optimized_params[-1])

time_range = times < res_goat_time.optimized_params[-1]

plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], label='optimized (over time) pulse')
plt.title('GOAT pulses (time optimization)')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_goat_time.optimized_controls[0])]]
evolution_time = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_time.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_time.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_time.states], '--', label="Fidelity")
plt.xlim(0, res_goat_time.optimized_params[-1][0])

plt.title('GOAT (optimized over time) performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

GOAT is still stuck in a local minimum, but the fidelity has improved.


### c) global optimization 

```python
res_goat_global = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.001
    },
    optimizer_kwargs={
       "method": "basinhopping",
       "max_iter": 1000
    }
)

print('Infidelity: ', res_goat_global.infidelity)
print('Optimized time: ', res_goat_global.optimized_params[-1])

global_range = times < res_goat_global.optimized_params[-1]

plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[0])[global_range], label='global optimized pulse')
plt.title('GOAT pulses (global)')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_goat_global.optimized_controls[0])]]
evolution_global = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_global.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_global.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_global.states], '--', label="Fidelity")
plt.xlim(0, res_goat_global.optimized_params[-1][0])

plt.title('GOAT (global) performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Comparison

```python
plt.plot(times, guess_pulse, color='blue', label='initial guess')
plt.plot(times, res_goat.optimized_controls[0], color='orange', label='optimized pulse')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], 
         color='green', label='optimized (over time) pulse')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[0])[global_range], 
         color='red', label='global optimized pulse')

plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
print('Guess Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))
print('GOAT Fidelity: ', 1 - res_goat.infidelity)
print('Time Fidelity: ', 1 - res_goat_time.infidelity)
print('GLobal Fidelity: ', 1 - res_goat_global.infidelity)

plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], color='blue', label="Guess")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], color='orange', label="GOAT")
plt.plot(times[time_range], [qt.fidelity(state, target_state) for state in evolution_time.states[:len(times[time_range])]], 
         color='green', label="Time")
plt.plot(times[global_range], [qt.fidelity(state, target_state) for state in evolution_global.states[:len(times[global_range])]], 
         color='red', label="Global")

plt.title('Fidelities')
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Validation

```python
guess_fidelity = qt.fidelity(evolution_guess.states[-1], target_state)

# target fidelity not reached in part a), check that it is better than the guess
assert 1 - res_goat.infidelity > guess_fidelity
assert np.allclose(np.abs(evolution.states[-1].overlap(target_state)), 1 - res_goat.infidelity, atol=1e-3)

# target fidelity not reached in part b), check that it is better than part a)
assert res_goat_time.infidelity < res_goat.infidelity
assert np.allclose(np.abs(evolution_time.states[len(times[time_range]) - 1].overlap(target_state)), 1 - res_goat_time.infidelity, atol=1e-3)

assert res_goat_global.infidelity < 0.001
assert np.abs(evolution_global.states[len(times[global_range]) - 1].overlap(target_state)) > 1 - 0.001
```

```python
qt.about()
```
