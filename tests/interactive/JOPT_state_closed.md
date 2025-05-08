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

# JOPT algorithm for a 2 level system


```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, Qobj)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

try:
    from jax import jit
    from jax import numpy as jnp
except ImportError:  # JAX not available, skip test
    import pytest
    pytest.skip("JAX not available")
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
jopt_guess = [1, 0.5]
guess_pulse = jopt_guess[0] * np.sin(jopt_guess[1] * times)

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

## JOPT algorithm

```python
@jit
def sin(t, c, **kwargs):
    return c[0] * jnp.sin(c[1] * t)

H = [Hd] + [[Hc, sin]]
```

### a) not optimized over time

```python
control_params = {
    "ctrl_x": {"guess": [1, 0], "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
}

res_jopt = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    minimizer_kwargs = {
        "method": "Nelder-Mead",
    },
    algorithm_kwargs = {
        "alg": "JOPT",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_jopt.infidelity)

plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_jopt.optimized_controls[0], label='optimized pulse')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_jopt.optimized_controls[0])]]
evolution = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")

plt.title('JOPT performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

Here, JOPT is stuck in a local minimum and does not reach the desired fidelity.


### b) optimized over time

```python
# treats time as optimization variable
control_params["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}

# run the optimization
res_jopt_time = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    minimizer_kwargs = {
        "method": "Nelder-Mead",
    },
    algorithm_kwargs = {
        "alg": "JOPT",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_jopt_time.infidelity)
print('Optimized time: ', res_jopt_time.optimized_params[-1])

time_range = times < res_jopt_time.optimized_params[-1]

plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_jopt.optimized_controls[0], label='optimized pulse')
plt.plot(times[time_range], np.array(res_jopt.optimized_controls[0])[time_range], label='optimized (over time) pulse')
plt.title('JOPT pulses (time optimization)')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_jopt_time.optimized_controls[0])]]
evolution_time = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_time.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_time.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_time.states], '--', label="Fidelity")
plt.xlim(0, res_jopt_time.optimized_params[-1][0])

plt.title('JOPT (optimized over time) performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

JOPT is still stuck in a local minimum, but the fidelity has improved.


### c) global optimization 

```python
res_jopt_global = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "JOPT",
        "fid_err_targ": 0.001
    },
    optimizer_kwargs={
       "method": "basinhopping",
       "max_iter": 1000
    }
)

print('Infidelity: ', res_jopt_global.infidelity)
print('Optimized time: ', res_jopt_global.optimized_params[-1])

global_range = times < res_jopt_global.optimized_params[-1]

plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_jopt.optimized_controls[0], label='optimized pulse')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[0])[global_range], label='global optimized pulse')
plt.title('JOPT pulses (global)')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_jopt_global.optimized_controls[0])]]
evolution_global = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_global.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_global.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_global.states], '--', label="Fidelity")
plt.xlim(0, res_jopt_global.optimized_params[-1][0])

plt.title('JOPT (global) performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Comparison

```python
plt.plot(times, guess_pulse, color='blue', label='initial guess')
plt.plot(times, res_jopt.optimized_controls[0], color='orange', label='optimized pulse')
plt.plot(times[time_range], np.array(res_jopt_time.optimized_controls[0])[time_range], 
         color='green', label='optimized (over time) pulse')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[0])[global_range], 
         color='red', label='global optimized pulse')

plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
print('Guess Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))
print('JOPT Fidelity: ', 1 - res_jopt.infidelity)
print('Time Fidelity: ', 1 - res_jopt_time.infidelity)
print('GLobal Fidelity: ', 1 - res_jopt_global.infidelity)

plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], color='blue', label="Guess")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], color='orange', label="Goat")
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
assert 1 - res_jopt.infidelity > guess_fidelity
assert np.allclose(np.abs(evolution.states[-1].overlap(target_state)), 1 - res_jopt.infidelity, atol=1e-3)

# target fidelity not reached in part b), check that it is better than part a)
assert res_jopt_time.infidelity < res_jopt.infidelity
assert np.allclose(np.abs(evolution_time.states[len(times[time_range]) - 1].overlap(target_state)), 1 - res_jopt_time.infidelity, atol=1e-3)

assert res_jopt_global.infidelity < 0.001
assert np.abs(evolution_global.states[len(times[global_range]) - 1].overlap(target_state)) > 1 - 0.001
```

```python
qt.about()
```
