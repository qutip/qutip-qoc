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

# JOPT algorithm for a 2 level system


```python
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, numpy
from qutip import (about, fidelity, Qobj, basis)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
```

## Problem setup

```python
# Energy levels
E1, E2 = 1.0, 1.0  

Hd = Qobj(np.diag([E1, E2]))
Hc = Qobj(np.array([
    [0, 1],
    [1, 0]
]))
H = [Hd, Hc]

initial_state = basis(2, 0)  # |1>
target_state = basis(2, 1)   # |2>

times = np.linspace(0, 2 * np.pi, 100)
```

## Guess

```python
jopt_guess = [1, 0.5]
guess_pulse = jopt_guess[0] * np.sin(jopt_guess[1] * times)

Hresult_guess = [Hd, [Hc, guess_pulse]]
evolution_guess = qt.sesolve(Hresult_guess, initial_state, times)

print('Infidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_guess.states], label="Initial Overlap")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_guess.states], label="Target Overlap")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], '--', label="Fidelity")
plt.legend()
plt.title("Guess performance")
plt.show()
```

## JOPT algorithm

```python
@jit
def sin_x(t, c, **kwargs):
    return c[0] * numpy.sin(c[1] * t)

H = [Hd] + [[Hc, sin_x]]

ctrl_parameters = {
    id: {"guess": [1, 0], "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
}
```

### a) not optimized over time

```python
res_jopt = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=ctrl_parameters,
    tlist=times,
    minimizer_kwargs={
        "method": "Nelder-Mead",

    },
    algorithm_kwargs={
        "alg": "JOPT",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_jopt.infidelity)

plt.plot(times, sin_x(times, jopt_guess), label='initial guess')
plt.plot(times, res_jopt.optimized_controls[0], label='optimized pulse')
plt.title('JOPT pulses')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_jopt.optimized_controls[0])]]
evolution = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state))**2 for state in evolution.states], label="Overlap with intiial state")
plt.plot(times, [np.abs(state.overlap(target_state))**2 for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")

plt.title('JOPT performance')
plt.xlabel('time')
plt.legend()
plt.show()

```

### b) optimized over time

```python
# treats time as optimization variable
ctrl_parameters["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}

# run the optimization
res_jopt_time = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=ctrl_parameters,
    tlist=times,
    minimizer_kwargs={
        "method": "Nelder-Mead",

    },
    algorithm_kwargs={
        "alg": "JOPT",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_jopt_time.infidelity)
print('optimized time: ', res_jopt_time.optimized_params[-1])

time_range = times < res_jopt_time.optimized_params[-1]

plt.plot(times, res_jopt.optimized_controls[0], color='orange', label='optimized pulse')
plt.plot(times[time_range], np.array(res_jopt.optimized_controls[0])[time_range], 
         color='green', label='optimized (over time) pulse')
plt.title('JOPT pulses')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc, np.array(res_jopt_time.optimized_controls[0])]]
evolution_time = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state))**2 for state in evolution_time.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state))**2 for state in evolution_time.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_time.states], '--', label="Fidelity")
plt.xlim(0, res_jopt_time.optimized_params[-1][0])

plt.title('JOPT (optimized over time) performance')
plt.xlabel('time')
plt.legend()
plt.show()
```

## Global optimization 

```python
res_jopt_global = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs={
        "alg": "JOPT",
        "fid_err_targ": 0.001,
    },
    optimizer_kwargs={
       "method": "basinhopping",
       "max_iter": 1000,
    }
)

print('Infidelity: ', res_jopt_global.infidelity)
print('optimized time: ', res_jopt_global.optimized_params[-1])

global_range = times < res_jopt_global.optimized_params[-1]

plt.plot(times, res_jopt.optimized_controls[0], color='orange', label='optimized pulse')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[0])[global_range], color='red', label='global optimized pulse')
plt.title('JOPT pulses (global)')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()

```

```python
Hresult = [Hd, [Hc, np.array(res_jopt_global.optimized_controls[0])]]
evolution_global = qt.sesolve(Hresult, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state))**2 for state in evolution_global.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state))**2 for state in evolution_global.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_global.states], '--', label="Fidelity")
plt.xlim(0, res_jopt_global.optimized_params[-1][0])

plt.title('JOPT (global) performance')
plt.xlabel('time')
plt.legend()
plt.show()
```

## Comparison

```python
plt.plot(times, sin_x(times, jopt_guess), color='blue', label='initial guess')
plt.plot(times, res_jopt.optimized_controls[0], color='orange', label='optimized pulse')
plt.plot(times[time_range], np.array(res_jopt_time.optimized_controls[0])[time_range], 
         color='green', label='optimized (over time) pulse')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[0])[global_range], 
         color='red', label='global optimized pulse')
plt.title('GOAT pulses')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
print('Guess Infidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))
print('JOPT Infidelity: ', res_jopt.infidelity)
print('Time Infidelity: ', res_jopt_time.infidelity)
print('GLobal Infidelity: ', res_jopt_global.infidelity)

plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], color='blue', label="Guess")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], color='orange', label="Goat")
plt.plot(times[time_range], [qt.fidelity(state, target_state) for state in evolution_time.states[:len(times[time_range])]], 
         color='green', label="Time")
plt.plot(times[global_range], [qt.fidelity(state, target_state) for state in evolution_global.states[:len(times[global_range])]], 
         color='red', label="Global")

np.array(res_jopt_global.optimized_controls[0])[global_range],

plt.title('Fidelities')
plt.xlabel('time')
plt.legend()
plt.show()
```

## Validation

```python
assert res_jopt.infidelity < 0.02
assert np.abs(evolution.states[-1].overlap(target_state)) > 1-0.02
assert res_jopt_time.infidelity < 0.001
assert max([np.abs(state.overlap(target_state)) for state in evolution_time.states]) > 1-0.001
assert res_jopt_global.infidelity < 0.001
assert max([np.abs(state.overlap(target_state)) for state in evolution_global.states]) > 1-0.001
```

```python
qt.about()
```

```python

```
