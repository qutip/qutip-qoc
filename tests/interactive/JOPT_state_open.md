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

# JOPT algorithm for a 2 level system



```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, fidelity, liouvillian, ket2dm, Qobj, basis, sigmam)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

try:
    from jax import jit, numpy
except ImportError:  # JAX not available, skip test
    import pytest
    pytest.skip("JAX not available")
```

## Problem setup


```python
# Energy levels
E1, E2 = 1.0, 2.0  

hbar = 1
omega = 0.1  # energy splitting
delta = 1.0  # tunneling
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
jopt_guess = [1, 0.5]
guess_pulse = jopt_guess[0] * np.sin(jopt_guess[1] * times)

H_result_guess = [Hd, [Hc, guess_pulse]]
evolution_guess = qt.mesolve(H_result_guess, initial_state, times)

print('Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))

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

H = [Hd, [Hc, sin_x]]
```

### a) not optimized over time


```python
ctrl_parameters = {
    id: {"guess": [1, 0], "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
}
alg_args = {
    "alg": "JOPT",
    "fid_err_targ": 0.01,
}

res_jopt = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=ctrl_parameters,
    tlist=times,
    minimizer_kwargs={
        "method": "Nelder-Mead",
    },
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_jopt.infidelity)

plt.plot(times, sin_x(times, jopt_guess), label='initial guess')
plt.plot(times, res_jopt.optimized_controls[0], label='optimized pulse')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd, [Hc, np.array(res_jopt.optimized_controls[0])]]
evolution = qt.sesolve(H_result, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with intiial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")
plt.title('JOPT performance')
plt.xlabel('Time')
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
alg_args = {
    "alg": "JOPT",
    "fid_err_targ": 0.001,
}

# run the optimization
res_jopt_time = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=ctrl_parameters,
    tlist=times,
    minimizer_kwargs={
        "method": "Nelder-Mead",

    },
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_jopt_time.infidelity)
print('optimized time: ', res_jopt_time.optimized_params[-1])

time_range = times < res_jopt_time.optimized_params[-1]

plt.plot(times, res_jopt.optimized_controls[0], label='Optimized pulse')
plt.plot(times[time_range], np.array(res_jopt.optimized_controls[0])[time_range], label='Optimized (over time) pulse')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result_time = [Hd, [Hc, np.array(res_jopt_time.optimized_controls[0])]]
evolution_time = qt.sesolve(H_result_time, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_time.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_time.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_time.states], '--', label="Fidelity")
plt.title('JOPT performance (optimized over time)')
plt.xlabel('Time')
plt.xlim(0, res_jopt_time.optimized_params[-1][0])
plt.legend()
plt.show()
```
## Global optimization 


```python
opt_args = {
    "method": "basinhopping",
    "max_iter": 1000,
}

res_jopt_global = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs=alg_args,
    optimizer_kwargs=opt_args
)

global_time = res_jopt_global.optimized_params[-1]
global_range = times < global_time

print('Infidelity: ', res_jopt_global.infidelity)
print('Optimized time: ', res_jopt_global.optimized_params[-1])

plt.plot(times, res_jopt.optimized_controls[0], label='Optimized pulse')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[0])[global_range], label='Global optimized pulse')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result_global = [Hd, [Hc, np.array(res_jopt_global.optimized_controls[0])]]
evolution_global = qt.sesolve(H_result_global, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state))**2 for state in evolution_global.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state))**2 for state in evolution_global.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_global.states], '--', label="Fidelity")
plt.xlim(0, res_jopt_global.optimized_params[-1][0])
plt.title('JOPT performance (global)')
plt.xlabel('Time')
plt.legend()
plt.show()
```
## Comparison


```python
plt.plot(times, sin_x(times, jopt_guess), label='Initial guess')
plt.plot(times, res_jopt.optimized_controls[0], label='Optimized pulse')
plt.plot(times[time_range], np.array(res_jopt_time.optimized_controls[0])[time_range], label='Optimized (over time) pulse')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[0])[global_range], label='Global optimized pulse')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```
```python
print('Guess Infidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))
print('JOPT Infidelity: ', res_jopt.infidelity)
print('Time Infidelity: ', res_jopt_time.infidelity)
print('GLobal Infidelity: ', res_jopt_global.infidelity)

plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], label="Guess")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], label="Goat")
plt.plot(times[time_range], [qt.fidelity(state, target_state) for state in evolution_time.states[:len(times[time_range])]], label="Time")
plt.plot(times[global_range], [qt.fidelity(state, target_state) for state in evolution_global.states[:len(times[global_range])]], label="Global")

plt.title('Fidelities')
plt.xlabel('time')
plt.legend()
plt.show()
```

## Validation


```python
assert res_jopt.infidelity < 0.001
assert res_jopt_time.infidelity < 0.001
assert res_jopt_global.infidelity < 0.001
```


```python
qt.about()
```

