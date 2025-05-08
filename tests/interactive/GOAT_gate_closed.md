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

# GOAT algorithm for a closed system

```python
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, numpy
from qutip import (about, Qobj, gates, liouvillian, qeye, sigmam, sigmax, sigmay, sigmaz, tensor)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
```

## Problem setup

```python
hbar = 1
omega = 0.1  # energy splitting
delta = 1.0  # tunneling
gamma = 0.1  # amplitude damping
sx, sy, sz = sigmax(), sigmay(), sigmaz()

Hd = 1 / 2 * hbar * omega * sz
Hc = [sx, sy, sz]
H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial = qeye(2)
target = Qobj(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))

times = np.linspace(0, 2*np.pi, 100)
```

## Guess

```python
guess = [1, 1]
guess_pulse = guess[0] * np.sin(guess[1] * times)

Hresult_guess = [Hd] + [[hc, guess_pulse] for hc in Hc]
evolution_guess = qt.sesolve(Hresult_guess, initial, times)

print('Infidelity: ', qt.fidelity(evolution_guess.states[-1], target))

plt.plot(times, [np.abs(state.overlap(initial) / initial.norm())**2 for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target) / initial.norm())**2 for state in evolution_guess.states], label="Overlap with target state")
plt.legend()
plt.title("Guess performance")
plt.show()
```

## GOAT algorithm
### a) not optimized over time

```python
def sin(t, c):
    return c[0] * np.sin(c[1] * t)


# derivatives
def grad_sin(t, c, idx):
    if idx == 0:  # w.r.t. c0
        return np.sin(c[1] * t)
    if idx == 1:  # w.r.t. c1
        return c[0] * np.cos(c[1] * t) * t
    if idx == 2:  # w.r.t. time
        return c[0] * np.cos(c[1] * t) * c[1]
    
H = [Hd] + [[hc, sin, {"grad": grad_sin}] for hc in Hc]

ctrl_parameters = {
    id: {"guess": guess, "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
    for id in ['x', 'y', 'z']
}
```

```python
# run the optimization
res_goat = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_goat.infidelity)

plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse sx')
plt.plot(times, res_goat.optimized_controls[1], label='optimized pulse sy')
plt.plot(times, res_goat.optimized_controls[2], label='optimized pulse sz')
plt.title('GOAT pulse')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc[0], np.array(res_goat.optimized_controls[0])], [Hc[1], np.array(res_goat.optimized_controls[1])], 
           [Hc[2], np.array(res_goat.optimized_controls[2])]]
evolution = qt.sesolve(Hresult, initial, times)

plt.plot(times, [np.abs(state.overlap(initial) / initial.norm())**2 for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target) / target.norm())**2 for state in evolution.states], label="Overlap with target state")
plt.title('GOAT performance')
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
res_goat_time = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_goat_time.infidelity)
print("time: ", times[-1])
print('optimized time: ', res_goat_time.optimized_params[-1][0])

time_range = times < res_goat_time.optimized_params[-1]

plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], label='optimized (over time) pulse')
plt.title('GOAT pulses')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc[0], np.array(res_goat_time.optimized_controls[0])], [Hc[1], np.array(res_goat_time.optimized_controls[1])], 
            [Hc[2], np.array(res_goat_time.optimized_controls[2])]]
evolution_time = qt.sesolve(Hresult, initial, times)

plt.plot(times, [np.abs(state.overlap(initial) / initial.norm())**2 for state in evolution_time.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target) / target.norm())**2 for state in evolution_time.states], label="Overlap with target state")
plt.xlim(0, res_goat_time.optimized_params[-1][0])

plt.title('GOAT (optimized over time) performance')
plt.xlabel('time')
plt.legend()
plt.show()
```

## Global optimization

```python
res_goat_global = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.001,
    },
    optimizer_kwargs={
       "method": "basinhopping",
       "max_iter": 100,
    }
)

print('Infidelity: ', res_goat_global.infidelity)
print('optimized time: ', res_goat_global.optimized_params[-1])

global_range = times < res_goat_global.optimized_params[-1]

plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[0])[global_range], label='global optimized pulse sx')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[1])[global_range], label='global optimized pulse sy')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[2])[global_range], label='global optimized pulse sz')
plt.title('GOAT pulses (global)')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
Hresult = [Hd, [Hc[0], np.array(res_goat_global.optimized_controls[0])], [Hc[1], np.array(res_goat_global.optimized_controls[1])], 
           [Hc[2], np.array(res_goat_global.optimized_controls[2])]]
evolution_global = qt.sesolve(Hresult, initial, times)

plt.plot(times, [np.abs(state.overlap(initial) / initial.norm()) for state in evolution_global.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target) / target.norm()) for state in evolution_global.states], label="Overlap with target state")
plt.xlim(0, res_goat_global.optimized_params[-1][0])

plt.title('GOAT (global) performance')
plt.xlabel('time')
plt.legend()
plt.show()
```

## Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 4))  # 1 row, 3 columns

titles = ["GOAT s_x pulses", "GOAT s_y pulses", "GOAT s_z pulses"]

for i in range(3):
    ax = axes[i]
    ax.plot(times, sin(times, guess), label='initial guess')
    ax.plot(times, res_goat.optimized_controls[i], color='orange', label='optimized pulse')
    ax.plot(times[time_range], np.array(res_goat_time.optimized_controls[i])[time_range], label='optimized (over time) pulse')
    ax.plot(times[global_range], np.array(res_goat_global.optimized_controls[i])[global_range], label='global optimized pulse')
    ax.set_title(titles[i])
    ax.set_xlabel('time')
    ax.set_ylabel('Pulse amplitude')
    ax.legend()

plt.tight_layout()
plt.show()

```

## Validation

```python
assert res_goat.infidelity < 0.001
assert np.abs(evolution.states[-1].overlap(target)) > 1-0.001
assert res_goat_time.infidelity < 0.001
assert max([np.abs(state.overlap(target)) for state in evolution_time.states]) > 1-0.001
assert res_goat_global.infidelity < 0.001
assert max([np.abs(state.overlap(target)) for state in evolution_global.states]) > 1-0.001

```

```python
qt.about()
```

```python

```
