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

# GOAT algorithm for an open system (state transfer)

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
goat_guess = [1, 0.5]
guess_pulse = goat_guess[0] * np.sin(goat_guess[1] * times)

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
    "ctrl_x": {"guess": goat_guess, "bounds": [(-3, 3), (0, 2 * np.pi)]}  # c0 and c1
}

# run the optimization
res_goat = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.01
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
H_result = [Hd, [Hc, np.array(res_goat.optimized_controls[0])]]
evolution = qt.mesolve(H_result, initial_state, times)

plt.plot(times, [fidelity(state, initial_state) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [fidelity(state, target_state) for state in evolution.states], label="Overlap with target state")

plt.title("GOAT performance")
plt.xlabel("Time")
plt.legend()
plt.show()
```
The desired fidelity is not reached.


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
        "fid_err_targ": 0.01
    },
)

opt_time = res_goat_time.optimized_params[-1][0]
time_range = times < opt_time

print('Infidelity: ', res_goat_time.infidelity)
print('Optimized time: ', opt_time)

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
times2 = times[time_range]
if opt_time not in times2:
    times2 = np.append(times2, opt_time)
    
H_result = qt.QobjEvo([Hd, [Hc, np.array(res_goat_time.optimized_controls[0])]], tlist=times)
evolution_time = qt.mesolve(H_result, initial_state, times2)

plt.plot(times2, [fidelity(state, initial_state) for state in evolution_time.states], label="Overlap with initial state")
plt.plot(times2, [fidelity(state, target_state) for state in evolution_time.states], label="Overlap with target state")

plt.title('GOAT (optimized over time) performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```
The desired fidelity is still not reached.


### c) global optimization 

```python
res_goat_global = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.01
    },
    optimizer_kwargs={
       "method": "basinhopping",
       "max_iter": 1000
    }
)

global_time = res_goat_global.optimized_params[-1][0]
global_range = times < global_time

print('Infidelity: ', res_goat_global.infidelity)
print('Optimized time: ', global_time)

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
times3 = times[global_range]
if global_time not in times3:
    times3 = np.append(times3, global_time)
    
H_result = qt.QobjEvo([Hd, [Hc, np.array(res_goat_global.optimized_controls[0])]], tlist=times)
evolution_global = qt.mesolve(H_result, initial_state, times3)

plt.plot(times3, [fidelity(state, initial_state) for state in evolution_global.states], label="Overlap with initial state")
plt.plot(times3, [fidelity(state, target_state) for state in evolution_global.states], label="Overlap with target state")

plt.title('GOAT (global) performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```
## Comparison

```python
plt.plot(times, guess_pulse, label='initial guess')
plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], label='optimized (over time) pulse')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[0])[global_range], label='global optimized pulse')
plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```
```python
print('Guess Fidelity: ', fidelity(evolution_guess.states[-1], target_state))
print('GOAT Fidelity: ', 1 - res_goat.infidelity)
print('Time Fidelity: ', 1 - res_goat_time.infidelity)
print('GLobal Fidelity: ', 1 - res_goat_global.infidelity)

plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], 'k--', label="Guess")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], label="GOAT")
plt.plot(times2, [qt.fidelity(state, target_state) for state in evolution_time.states], label="Time")
plt.plot(times3, [qt.fidelity(state, target_state) for state in evolution_global.states], label="Global")

plt.title('GOAT Fidelities')
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Validation


```python
guess_fidelity = fidelity(evolution_guess.states[-1], target_state)

# target fidelity not reached in part a), check that it is better than the guess
assert 1 - res_goat.infidelity > guess_fidelity
assert np.allclose(fidelity(evolution.states[-1], target_state), 1 - res_goat.infidelity, atol=1e-3)

# target fidelity not reached in part b), check that it is better than part a)
assert res_goat_time.infidelity < res_goat.infidelity
assert np.allclose(fidelity(evolution_time.states[-1], target_state), 1 - res_goat_time.infidelity, atol=1e-3)

assert res_goat_global.infidelity < 0.01
assert np.allclose(fidelity(evolution_global.states[-1], target_state), 1 - res_goat_global.infidelity, atol=1e-3)
```

```python
qt.about()
```