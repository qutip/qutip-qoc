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

# GOAT algorithm for an open system (gate synthesis)

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import gates, qeye, liouvillian, sigmam, sigmax, sigmay, sigmaz
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

def fidelity(gate_super, target_super):
    gate_oper = qt.Qobj(gate_super.data)
    target_oper = qt.Qobj(target_super.data)
    
    return np.abs(gate_oper.overlap(target_oper) / target_oper.norm())
```

## Problem setup


```python
omega = 0.1  # energy splitting
gamma = 0.1  # amplitude damping
sx, sy, sz = sigmax(), sigmay(), sigmaz()
c_ops = [np.sqrt(gamma) * sigmam()]

Hd = 1 / 2 * omega * sz
Hd = liouvillian(H=Hd, c_ops=c_ops)
Hc = [liouvillian(sx), liouvillian(sy), liouvillian(sz)]
H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial_gate = qeye(2)
target_gate = gates.hadamard_transform()

times = np.linspace(0, np.pi / 2, 250)
```

## Guess


```python
goat_guess = [1, 1]
guess_pulse = goat_guess[0] * np.sin(goat_guess[1] * times)

initial_super = qt.to_super(initial_gate)
target_super = qt.to_super(target_gate)

H_guess = [Hd] + [[hc, guess_pulse] for hc in Hc]
evolution_guess = qt.mesolve(H_guess, initial_super, times)

plt.plot(times, [fidelity(gate, initial_super) for gate in evolution_guess.states], label="Overlap with initial gate")
plt.plot(times, [fidelity(gate, target_super) for gate in evolution_guess.states], label="Overlap with target gate")
plt.title("Guess performance")
plt.xlabel('Time')
plt.legend()
plt.show()
```
## GOAT algorithm
### a) not optimized over time

```python
# control function
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
```


```python
control_params = {
    id: {"guess": goat_guess, "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
    for id in ['x', 'y', 'z']
}

# run the optimization
res_goat = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.001,
    },
)

print('Infidelity: ', res_goat.infidelity)

plt.plot(times, res_goat.optimized_controls[0], 'b', label='optimized pulse sx')
plt.plot(times, res_goat.optimized_controls[1], 'g', label='optimized pulse sy')
plt.plot(times, res_goat.optimized_controls[2], 'r', label='optimized pulse sz')
plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd,
            [Hc[0], np.array(res_goat.optimized_controls[0])],
            [Hc[1], np.array(res_goat.optimized_controls[1])],
            [Hc[2], np.array(res_goat.optimized_controls[2])]]
evolution = qt.mesolve(H_result, initial_super, times)

plt.plot(times, [fidelity(gate, initial_super) for gate in evolution.states], label="Overlap with initial gate")
plt.plot(times, [fidelity(gate, target_super) for gate in evolution.states], label="Overlap with target gate")

plt.title("GOAT performance")
plt.xlabel("Time")
plt.legend()
plt.show()
```
### b) optimized over time


```python
# treats time as optimization variable
control_params["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}

# run the optimization
res_goat_time = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.001,
    },
)

opt_time = res_goat_time.optimized_params[-1][0]
time_range = times < opt_time

print('Infidelity: ', res_goat_time.infidelity)
print('Optimized time: ', opt_time)

plt.plot(times, guess_pulse, 'k--', label='guess pulse sx, sy, sz')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], 'b', label='optimized pulse sx')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[1])[time_range], 'g', label='optimized pulse sy')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[2])[time_range], 'r', label='optimized pulse sz')
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

H_result = qt.QobjEvo(
    [Hd, [Hc[0], np.array(res_goat_time.optimized_controls[0])],
         [Hc[1], np.array(res_goat_time.optimized_controls[1])],
         [Hc[2], np.array(res_goat_time.optimized_controls[2])]], tlist=times)
evolution_time = qt.mesolve(H_result, initial_super, times2)

plt.plot(times2, [fidelity(gate, initial_super) for gate in evolution_time.states], label="Overlap with initial gate")
plt.plot(times2, [fidelity(gate, target_super) for gate in evolution_time.states], label="Overlap with target gate")

plt.title('GOAT (optimized over time) performance')
plt.xlabel("Time")
plt.legend()
plt.show()
```
### c) global optimization 

```python
res_goat_global = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GOAT",
        "fid_err_targ": 0.001,
    },
    optimizer_kwargs = {
       "method": "basinhopping",
       "max_iter": 100,
    }
)

global_time = res_goat_global.optimized_params[-1][0]
global_range = times < global_time

print('Infidelity: ', res_goat_global.infidelity)
print('Optimized time: ', global_time)

plt.plot(times, guess_pulse, 'k--', label='guess pulse sx, sy, sz')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[0])[global_range], 'b', label='optimized pulse sx')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[1])[global_range], 'g', label='optimized pulse sy')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[2])[global_range], 'r', label='optimized pulse sz')
plt.title('GOAT pulses (global optimization)')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
times3 = times[global_range]
if global_time not in times3:
    times3 = np.append(times3, global_time)

H_result = qt.QobjEvo(
    [Hd, [Hc[0], np.array(res_goat_global.optimized_controls[0])],
         [Hc[1], np.array(res_goat_global.optimized_controls[1])], 
         [Hc[2], np.array(res_goat_global.optimized_controls[2])]], tlist=times)
evolution_global = qt.mesolve(H_result, initial_super, times3)

plt.plot(times3, [fidelity(gate, initial_super) for gate in evolution_global.states], label="Overlap with initial gate")
plt.plot(times3, [fidelity(gate, target_super) for gate in evolution_global.states], label="Overlap with target gate")

plt.title('GOAT (global optimization) performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```
## Comparison


```python
fig, axes = plt.subplots(1, 3, figsize=(18, 4))  # 1 row, 3 columns

titles = ["GOAT sx pulses", "GOAT sy pulses", "GOAT sz pulses"]

for i, ax in enumerate(axes):
    ax.plot(times, guess_pulse, label='initial guess')
    ax.plot(times, res_goat.optimized_controls[i], label='optimized pulse')
    ax.plot(times[time_range], np.array(res_goat_time.optimized_controls[i])[time_range], label='optimized (over time) pulse')
    ax.plot(times[global_range], np.array(res_goat_global.optimized_controls[i])[global_range], label='global optimized pulse')
    ax.set_title(titles[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Pulse amplitude')
    ax.legend()

plt.tight_layout()
plt.show()
```
## Validation


```python
guess_fidelity = fidelity(evolution_guess.states[-1], target_super)

# target fidelity not reached in part a), check that it is better than the guess
assert 1 - res_goat.infidelity >= guess_fidelity
assert np.allclose(fidelity(evolution.states[-1], target_super), 1 - res_goat.infidelity, atol=1e-3)

# target fidelity not reached in part b), check that it is better than part a)
assert res_goat_time.infidelity <= res_goat.infidelity
assert np.allclose(fidelity(evolution_time.states[-1], target_super), 1 - res_goat_time.infidelity, atol=1e-3)

assert res_goat_global.infidelity <= res_goat_time.infidelity
assert np.allclose(fidelity(evolution_global.states[-1], target_super), 1 - res_goat_global.infidelity, atol=1e-3)
```


```python
qt.about()
```