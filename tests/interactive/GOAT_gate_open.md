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

# GOAT algorithm

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
Hd = liouvillian(H=Hd, c_ops=[np.sqrt(gamma) * sigmam()])
Hc = [liouvillian(sx), liouvillian(sy), liouvillian(sz)]

H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial_gate = qeye(2)
target_gate = Qobj(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))

times = np.linspace(0, np.pi / 2, 250)
```

## Guess

```python
goat_guess = [1, 1]
guess_pulse = goat_guess[0] * np.sin(goat_guess[1] * times)
```

## Goat algorithm
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
ctrl_params = {
    id: {"guess": goat_guess, "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1 
    for id in ['x', 'y', 'z']
}
alg_args = {
    "alg": "GOAT",
    "fid_err_targ": 0.001,
}

# run the optimization
res_goat = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = ctrl_params,
    tlist = times,
    algorithm_kwargs = alg_args,
)

print('Infidelity: ', res_goat.infidelity)

plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times, res_goat.optimized_controls[1], label='optimized pulse')
plt.plot(times, res_goat.optimized_controls[2], label='optimized pulse')
plt.title('GOAT pulse')
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


identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution = qt.mesolve(H_result, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution.states]
target_overlaps = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution.states]

plt.plot(times, initial_overlaps, label="Overlap with initial gate")
plt.plot(times, target_overlaps, label="Overlap with target gate")
plt.title("GOAT performance")
plt.xlabel("global")
plt.legend()
plt.show()

```

### b) optimized over global

```python
# treats time as optimization variable
ctrl_params["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}
alg_args = {
    "alg": "GOAT",
    "fid_err_targ": 0.001,
}

# run the optimization
res_goat_time = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = ctrl_params,
    tlist = times,
    algorithm_kwargs = alg_args,
)

time_time = res_goat_time.optimized_params[-1]
time_range = times < time_time

print('Infidelity: ', res_goat_time.infidelity)
print("Time: ", times[-1])
print('Optimized time: ', res_goat_time.optimized_params[-1][0])

plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], label='optimized (over time) pulse sx')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[1])[time_range], label='optimized (over time) pulse sy')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[2])[time_range], label='optimized (over time) pulse sz')
plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd,
            [Hc[0], np.array(res_goat_time.optimized_controls[0])],
            [Hc[1], np.array(res_goat_time.optimized_controls[1])],
            [Hc[2], np.array(res_goat_time.optimized_controls[2])]]


identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution_time = qt.mesolve(H_result, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution_time.states]
target_overlaps = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution_time.states]

plt.plot(times, initial_overlaps, label="Overlap with initial gate")
plt.plot(times, target_overlaps, label="Overlap with target gate")
plt.title("GOAT performance (optimized over time)")
plt.xlabel("Time")
plt.legend()
plt.show()

```

## Global optimization

```python
alg_args = {
    "alg": "GOAT",
    "fid_err_targ": 0.1,
}
opt_args = {
    "method": "basinhopping",
    "max_iter": 100,
}

res_goat_global = optimize_pulses(
    objectives=Objective(initial_gate, H, target_gate),
    control_parameters = ctrl_params,
    tlist = times,
    algorithm_kwargs = alg_args,
    optimizer_kwargs = opt_args
)

global_time = res_goat_global.optimized_params[-1]
global_range = times < global_time

print('Infidelity: ', res_goat_global.infidelity)
print('optimized time: ', res_goat_global.optimized_params[-1])

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
H_result = [Hd,
            [Hc[0], np.array(res_goat_global.optimized_controls[0])],
            [Hc[1], np.array(res_goat_global.optimized_controls[1])],
            [Hc[2], np.array(res_goat_global.optimized_controls[2])]]


identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution_global = qt.mesolve(H_result, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution_global.states]
target_overlaps = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution_global.states]

plt.plot(times, initial_overlaps, label="Overlap with initial gate")
plt.plot(times, target_overlaps, label="Overlap with target gate")
plt.title("GOAT (global) performance")
plt.xlabel("Time")
plt.legend()
plt.show()

```

## Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 4))  # 1 row, 3 columns

titles = ["GOAT s_x pulses", "GOAT s_y pulses", "GOAT s_z pulses"]

for i in range(3):
    ax = axes[i]
    ax.plot(times, sin(times, goat_guess), 'k--', label='initial guess')
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
assert res_goat.infidelity < 0.1
assert res_goat_time.infidelity < 0.1
assert res_goat_global.infidelity < 0.01
```

```python
qt.about()
```


