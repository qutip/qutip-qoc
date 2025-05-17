# JOPT algorithm


```python
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, numpy
from qutip import (about, Qobj, gates, liouvillian, qeye, sigmam, sigmax, sigmay, sigmaz, fidelity)
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
jopt_guess = [1, 1]
guess_pulse = jopt_guess[0] * np.sin(jopt_guess[1] * times)
```


```python
H_result_guess = [Hd,
            [Hc[0], guess_pulse],
            [Hc[1], guess_pulse],
            [Hc[2], guess_pulse]]

identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution_guess = qt.mesolve(H_result_guess, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps_guess = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution_guess.states]
target_overlaps_guess = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution_guess.states]

plt.plot(times, initial_overlaps_guess, label="Overlap with initial gate")
plt.plot(times, target_overlaps_guess, label="Overlap with target gate")
plt.title("Guess performance")
plt.xlabel("Time")
plt.legend()
plt.show()

```

## JOPT algorithm


```python
@jit
def sin_x(t, c, **kwargs):
    return c[0] * numpy.sin(c[1] * t)

H = [Hd] + [[hc, sin_x, {"grad": sin_x}] for hc in Hc]
```


    The Kernel crashed while executing code in the current cell or a previous cell. 
    

    Please review the code in the cell(s) to identify a possible cause of the failure. 
    

    Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 
    

    View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.


### a) not optimized over time


```python
ctrl_parameters = {
    id: {"guess": jopt_guess, "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1 
    for id in ['x', 'y', 'z']
}
alg_args = {
    "alg": "JOPT",
    "fid_err_targ": 0.01,
}

res_jopt = optimize_pulses(
    objectives=Objective(initial_gate, H, target_gate),
    control_parameters=ctrl_parameters,
    tlist=times,
    minimizer_kwargs={
        "method": "Nelder-Mead",
    },
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_jopt.infidelity)

plt.plot(times, res_jopt.optimized_controls[0], label='optimized pulse')
plt.plot(times, res_jopt.optimized_controls[1], label='optimized pulse')
plt.plot(times, res_jopt.optimized_controls[2], label='optimized pulse')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```


```python
H_result = [Hd,
            [Hc[0], np.array(res_jopt.optimized_controls[0])],
            [Hc[1], np.array(res_jopt.optimized_controls[1])],
            [Hc[2], np.array(res_jopt.optimized_controls[2])]]

identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution = qt.mesolve(H_result, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution.states]
target_overlaps = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution.states]

plt.plot(times, initial_overlaps, label="Overlap with initial gate")
plt.plot(times, target_overlaps, label="Overlap with target gate")
plt.title("JOPT performance")
plt.xlabel("Time")
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
    "fid_err_targ": 0.01,
}

# run the optimization
res_jopt_time = optimize_pulses(
    objectives=Objective(initial_gate, H, target_gate),
    control_parameters=ctrl_parameters,
    tlist=times,
    minimizer_kwargs={
        "method": "Nelder-Mead",
    },
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_jopt_time.infidelity)
print('Time: ', times[-1])
print('optimized time: ', res_jopt_time.optimized_params[-1])

time_range = times < res_jopt_time.optimized_params[-1]

plt.plot(times[time_range], np.array(res_jopt_time.optimized_controls[0])[time_range], label='Optimized (over time) pulse sx')
plt.plot(times[time_range], np.array(res_jopt_time.optimized_controls[1])[time_range], label='Optimized (over time) pulse sy')
plt.plot(times[time_range], np.array(res_jopt_time.optimized_controls[2])[time_range], label='Optimized (over time) pulse sz')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```


```python
H_result_time = [Hd,
            [Hc[0], np.array(res_jopt_time.optimized_controls[0])],
            [Hc[1], np.array(res_jopt_time.optimized_controls[1])],
            [Hc[2], np.array(res_jopt_time.optimized_controls[2])]]


identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution_time = qt.mesolve(H_result_time, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps_time = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution_time.states]
target_overlaps_time = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution_time.states]

plt.plot(times, initial_overlaps_time, label="Overlap with initial gate")
plt.plot(times, target_overlaps_time, label="Overlap with target gate")
plt.title("JOPT performance (optimized over time)")
plt.xlabel("Time")
plt.legend()
plt.show()

```

## Global optimization


```python
alg_args = {
    "alg": "JOPT",
    "fid_err_targ": 0.001,
}
opt_args = {
    "method": "basinhopping",
    "max_iter": 1000,
}

res_jopt_global = optimize_pulses(
    objectives=Objective(initial_gate, H, target_gate),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs=alg_args,
    optimizer_kwargs=opt_args
)

print('Infidelity: ', res_jopt_global.infidelity)
print('optimized time: ', res_jopt_global.optimized_params[-1])

global_range = times < res_jopt_global.optimized_params[-1]

plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[0])[global_range], label='Global optimized pulse sx')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[1])[global_range], label='Global optimized pulse sy')
plt.plot(times[global_range], np.array(res_jopt_global.optimized_controls[2])[global_range], label='Global optimized pulse sz')
plt.title('JOPT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()

```


```python
H_result_global = [Hd,
            [Hc[0], np.array(res_jopt_time.optimized_controls[0])],
            [Hc[1], np.array(res_jopt_time.optimized_controls[1])],
            [Hc[2], np.array(res_jopt_time.optimized_controls[2])]]


identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution_global = qt.mesolve(H_result, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps_global = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution_global.states]
target_overlaps_global = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution_global.states]

plt.plot(times, initial_overlaps_global, label="Overlap with initial gate")
plt.plot(times, target_overlaps_global, label="Overlap with target gate")
plt.title("JOPT performance (global)")
plt.xlabel("Time")
plt.legend()
plt.show()

```

## Comparison


```python
fig, axes = plt.subplots(1, 3, figsize=(18, 4))  # 1 row, 3 columns

titles = ["JOPT s_x pulses", "JOPT s_y pulses", "JOPT s_z pulses"]

for i in range(3):
    ax = axes[i]
    ax.plot(times, sin_x(times, jopt_guess), label='Initial guess')
    ax.plot(times, res_jopt.optimized_controls[i], label='Optimized pulse')
    ax.plot(times[time_range], np.array(res_jopt_time.optimized_controls[i])[time_range], label='Optimized (over time) pulse')
    ax.plot(times[global_range], np.array(res_jopt_global.optimized_controls[i])[global_range], label='Global optimized pulse')
    ax.set_title(titles[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Pulse amplitude')
    ax.legend()

plt.tight_layout()
plt.show()

```

## Validation


```python
assert res_jopt.infidelity < 0.001
assert res_jopt_time.infidelity < 0.001
assert res_jopt_global.infidelity < 0.001
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 assert res_jopt.infidelity < 0.001
          2 assert res_jopt_time.infidelity < 0.001
          3 assert res_jopt_global.infidelity < 0.001
    

    NameError: name 'res_jopt' is not defined



```python
qt.about()
```
