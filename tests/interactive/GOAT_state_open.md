# GOAT algorithm for a 2 level system


```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (fidelity, liouvillian, ket2dm, Qobj, basis, sigmam)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
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
goat_guess = [1, 0.5]
guess_pulse = goat_guess[0] * np.sin(goat_guess[1] * times)

H_result_guess = [Hd, [Hc, guess_pulse]]
evolution_guess = qt.mesolve(H_result_guess, initial_state, times, c_ops)

print('Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_guess.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], '--', label="Fidelity")
plt.legend()
plt.title("Guess performance")
plt.xlabel("Time")
plt.show()
```

    Fidelity:  0.8176293626457246
    


    
![png](GOAT_state_open_files/GOAT_state_open_5_1.png)
    


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
alg_args = {
    "alg": "GOAT",
    "fid_err_targ": 0.01,
}
ctrl_params = {
    id: {"guess": goat_guess, "bounds": [(-10, 10), (0, 2*np.pi)]}  # c0 and c1
}

# run the optimization
res_goat = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = ctrl_params,
    tlist = times,
    algorithm_kwargs = alg_args,
)

print('Infidelity: ', res_goat.infidelity)

plt.plot(times, sin(times, goat_guess), 'k--',  label='initial guess')
plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

    Infidelity:  0.07178736737494786
    


    
![png](GOAT_state_open_files/GOAT_state_open_9_1.png)
    



```python
H_result = [Hd] + [[Hc, np.array(res_goat.optimized_controls[0])]]
evolution = qt.mesolve(H_result, initial_state, times, c_ops)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")
plt.title("GOAT performance")
plt.xlabel("Time")
plt.ylim(0, 1)
plt.legend()
plt.show()
```


    
![png](GOAT_state_open_files/GOAT_state_open_10_0.png)
    


### b) optimized over time


```python
# treats time as optimization variable
ctrl_params["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}

# run the optimization
res_goat_time = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = ctrl_params,
    tlist = times,
    algorithm_kwargs = alg_args,
)

opt_time = res_goat_time.optimized_params[-1]
time_range = times < opt_time

print('Infidelity: ', res_goat_time.infidelity)
print('optimized time: ', res_goat_time.optimized_params[-1])

plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], label='optimized (over time) pulse')
plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show() 
```

    Infidelity:  0.009074574820662817
    optimized time:  [0.37202273025971855]
    


    
![png](GOAT_state_open_files/GOAT_state_open_12_1.png)
    



```python
times2 = times[time_range]
if opt_time not in times2:
    times2 = np.append(times2, opt_time)
    
H_result_time = [Hd] + [[Hc, np.array(res_goat_time.optimized_controls[0])]]
evolution_time = qt.mesolve(H_result_time, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_time.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_time.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_time.states], '--', label="Fidelity")
plt.title('GOAT performance (optimized over time)')
plt.xlabel('Time')
plt.xlim(0, res_goat_time.optimized_params[-1][0])
plt.legend()
plt.show()
```


    
![png](GOAT_state_open_files/GOAT_state_open_13_0.png)
    


### Global optimization 


```python
alg_args = {
    "alg": "GOAT",
    "fid_err_targ": 0.01,
}
opt_args = {
    "method": "basinhopping",
    "max_iter": 100,
}

res_goat_global = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = ctrl_params,
    tlist = times,
    algorithm_kwargs = alg_args,
    optimizer_kwargs = opt_args
)

global_time = res_goat_global.optimized_params[-1]
global_range = times < global_time

print('Infidelity: ', res_goat_global.infidelity)
print('optimized time: ', res_goat_global.optimized_params[-1])

plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[0])[global_range], label='global optimized pulse')
plt.title('GOAT pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()

```

    Infidelity:  0.007905283658334122
    optimized time:  [1.5444538341679324]
    


    
![png](GOAT_state_open_files/GOAT_state_open_15_1.png)
    



```python
times3 = times[global_range]
if global_time not in times3:
    times3 = np.append(times3, global_time)
    
H_result_global = [Hd, [Hc, np.array(res_goat_global.optimized_controls[0])]]
evolution_global = qt.mesolve(H_result_global, initial_state, times)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_global.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_global.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_global.states], '--', label="Fidelity")
plt.xlim(0, res_goat_global.optimized_params[-1][0])

plt.title('GOAT performance (global)')
plt.xlabel('Time')
plt.legend()
plt.show()
```


    
![png](GOAT_state_open_files/GOAT_state_open_16_0.png)
    


## Comparison


```python
plt.plot(times, guess_pulse, 'k--', label='initial guess')
plt.plot(times, res_goat.optimized_controls[0], label='optimized pulse')
plt.plot(times[time_range], np.array(res_goat_time.optimized_controls[0])[time_range], label='optimized (over time) pulse')
plt.plot(times[global_range], np.array(res_goat_global.optimized_controls[0])[global_range], label='global optimized pulse')
plt.title('GOAT pulses')
plt.xlabel('time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```


    
![png](GOAT_state_open_files/GOAT_state_open_18_0.png)
    



```python
print('Guess Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))
print('GOAT Infidelity: ', res_goat.infidelity)
print('Time Infidelity: ', res_goat_time.infidelity)
print('GLobal Infidelity: ', res_goat_global.infidelity)

plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], 'k--', label="Guess")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], label="Goat")
plt.plot(times[time_range], [qt.fidelity(state, target_state) for state in evolution_time.states[:len(times[time_range])]], label="Time")
plt.plot(times[global_range], [qt.fidelity(state, target_state) for state in evolution_global.states[:len(times[global_range])]], label="Global")

plt.title('Fidelities')
plt.xlabel('Time')
plt.legend()
plt.show()

```

    Guess Fidelity:  0.8176293626457246
    GOAT Infidelity:  0.07178736737494786
    Time Infidelity:  0.009074574820662817
    GLobal Infidelity:  0.007905283658334122
    


    
![png](GOAT_state_open_files/GOAT_state_open_19_1.png)
    


## Validation


```python
assert res_goat.infidelity < 0.1
assert res_goat_time.infidelity < 0.01
assert res_goat_global.infidelity < 0.01
```


```python
qt.about()
```

    
    QuTiP: Quantum Toolbox in Python
    ================================
    Copyright (c) QuTiP team 2011 and later.
    Current admin team: Alexander Pitchford, Nathan Shammah, Shahnawaz Ahmed, Neill Lambert, Eric Giguère, Boxi Li, Simon Cross, Asier Galicia, Paul Menczel, and Patrick Hopf.
    Board members: Daniel Burgarth, Robert Johansson, Anton F. Kockum, Franco Nori and Will Zeng.
    Original developers: R. J. Johansson & P. D. Nation.
    Previous lead developers: Chris Granade & A. Grimsmo.
    Currently developed through wide collaboration. See https://github.com/qutip for details.
    
    QuTiP Version:      5.1.1
    Numpy Version:      1.26.4
    Scipy Version:      1.15.2
    Cython Version:     None
    Matplotlib Version: 3.10.0
    Python Version:     3.12.10
    Number of CPUs:     8
    BLAS Info:          Generic
    INTEL MKL Ext:      None
    Platform Info:      Windows (AMD64)
    Installation path:  c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\qutip
    
    Installed QuTiP family packages
    -------------------------------
    
    qutip-jax: 0.1.0
    qutip-qtrl: 0.1.5
    
    ================================================================================
    Please cite QuTiP in your publication.
    ================================================================================
    For your convenience a bibtex reference can be easily generated using `qutip.cite()`
    


```python

```
