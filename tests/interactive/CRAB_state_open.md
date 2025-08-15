# GRAPE algorithm for 2 level system


```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, fidelity, liouvillian, ket2dm, Qobj, basis, sigmam)
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

## CRAB algorithm


```python
n_params = 6 # adjust in steps of 3
control_params = {
    "ctrl_x": {
        "guess": [1 for _ in range(n_params)],
        "bounds": [(-1, 1)] * n_params,
    },
}
alg_args = {"alg": "CRAB", "fid_err_targ": 0.001, "fix_frequency": False} 

res_crab = optimize_pulses(
    objectives=Objective(initial_state, H, target_state),
    control_parameters=control_params,
    tlist=times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_crab.infidelity)

plt.plot(times, res_crab.optimized_controls[0], label='optimized pulse')
plt.title('CRAB pulse')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

    Infidelity:  0.0027237139157979926
    


    
![png](CRAB_state_open_files/CRAB_state_open_5_1.png)
    



```python
H_result = [Hd, [Hc, res_crab.optimized_controls[0]]]
evolution = qt.mesolve(H_result, initial_state, times, c_ops)

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")
plt.title("CRAB performance")
plt.xlabel('Time')
plt.legend()
plt.show()
```


    
![png](CRAB_state_open_files/CRAB_state_open_6_0.png)
    


## Validation


```python
assert res_crab.infidelity < 0.01
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
