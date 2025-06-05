# CRAB algorithm


```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, Qobj, liouvillian, qeye, sigmam, sigmax, sigmay, sigmaz, tensor)
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
c_ops = [np.sqrt(gamma) * sigmam()]

Hd = 1 / 2 * hbar * omega * sz
Hd = liouvillian(H=Hd, c_ops=c_ops)
Hc = [liouvillian(sx), liouvillian(sy), liouvillian(sz)]
H = [Hd, Hc[0], Hc[1], Hc[2]]

initial_gate = qeye(2)
target_gate = Qobj(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))

times = np.linspace(0, np.pi / 2, 250)
```

## Crab algorithm


```python
n_params = 6 # adjust in steps of 3
control_params = {
    "ctrl_x": {
        "guess": [1 for _ in range(n_params)],
        "bounds": [(-1, 1)] * n_params,
    },
    "ctrl_y": {
        "guess": [1 for _ in range(n_params)],
        "bounds": [(-1, 1)] * n_params,
    },
    "ctrl_z": {
        "guess": [1 for _ in range(n_params)],
        "bounds": [(-1, 1)] * n_params,
    },
}
alg_args = {"alg": "CRAB", "fid_err_targ": 0.01}


res_crab = optimize_pulses(
    objectives=Objective(initial_gate, H, target_gate),
    control_parameters=control_params,
    tlist=times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_crab.infidelity)

plt.plot(times, res_crab.optimized_controls[0], label='optimized pulse sx')
plt.plot(times, res_crab.optimized_controls[1], label='optimized pulse sy')
plt.plot(times, res_crab.optimized_controls[2], label='optimized pulse sz')
plt.title('CRAB pulse')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

    c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_basinhopping.py:294: RuntimeWarning: Method Nelder-Mead does not use gradient information (jac).
      return self.minimizer(self.func, x0, **self.kwargs)
    c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_basinhopping.py:294: OptimizeWarning: Unknown solver options: gtol
      return self.minimizer(self.func, x0, **self.kwargs)
    

    Infidelity:  0.007575498222395387
    


    
![png](CRAB_gate_open_files/CRAB_gate_open_5_2.png)
    



```python
H_result = [Hd,
            [Hc[0], res_crab.optimized_controls[0]],
            [Hc[1], res_crab.optimized_controls[1]],
            [Hc[2], res_crab.optimized_controls[2]]]

identity_op = qt.qeye(2)
identity_super = qt.spre(identity_op)

evolution = qt.mesolve(H_result, identity_super, times)

target_super = qt.to_super(target_gate)
initial_super = qt.to_super(initial_gate)

initial_overlaps = [np.abs((prop.dag() * initial_super).tr()) / (prop.norm() ) for prop in evolution.states]
target_overlaps = [np.abs((prop.dag() * target_super).tr()) / (prop.norm() ) for prop in evolution.states]

plt.plot(times, initial_overlaps, label="Overlap with initial gate")
plt.plot(times, target_overlaps, label="Overlap with target gate")
plt.title("CRAB performance")
plt.xlabel("Time")
plt.legend()
plt.show()

```


    
![png](CRAB_gate_open_files/CRAB_gate_open_6_0.png)
    


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
    


