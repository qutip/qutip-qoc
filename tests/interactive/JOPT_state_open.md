# JOPT algorithm for a 2 level system



```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, fidelity, liouvillian, ket2dm, Qobj, basis, sigmam)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
from jax import jit, numpy
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

    Fidelity:  0.7628226855907076
    


    
![png](JOPT_state_open_files/JOPT_state_open_5_1.png)
    


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

    c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_basinhopping.py:294: RuntimeWarning: Method Nelder-Mead does not use gradient information (jac).
      return self.minimizer(self.func, x0, **self.kwargs)
    c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_basinhopping.py:294: OptimizeWarning: Unknown solver options: gtol
      return self.minimizer(self.func, x0, **self.kwargs)
    c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\equinox\_jit.py:55: UserWarning: Complex dtype support in Diffrax is a work in progress and may not yet produce correct results. Consider splitting your computation into real and imaginary parts instead.
      out = fun(*args, **kwargs)
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[17], line 9
          1 ctrl_parameters = {
          2     id: {"guess": [1, 0], "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
          3 }
          4 alg_args = {
          5     "alg": "JOPT",
          6     "fid_err_targ": 0.01,
          7 }
    ----> 9 res_jopt = optimize_pulses(
         10     objectives=Objective(initial_state, H, target_state),
         11     control_parameters=ctrl_parameters,
         12     tlist=times,
         13     minimizer_kwargs={
         14         "method": "Nelder-Mead",
         15     },
         16     algorithm_kwargs=alg_args,
         17 )
         19 print('Infidelity: ', res_jopt.infidelity)
         21 plt.plot(times, sin_x(times, jopt_guess), label='initial guess')
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\qutip_qoc\pulse_optim.py:422, in optimize_pulses(objectives, control_parameters, tlist, algorithm_kwargs, optimizer_kwargs, minimizer_kwargs, integrator_kwargs, optimization_type)
        419     rl_env.train()
        420     return rl_env.result()
    --> 422 return _global_local_optimization(
        423     objectives,
        424     control_parameters,
        425     time_interval,
        426     time_options,
        427     algorithm_kwargs,
        428     optimizer_kwargs,
        429     minimizer_kwargs,
        430     integrator_kwargs,
        431     qtrl_optimizers,
        432 )
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\qutip_qoc\_optimizer.py:359, in _global_local_optimization(objectives, control_parameters, time_interval, time_options, algorithm_kwargs, optimizer_kwargs, minimizer_kwargs, integrator_kwargs, qtrl_optimizers)
        356 cllbck = _Callback(result, fid_err_targ, max_wall_time, bounds, disp)
        358 # run the optimization
    --> 359 min_res = optimizer(
        360     func=multi_objective.goal_fun,
        361     minimizer_kwargs={
        362         "jac": multi_objective.grad_fun,
        363         "callback": cllbck.min_callback,
        364         **minimizer_kwargs,
        365     },
        366     callback=cllbck.opt_callback,
        367     **optimizer_kwargs,
        368 )
        370 cllbck.stop_clock()  # stop the clock
        372 # some global optimization methods do not return the minimum result
        373 # when terminated through StopIteration (see min_callback)
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\_lib\_util.py:440, in _transition_to_rng.<locals>.decorator.<locals>.wrapper(*args, **kwargs)
        433     message = (
        434         "The NumPy global RNG was seeded by calling "
        435         f"`np.random.seed`. Beginning in {end_version}, this "
        436         "function will no longer use the global RNG."
        437     ) + cmn_msg
        438     warnings.warn(message, FutureWarning, stacklevel=2)
    --> 440 return fun(*args, **kwargs)
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_basinhopping.py:696, in basinhopping(func, x0, niter, T, stepsize, minimizer_kwargs, take_step, accept_test, callback, interval, disp, niter_success, rng, target_accept_rate, stepwise_factor)
        693 if niter_success is None:
        694     niter_success = niter + 2
    --> 696 bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
        697                         accept_tests, disp=disp)
        699 # The wrapped minimizer is called once during construction of
        700 # BasinHoppingRunner, so run the callback
        701 if callable(callback):
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_basinhopping.py:78, in BasinHoppingRunner.__init__(self, x0, minimizer, step_taking, accept_tests, disp)
         75 self.res.minimization_failures = 0
         77 # do initial minimization
    ---> 78 minres = minimizer(self.x)
         79 if not minres.success:
         80     self.res.minimization_failures += 1
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_basinhopping.py:294, in MinimizerWrapper.__call__(self, x0)
        292     return self.minimizer(x0, **self.kwargs)
        293 else:
    --> 294     return self.minimizer(self.func, x0, **self.kwargs)
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_minimize.py:726, in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        723 callback = _wrap_callback(callback, meth)
        725 if meth == 'nelder-mead':
    --> 726     res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
        727                                **options)
        728 elif meth == 'powell':
        729     res = _minimize_powell(fun, x0, args, callback, bounds, **options)
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_optimize.py:833, in _minimize_neldermead(func, x0, args, callback, maxiter, maxfev, disp, return_all, initial_simplex, xatol, fatol, adaptive, bounds, **unknown_options)
        831 try:
        832     for k in range(N + 1):
    --> 833         fsim[k] = func(sim[k])
        834 except _MaxFuncCallError:
        835     pass
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\scipy\optimize\_optimize.py:542, in _wrap_scalar_function_maxfun_validation.<locals>.function_wrapper(x, *wrapper_args)
        540 ncalls[0] += 1
        541 # A copy of x is sent to the user function (gh13740)
    --> 542 fx = function(np.copy(x), *(wrapper_args + args))
        543 # Ideally, we'd like to a have a true scalar returned from f(x). For
        544 # backwards-compatibility, also allow np.array([1.3]),
        545 # np.array([[1.3]]) etc.
        546 if not np.isscalar(fx):
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\qutip_qoc\objective.py:143, in _MultiObjective.goal_fun(self, params)
        141 infid = 0
        142 for i, alg in enumerate(self._alg_list):
    --> 143     infid += self._weights[i] * alg.infidelity(params)
        144 return infid
    

        [... skipping hidden 14 frame]
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\qutip_qoc\_jopt.py:153, in _JOPT._infid(self, params)
        151 diff = X - self._target
        152 # to prevent if/else in qobj.dag() and qobj.tr()
    --> 153 diff_dag = Qobj(diff.data.adjoint(), dims=diff.dims)
        154 g = 1 / 2 * (diff_dag * diff).data.trace()
        155 infid = jnp.real(self._norm_fac * g)
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\qutip\core\qobj.py:279, in Qobj.__init__(self, arg, dims, copy, superrep, isherm, isunitary)
        277 self._isherm = isherm
        278 self._isunitary = isunitary
    --> 279 self._initialize_data(arg, dims, copy)
        281 if superrep is not None:
        282     self.superrep = superrep
    

    File c:\Users\julia\miniforge3\envs\qutip-dev\Lib\site-packages\qutip\core\qobj.py:265, in Qobj._initialize_data(self, arg, dims, copy)
        261     self._dims = Dimensions(
        262         dims or [[self._data.shape[0]], [self._data.shape[1]]]
        263     )
        264 if self._dims.shape != self._data.shape:
    --> 265     raise ValueError('Provided dimensions do not match the data: ' +
        266                      f"{self._dims.shape} vs {self._data.shape}")
    

    ValueError: Provided dimensions do not match the data: (4, 1) vs (1, 4)



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


    
![png](JOPT_state_open_files/JOPT_state_open_10_0.png)
    


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

    Infidelity:  0.000510596900193172
    optimized time:  [6.283185307179586]
    


    
![png](JOPT_state_open_files/JOPT_state_open_12_1.png)
    



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


    
![png](JOPT_state_open_files/JOPT_state_open_13_0.png)
    


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

    Infidelity:  6.418314632516964e-05
    optimized time:  [2.990647906220658]
    


    
![png](JOPT_state_open_files/JOPT_state_open_15_1.png)
    



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


    
![png](JOPT_state_open_files/JOPT_state_open_16_0.png)
    


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


    
![png](JOPT_state_open_files/JOPT_state_open_18_0.png)
    



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

    Guess Infidelity:  0.7568009078458054
    JOPT Infidelity:  0.0003454363645383207
    Time Infidelity:  0.000510596900193172
    GLobal Infidelity:  6.418314632516964e-05
    


    
![png](JOPT_state_open_files/JOPT_state_open_19_1.png)
    


## Validation


```python
assert res_jopt.infidelity < 0.001
assert res_jopt_time.infidelity < 0.001
assert res_jopt_global.infidelity < 0.001
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
