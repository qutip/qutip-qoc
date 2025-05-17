# GRAPE algorithm for 2 level system


```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, fidelity, liouvillian, ket2dm, Qobj, basis, sigmam)
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

import logging
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
grape_guess = np.sin(times)

H_result_guess = [Hd, [Hc, grape_guess]]
evolution_guess = qt.mesolve(H_result_guess, initial_state, times)

print('Fidelity: ', qt.fidelity(evolution_guess.states[-1], target_state))

plt.plot(times, [np.abs(state.overlap(initial_state)) for state in evolution_guess.states], label="Overlap with initial state")
plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution_guess.states], label="Overlap with target state")
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution_guess.states], '--', label="Fidelity")
plt.legend()
plt.title("Guess performance")
plt.xlabel("Time")
plt.show()
```

    Fidelity:  0.4768516474214033
    


    
![png](GRAPE_state_open_files/GRAPE_state_open_5_1.png)
    


## GRAPE algorithm


```python
alg_args = {"alg": "GRAPE", "fid_err_targ": 0.001, "log_level": logging.DEBUG - 2}
control_params = {
    "ctrl_1": {"guess": grape_guess, "bounds": [-1, 1]},  # Control pulse for Hc1
}

res_grape = optimize_pulses(
    objectives = Objective(initial_state, H, target_state),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs=alg_args,
)

print('Infidelity: ', res_grape.infidelity)

plt.plot(times, grape_guess, label='initial guess')
plt.plot(times, res_grape.optimized_controls[0], label='optimized pulse')
plt.title('GRAPE pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

    DEBUG:qutip_qtrl.pulseoptim:Optimisation config summary...
      object classes:
        optimizer: OptimizerLBFGSB
        dynamics: DynamicsGenMat
        tslotcomp: TSlotCompUpdateAll
        fidcomp: FidCompTraceDiff
        propcomp: PropCompFrechet
        pulsegen: PulseGenRandom
    INFO:qutip_qtrl.dynamics:Setting memory optimisations for level 0
    INFO:qutip_qtrl.dynamics:Internal operator data type choosen to be <class 'numpy.ndarray'>
    INFO:qutip_qtrl.dynamics:phased dynamics generator caching False
    INFO:qutip_qtrl.dynamics:propagator gradient caching True
    INFO:qutip_qtrl.dynamics:eigenvector adjoint caching True
    INFO:qutip_qtrl.dynamics:use sparse eigen decomp False
    DEBUG:qutip_qtrl.fidcomp:Scale factor calculated as 0.125
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77189877-2.55094136e-17j]
     [-0.06888728+2.98015228e-02j]
     [-0.06888728-2.98015228e-02j]
     [ 0.22810123+3.09932752e-17j]]
    Evo final diff:
    [[-0.77189877+2.55094136e-17j]
     [ 0.06888728-2.98015228e-02j]
     [ 0.06888728+2.98015228e-02j]
     [ 0.77189877-3.09932752e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15036532376635012
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0045647848138128475 
    DEBUG:qutip_qtrl.tslotcomp:250 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77183948+2.11668240e-17j]
     [-0.06878968+3.01708340e-02j]
     [-0.06878968-3.01708340e-02j]
     [ 0.22816052-3.88175187e-17j]]
    Evo final diff:
    [[-0.77183948-2.11668240e-17j]
     [ 0.06878968-3.01708340e-02j]
     [ 0.06878968+3.01708340e-02j]
     [ 0.77183948+3.88175187e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15034462159320172
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004587903075889462 
    DEBUG:qutip_qtrl.tslotcomp:246 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77177915+1.40721819e-16j]
     [-0.06869664+3.05472003e-02j]
     [-0.06869664-3.05472003e-02j]
     [ 0.22822085-1.30622752e-16j]]
    Evo final diff:
    [[-0.77177915-1.40721819e-16j]
     [ 0.06869664-3.05472003e-02j]
     [ 0.06869664+3.05472003e-02j]
     [ 0.77177915+1.30622752e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15032385401393628
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004611869361896642 
    DEBUG:qutip_qtrl.tslotcomp:245 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.7717178 -1.27328110e-16j]
     [-0.06860605+3.09304740e-02j]
     [-0.06860605-3.09304740e-02j]
     [ 0.2282822 +1.19721597e-16j]]
    Evo final diff:
    [[-0.7717178 +1.27328110e-16j]
     [ 0.06860605-3.09304740e-02j]
     [ 0.06860605+3.09304740e-02j]
     [ 0.7717178 -1.19721597e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15030296035733728
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004636702772677686 
    DEBUG:qutip_qtrl.tslotcomp:243 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77165535+5.88065328e-17j]
     [-0.06851891+3.13212862e-02j]
     [-0.06851891-3.13212862e-02j]
     [ 0.22834465-8.84404665e-17j]]
    Evo final diff:
    [[-0.77165535-5.88065328e-17j]
     [ 0.06851891-3.13212862e-02j]
     [ 0.06851891+3.13212862e-02j]
     [ 0.77165535+8.84404665e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1502819613893671
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004662458936031275 
    DEBUG:qutip_qtrl.tslotcomp:243 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77160369-1.56147982e-16j]
     [-0.06844693+3.16424209e-02j]
     [-0.06844693-3.16424209e-02j]
     [ 0.22839631+1.61263908e-16j]]
    Evo final diff:
    [[-0.77160369+1.56147982e-16j]
     [ 0.06844693-3.16424209e-02j]
     [ 0.06844693+3.16424209e-02j]
     [ 0.77160369-1.61263908e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15026461863006424
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004683963412756786 
    DEBUG:qutip_qtrl.tslotcomp:242 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77153912-2.09592373e-16j]
     [-0.06836152+3.20493470e-02j]
     [-0.06836152-3.20493470e-02j]
     [ 0.22846088+2.41356774e-16j]]
    Evo final diff:
    [[-0.77153912+2.09592373e-16j]
     [ 0.06836152-3.20493470e-02j]
     [ 0.06836152+3.20493470e-02j]
     [ 0.77153912-2.41356774e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15024327002109059
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004711635679508653 
    DEBUG:qutip_qtrl.tslotcomp:241 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77147339-7.78719737e-17j]
     [-0.06827832+3.24638088e-02j]
     [-0.06827832-3.24638088e-02j]
     [ 0.22852661+8.00011105e-17j]]
    Evo final diff:
    [[-0.77147339+7.78719737e-17j]
     [ 0.06827832-3.24638088e-02j]
     [ 0.06827832+3.24638088e-02j]
     [ 0.77147339-8.00011105e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15022175486668604
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0047402962406730585 
    DEBUG:qutip_qtrl.tslotcomp:240 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77140648+2.80046031e-17j]
     [-0.06819712+3.28860163e-02j]
     [-0.06819712-3.28860163e-02j]
     [ 0.22859352-3.81147377e-17j]]
    Evo final diff:
    [[-0.77140648-2.80046031e-17j]
     [ 0.06819712-3.28860163e-02j]
     [ 0.06819712+3.28860163e-02j]
     [ 0.77140648+3.81147377e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1502000733832908
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004769979617145052 
    DEBUG:qutip_qtrl.tslotcomp:240 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77125928-2.61512952e-17j]
     [-0.0680189 +3.38029002e-02j]
     [-0.0680189 -3.38029002e-02j]
     [ 0.22874072+5.76477707e-18j]]
    Evo final diff:
    [[-0.77125928+2.61512952e-17j]
     [ 0.0680189 -3.38029002e-02j]
     [ 0.0680189 +3.38029002e-02j]
     [ 0.77125928-5.76477707e-18j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15015251951338354
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004836150174125501 
    DEBUG:qutip_qtrl.tslotcomp:239 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77118807+1.22057448e-17j]
     [-0.06793595+3.42564861e-02j]
     [-0.06793595-3.42564861e-02j]
     [ 0.22881193-7.62023164e-19j]]
    Evo final diff:
    [[-0.77118807-1.22057448e-17j]
     [ 0.06793595-3.42564861e-02j]
     [ 0.06793595+3.42564861e-02j]
     [ 0.77118807+7.62023164e-19j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15012995838876778
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004869736714693404 
    DEBUG:qutip_qtrl.tslotcomp:239 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77115973+1.17718943e-16j]
     [-0.06790298+3.44358790e-02j]
     [-0.06790298-3.44358790e-02j]
     [ 0.22884027-1.28526484e-16j]]
    Evo final diff:
    [[-0.77115973-1.17718943e-16j]
     [ 0.06790298-3.44358790e-02j]
     [ 0.06790298+3.44358790e-02j]
     [ 0.77115973+1.28526484e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15012099202714108
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004883171298510743 
    DEBUG:qutip_qtrl.tslotcomp:238 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77108648+1.01261093e-16j]
     [-0.06782731+3.49004135e-02j]
     [-0.06782731-3.49004135e-02j]
     [ 0.22891352-1.05943800e-16j]]
    Evo final diff:
    [[-0.77108648-1.01261093e-16j]
     [ 0.06782731-3.49004135e-02j]
     [ 0.06782731+3.49004135e-02j]
     [ 0.77108648+1.05943800e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1500982352062954
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0049183050014321335 
    DEBUG:qutip_qtrl.tslotcomp:236 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77101167-1.58451063e-16j]
     [-0.06775199+3.53753146e-02j]
     [-0.06775199-3.53753146e-02j]
     [ 0.22898833+1.42781948e-16j]]
    Evo final diff:
    [[-0.77101167+1.58451063e-16j]
     [ 0.06775199-3.53753146e-02j]
     [ 0.06775199+3.53753146e-02j]
     [ 0.77101167-1.42781948e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.15007518435797793
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004954795160336729 
    DEBUG:qutip_qtrl.tslotcomp:236 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77070543+2.84974062e-16j]
     [-0.06744441+3.72748920e-02j]
     [-0.06744441-3.72748920e-02j]
     [ 0.22929457-2.86800334e-16j]]
    Evo final diff:
    [[-0.77070543-2.84974062e-16j]
     [ 0.06744441-3.72748920e-02j]
     [ 0.06744441+3.72748920e-02j]
     [ 0.77070543+2.86800334e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1499812557582843
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00510634922996008 
    DEBUG:qutip_qtrl.tslotcomp:236 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.7706754 +1.79711587e-16j]
     [-0.06741431+3.74574774e-02j]
     [-0.06741431-3.74574774e-02j]
     [ 0.2293246 -1.73633551e-16j]]
    Evo final diff:
    [[-0.7706754 -1.79711587e-16j]
     [ 0.06741431-3.74574774e-02j]
     [ 0.06741431+3.74574774e-02j]
     [ 0.7706754 +1.73633551e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1499720816101175
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005121371114696426 
    DEBUG:qutip_qtrl.tslotcomp:235 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77059155+9.70869127e-17j]
     [-0.06733804+3.79884020e-02j]
     [-0.06733804-3.79884020e-02j]
     [ 0.22940845-1.03221474e-16j]]
    Evo final diff:
    [[-0.77059155-9.70869127e-17j]
     [ 0.06733804-3.79884020e-02j]
     [ 0.06733804+3.79884020e-02j]
     [ 0.77059155+1.03221474e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1499472181273278
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005165475053769101 
    DEBUG:qutip_qtrl.tslotcomp:235 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77050403+7.53639071e-17j]
     [-0.06725851+3.85367553e-02j]
     [-0.06725851-3.85367553e-02j]
     [ 0.22949597-5.80013822e-17j]]
    Evo final diff:
    [[-0.77050403-7.53639071e-17j]
     [ 0.06725851-3.85367553e-02j]
     [ 0.06725851+3.85367553e-02j]
     [ 0.77050403+5.80013822e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1499213110823868
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005211694407456673 
    DEBUG:qutip_qtrl.tslotcomp:234 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77041598-4.61548226e-17j]
     [-0.06718426+3.90903520e-02j]
     [-0.06718426-3.90903520e-02j]
     [ 0.22958402+5.80490602e-17j]]
    Evo final diff:
    [[-0.77041598+4.61548226e-17j]
     [ 0.06718426-3.90903520e-02j]
     [ 0.06718426+3.90903520e-02j]
     [ 0.77041598-5.80490602e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1498956394029262
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005259008966628942 
    DEBUG:qutip_qtrl.tslotcomp:234 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.7703487 -2.27062418e-16j]
     [-0.06712757+3.95093047e-02j]
     [-0.06712757-3.95093047e-02j]
     [ 0.2296513 +2.08090810e-16j]]
    Evo final diff:
    [[-0.7703487 +2.27062418e-16j]
     [ 0.06712757-3.95093047e-02j]
     [ 0.06712757+3.95093047e-02j]
     [ 0.7703487 -2.08090810e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14987605470290266
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005295254196829969 
    DEBUG:qutip_qtrl.tslotcomp:233 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77025662+7.73849242e-18j]
     [-0.06705288+4.00861662e-02j]
     [-0.06705288-4.00861662e-02j]
     [ 0.22974338-1.06700051e-17j]]
    Evo final diff:
    [[-0.77025662-7.73849242e-18j]
     [ 0.06705288-4.00861662e-02j]
     [ 0.06705288+4.00861662e-02j]
     [ 0.77025662+1.06700051e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14984956260293078
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005345766414596359 
    DEBUG:qutip_qtrl.tslotcomp:233 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77014532-5.51960516e-17j]
     [-0.06696267+4.07748148e-02j]
     [-0.06696267-4.07748148e-02j]
     [ 0.22985468+5.32329613e-17j]]
    Evo final diff:
    [[-0.77014532+5.51960516e-17j]
     [ 0.06696267-4.07748148e-02j]
     [ 0.06696267+4.07748148e-02j]
     [ 0.77014532-5.32329613e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14981760064067148
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0054069598138275846 
    DEBUG:qutip_qtrl.tslotcomp:232 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77004778-2.97572818e-16j]
     [-0.06688661+4.13813427e-02j]
     [-0.06688661-4.13813427e-02j]
     [ 0.22995222+2.98558151e-16j]]
    Evo final diff:
    [[-0.77004778+2.97572818e-16j]
     [ 0.06688661-4.13813427e-02j]
     [ 0.06688661+4.13813427e-02j]
     [ 0.77004778-2.98558151e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14978995562149755
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00546164189499911 
    DEBUG:qutip_qtrl.tslotcomp:232 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.77004778-2.13858864e-16j]
     [-0.06688661+4.13813660e-02j]
     [-0.06688661-4.13813660e-02j]
     [ 0.22995222+2.06379278e-16j]]
    Evo final diff:
    [[-0.77004778+2.13858864e-16j]
     [ 0.06688661-4.13813660e-02j]
     [ 0.06688661+4.13813660e-02j]
     [ 0.77004778-2.06379278e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14978995455654862
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005461644004449913 
    DEBUG:qutip_qtrl.tslotcomp:231 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76994788-2.15691840e-16j]
     [-0.06681423+4.19987925e-02j]
     [-0.06681423-4.19987925e-02j]
     [ 0.23005212+2.35671921e-16j]]
    Evo final diff:
    [[-0.76994788+2.15691840e-16j]
     [ 0.06681423-4.19987925e-02j]
     [ 0.06681423+4.19987925e-02j]
     [ 0.76994788-2.35671921e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14976194316334568
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005518008991489866 
    DEBUG:qutip_qtrl.tslotcomp:231 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76953622+5.10422021e-17j]
     [-0.06651603+4.44683426e-02j]
     [-0.06651603-4.44683426e-02j]
     [ 0.23046378-3.06618180e-17j]]
    Evo final diff:
    [[-0.76953622-5.10422021e-17j]
     [ 0.06651603-4.44683426e-02j]
     [ 0.06651603+4.44683426e-02j]
     [ 0.76953622+3.06618180e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14964695200330014
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005750410323258551 
    DEBUG:qutip_qtrl.tslotcomp:231 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76948242+1.04422209e-16j]
     [-0.06647706+4.47827260e-02j]
     [-0.06647706-4.47827260e-02j]
     [ 0.23051758-1.12913995e-16j]]
    Evo final diff:
    [[-0.76948242-1.04422209e-16j]
     [ 0.06647706-4.47827260e-02j]
     [ 0.06647706+4.47827260e-02j]
     [ 0.76948242+1.12913995e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1496319734342141
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005780753455337895 
    DEBUG:qutip_qtrl.tslotcomp:230 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76936661-1.26207892e-16j]
     [-0.0664024 +4.54761475e-02j]
     [-0.0664024 -4.54761475e-02j]
     [ 0.23063339+1.46787011e-16j]]
    Evo final diff:
    [[-0.76936661+1.26207892e-16j]
     [ 0.0664024 -4.54761475e-02j]
     [ 0.0664024 +4.54761475e-02j]
     [ 0.76936661-1.46787011e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14960058379623659
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005848237608974934 
    DEBUG:qutip_qtrl.tslotcomp:230 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76922636-4.78742954e-17j]
     [-0.06631196+4.63037782e-02j]
     [-0.06631196-4.63037782e-02j]
     [ 0.23077364+5.14492464e-17j]]
    Evo final diff:
    [[-0.76922636+4.78742954e-17j]
     [ 0.06631196-4.63037782e-02j]
     [ 0.06631196+4.63037782e-02j]
     [ 0.76922636-5.14492464e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.149562628360543
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.005929783212781529 
    DEBUG:qutip_qtrl.tslotcomp:229 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76910323-3.12267121e-16j]
     [-0.06623954+4.70288504e-02j]
     [-0.06623954-4.70288504e-02j]
     [ 0.23089677+3.14750096e-16j]]
    Evo final diff:
    [[-0.76910323+3.12267121e-16j]
     [ 0.06623954-4.70288504e-02j]
     [ 0.06623954+4.70288504e-02j]
     [ 0.76910323-3.14750096e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14952979306786862
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.006002055609451695 
    DEBUG:qutip_qtrl.tslotcomp:229 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76882804+9.48405649e-17j]
     [-0.06607742+4.86147590e-02j]
     [-0.06607742-4.86147590e-02j]
     [ 0.23117196-1.20738295e-16j]]
    Evo final diff:
    [[-0.76882804-9.48405649e-17j]
     [ 0.06607742-4.86147590e-02j]
     [ 0.06607742+4.86147590e-02j]
     [ 0.76882804+1.20738295e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1494565422264255
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.006162805692159602 
    DEBUG:qutip_qtrl.tslotcomp:228 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76869242-1.32726646e-16j]
     [-0.06600174+4.93936816e-02j]
     [-0.06600174-4.93936816e-02j]
     [ 0.23130758+1.58832098e-16j]]
    Evo final diff:
    [[-0.76869242+1.32726646e-16j]
     [ 0.06600174-4.93936816e-02j]
     [ 0.06600174+4.93936816e-02j]
     [ 0.76869242-1.58832098e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14942100013087564
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0062430387096873265 
    DEBUG:qutip_qtrl.tslotcomp:228 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76813485+1.03087271e-16j]
     [-0.06568986+5.24849732e-02j]
     [-0.06568986-5.24849732e-02j]
     [ 0.23186515-1.14254687e-16j]]
    Evo final diff:
    [[-0.76813485-1.03087271e-16j]
     [ 0.06568986-5.24849732e-02j]
     [ 0.06568986+5.24849732e-02j]
     [ 0.76813485+1.14254687e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14927524457291205
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.006569083324916834 
    DEBUG:qutip_qtrl.tslotcomp:227 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76797714-1.02145374e-16j]
     [-0.06561178+5.33489773e-02j]
     [-0.06561178-5.33489773e-02j]
     [ 0.23202286+9.08441224e-17j]]
    Evo final diff:
    [[-0.76797714+1.02145374e-16j]
     [ 0.06561178-5.33489773e-02j]
     [ 0.06561178+5.33489773e-02j]
     [ 0.76797714-9.08441224e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14923497606342362
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.006662193714221294 
    DEBUG:qutip_qtrl.tslotcomp:227 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76787816-1.72286453e-16j]
     [-0.06556271+5.38845233e-02j]
     [-0.06556271-5.38845233e-02j]
     [ 0.23212184+1.84650925e-16j]]
    Evo final diff:
    [[-0.76787816+1.72286453e-16j]
     [ 0.06556271-5.38845233e-02j]
     [ 0.06556271+5.38845233e-02j]
     [ 0.76787816-1.84650925e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14920971986666118
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.006720319708394118 
    DEBUG:qutip_qtrl.tslotcomp:226 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76771181-1.31229083e-16j]
     [-0.06548706+5.47778875e-02j]
     [-0.06548706-5.47778875e-02j]
     [ 0.23228819+1.36316939e-16j]]
    Evo final diff:
    [[-0.76771181+1.31229083e-16j]
     [ 0.06548706-5.47778875e-02j]
     [ 0.06548706+5.47778875e-02j]
     [ 0.76771181-1.36316939e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1491676467800418
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0068179342978206235 
    DEBUG:qutip_qtrl.tslotcomp:226 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76702056-1.20156666e-16j]
     [-0.06517078+5.83495373e-02j]
     [-0.06517078-5.83495373e-02j]
     [ 0.23297944+1.13525377e-16j]]
    Evo final diff:
    [[-0.76702056+1.20156666e-16j]
     [ 0.06517078-5.83495373e-02j]
     [ 0.06517078+5.83495373e-02j]
     [ 0.76702056-1.13525377e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14899311089617343
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0072160460650746786 
    DEBUG:qutip_qtrl.tslotcomp:226 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76668499-1.60224692e-16j]
     [-0.06501627+6.00095078e-02j]
     [-0.06501627-6.00095078e-02j]
     [ 0.23331501+1.54517133e-16j]]
    Evo final diff:
    [[-0.76668499+1.60224692e-16j]
     [ 0.06501627-6.00095078e-02j]
     [ 0.06501627+6.00095078e-02j]
     [ 0.76668499-1.54517133e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14890853335678722
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.007404976831375055 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76647752-3.76827907e-17j]
     [-0.06493397+6.10382228e-02j]
     [-0.06493397-6.10382228e-02j]
     [ 0.23352248+2.67197039e-17j]]
    Evo final diff:
    [[-0.76647752+3.76827907e-17j]
     [ 0.06493397-6.10382228e-02j]
     [ 0.06493397+6.10382228e-02j]
     [ 0.76647752-2.67197039e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14885746656123683
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0075231231693443 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76561307-1.50751468e-16j]
     [-0.0645886 +6.51496310e-02j]
     [-0.0645886 -6.51496310e-02j]
     [ 0.23438693+1.32237557e-16j]]
    Evo final diff:
    [[-0.76561307+1.50751468e-16j]
     [ 0.0645886 -6.51496310e-02j]
     [ 0.0645886 +6.51496310e-02j]
     [ 0.76561307-1.32237557e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14864488392760608
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.008003096449816978 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76535417-1.15040567e-16j]
     [-0.06448449+6.63311800e-02j]
     [-0.06448449-6.63311800e-02j]
     [ 0.23464583+1.08198358e-16j]]
    Evo final diff:
    [[-0.76535417+1.15040567e-16j]
     [ 0.06448449-6.63311800e-02j]
     [ 0.06448449+6.63311800e-02j]
     [ 0.76535417-1.08198358e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14858126913238895
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.008143139088336046 
    DEBUG:qutip_qtrl.tslotcomp:224 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76509764+1.42755457e-16j]
     [-0.0643946 +6.74994960e-02j]
     [-0.0643946 -6.74994960e-02j]
     [ 0.23490236-1.53228477e-16j]]
    Evo final diff:
    [[-0.76509764-1.42755457e-16j]
     [ 0.0643946 -6.74994960e-02j]
     [ 0.0643946 +6.74994960e-02j]
     [ 0.76509764+1.53228477e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14851931020317485
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.008282389953314187 
    DEBUG:qutip_qtrl.tslotcomp:224 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76472984-1.31703374e-16j]
     [-0.06426517+6.91391073e-02j]
     [-0.06426517-6.91391073e-02j]
     [ 0.23527016+1.36538489e-16j]]
    Evo final diff:
    [[-0.76472984+1.31703374e-16j]
     [ 0.06426517-6.91391073e-02j]
     [ 0.06426517+6.91391073e-02j]
     [ 0.76472984-1.36538489e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1484304875308462
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.008479143336343914 
    DEBUG:qutip_qtrl.tslotcomp:223 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76445043-1.83047089e-16j]
     [-0.06417583+7.03657098e-02j]
     [-0.06417583-7.03657098e-02j]
     [ 0.23554957+1.81247991e-16j]]
    Evo final diff:
    [[-0.76445043+1.83047089e-16j]
     [ 0.06417583-7.03657098e-02j]
     [ 0.06417583+7.03657098e-02j]
     [ 0.76445043-1.81247991e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14836358394270233
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.008627229039483111 
    DEBUG:qutip_qtrl.tslotcomp:223 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76408158-2.58493949e-16j]
     [-0.0640573 +7.19516561e-02j]
     [-0.0640573 -7.19516561e-02j]
     [ 0.23591842+2.52982728e-16j]]
    Evo final diff:
    [[-0.76408158+2.58493949e-16j]
     [ 0.0640573 -7.19516561e-02j]
     [ 0.0640573 +7.19516561e-02j]
     [ 0.76408158-2.52982728e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14827525883941334
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.008819840297384056 
    DEBUG:qutip_qtrl.tslotcomp:222 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76377701-7.49657543e-17j]
     [-0.06396241+7.32411087e-02j]
     [-0.06396241-7.32411087e-02j]
     [ 0.23622299+6.72159694e-17j]]
    Evo final diff:
    [[-0.76377701+7.49657543e-17j]
     [ 0.06396241-7.32411087e-02j]
     [ 0.06396241+7.32411087e-02j]
     [ 0.76377701-6.72159694e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1482026935754332
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00897731903198797 
    DEBUG:qutip_qtrl.tslotcomp:222 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76250371-6.08101511e-17j]
     [-0.06356174+7.83902979e-02j]
     [-0.06356174-7.83902979e-02j]
     [ 0.23749629+4.20703986e-17j]]
    Evo final diff:
    [[-0.76250371+6.08101511e-17j]
     [ 0.06356174-7.83902979e-02j]
     [ 0.06356174+7.83902979e-02j]
     [ 0.76250371-4.20703986e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14789925896299908
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.009613410629698838 
    DEBUG:qutip_qtrl.tslotcomp:222 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76210324-9.78356040e-17j]
     [-0.0634346 +7.99371549e-02j]
     [-0.0634346 -7.99371549e-02j]
     [ 0.23789676+9.42272214e-17j]]
    Evo final diff:
    [[-0.76210324+9.78356040e-17j]
     [ 0.0634346 -7.99371549e-02j]
     [ 0.0634346 +7.99371549e-02j]
     [ 0.76210324-9.42272214e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14780381082268165
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.009806540598442064 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76172197+6.92533513e-17j]
     [-0.06332874+8.13972831e-02j]
     [-0.06332874-8.13972831e-02j]
     [ 0.23827803-6.20161667e-17j]]
    Evo final diff:
    [[-0.76172197-6.92533513e-17j]
     [ 0.06332874-8.13972831e-02j]
     [ 0.06332874+8.13972831e-02j]
     [ 0.76172197+6.20161667e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14771410153424488
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.009989482248806253 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76134696-7.44981707e-17j]
     [-0.06322419+8.28062610e-02j]
     [-0.06322419-8.28062610e-02j]
     [ 0.23865304+6.74565076e-17j]]
    Evo final diff:
    [[-0.76134696+7.44981707e-17j]
     [ 0.06322419-8.28062610e-02j]
     [ 0.06322419+8.28062610e-02j]
     [ 0.76134696-6.74565076e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14762584364038395
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.01016668750784279 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.76093571-2.86405250e-16j]
     [-0.06311941+8.43251460e-02j]
     [-0.06311941-8.43251460e-02j]
     [ 0.23906429+2.76667005e-16j]]
    Evo final diff:
    [[-0.76093571+2.86405250e-16j]
     [ 0.06311941-8.43251460e-02j]
     [ 0.06311941+8.43251460e-02j]
     [ 0.76093571-2.76667005e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1475294865389161
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.010358363315847515 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.7592134 -2.41302423e-16j]
     [-0.06267519+9.03846995e-02j]
     [-0.06267519-9.03846995e-02j]
     [ 0.2407866 +2.42893332e-16j]]
    Evo final diff:
    [[-0.7592134 +2.41302423e-16j]
     [ 0.06267519-9.03846995e-02j]
     [ 0.06267519+9.03846995e-02j]
     [ 0.7592134 -2.42893332e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1471256381754032
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.011129549789216958 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75860469-1.16619341e-16j]
     [-0.06251641+9.24221423e-02j]
     [-0.06251641-9.24221423e-02j]
     [ 0.24139531+1.09256970e-16j]]
    Evo final diff:
    [[-0.75860469+1.16619341e-16j]
     [ 0.06251641-9.24221423e-02j]
     [ 0.06251641+9.24221423e-02j]
     [ 0.75860469-1.09256970e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14698280674340142
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.011390945064876966 
    DEBUG:qutip_qtrl.tslotcomp:229 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75808376-1.64798275e-17j]
     [-0.06239916+9.41413961e-02j]
     [-0.06239916-9.41413961e-02j]
     [ 0.24191624+3.19520535e-17j]]
    Evo final diff:
    [[-0.75808376+1.64798275e-17j]
     [ 0.06239916-9.41413961e-02j]
     [ 0.06239916+9.41413961e-02j]
     [ 0.75808376-3.19520535e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1468618118606218
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.011612051434936736 
    DEBUG:qutip_qtrl.tslotcomp:229 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75589997-3.00391237e-16j]
     [-0.06190185+1.00993292e-01j]
     [-0.06190185-1.00993292e-01j]
     [ 0.24410003+2.96853700e-16j]]
    Evo final diff:
    [[-0.75589997+3.00391237e-16j]
     [ 0.06190185-1.00993292e-01j]
     [ 0.06190185+1.00993292e-01j]
     [ 0.75589997-2.96853700e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14635406186471064
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.01249906868426023 
    DEBUG:qutip_qtrl.tslotcomp:229 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75522407-3.38663468e-16j]
     [-0.06174636+1.03009891e-01j]
     [-0.06174636-1.03009891e-01j]
     [ 0.24477593+3.38545680e-16j]]
    Evo final diff:
    [[-0.75522407+3.38663468e-16j]
     [ 0.06174636-1.03009891e-01j]
     [ 0.06174636+1.03009891e-01j]
     [ 0.75522407-3.38545680e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14619676207164434
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.01276171507999066 
    DEBUG:qutip_qtrl.tslotcomp:228 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75456473-1.45174409e-16j]
     [-0.06161481+1.04944596e-01j]
     [-0.06161481-1.04944596e-01j]
     [ 0.24543527+1.34732021e-16j]]
    Evo final diff:
    [[-0.75456473+1.45174409e-16j]
     [ 0.06161481-1.04944596e-01j]
     [ 0.06161481+1.04944596e-01j]
     [ 0.75456473-1.34732021e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14604442241061646
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.013014106977598075 
    DEBUG:qutip_qtrl.tslotcomp:228 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75400074-2.07637869e-16j]
     [-0.06150177+1.06567497e-01j]
     [-0.06150177-1.06567497e-01j]
     [ 0.24599926+1.97342137e-16j]]
    Evo final diff:
    [[-0.75400074+2.07637869e-16j]
     [ 0.06150177-1.06567497e-01j]
     [ 0.06150177+1.06567497e-01j]
     [ 0.75400074-1.97342137e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14591405262312274
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.013226243087505554 
    DEBUG:qutip_qtrl.tslotcomp:227 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75329322-3.41590726e-16j]
     [-0.06137171+1.08567516e-01j]
     [-0.06137171-1.08567516e-01j]
     [ 0.24670678+3.39017993e-16j]]
    Evo final diff:
    [[-0.75329322+3.41590726e-16j]
     [ 0.06137171-1.08567516e-01j]
     [ 0.06137171+1.08567516e-01j]
     [ 0.75329322-3.39017993e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14575101699673845
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.013488035471472409 
    DEBUG:qutip_qtrl.tslotcomp:227 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.75032552-4.80734022e-16j]
     [-0.06081948+1.16524014e-01j]
     [-0.06081948-1.16524014e-01j]
     [ 0.24967448+4.80915113e-16j]]
    Evo final diff:
    [[-0.75032552+4.80734022e-16j]
     [ 0.06081948-1.16524014e-01j]
     [ 0.06081948+1.16524014e-01j]
     [ 0.75032552-4.80915113e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14506631033593037
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.014534290505311243 
    DEBUG:qutip_qtrl.tslotcomp:227 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.74812307-2.06065599e-16j]
     [-0.06040439+1.22036372e-01j]
     [-0.06040439-1.22036372e-01j]
     [ 0.25187693+2.02477618e-16j]]
    Evo final diff:
    [[-0.74812307+2.06065599e-16j]
     [ 0.06040439-1.22036372e-01j]
     [ 0.06040439+1.22036372e-01j]
     [ 0.74812307-2.02477618e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14455742521549136
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.015263009664036133 
    DEBUG:qutip_qtrl.tslotcomp:226 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.74717651-1.25265637e-16j]
     [-0.06024967+1.24328138e-01j]
     [-0.06024967-1.24328138e-01j]
     [ 0.25282349+1.26390254e-16j]]
    Evo final diff:
    [[-0.74717651+1.25265637e-16j]
     [ 0.06024967-1.24328138e-01j]
     [ 0.06024967+1.24328138e-01j]
     [ 0.74717651-1.26390254e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14434006304096134
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.01556638707147303 
    DEBUG:qutip_qtrl.tslotcomp:226 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.74622892-2.52875771e-16j]
     [-0.06009432+1.26572029e-01j]
     [-0.06009432-1.26572029e-01j]
     [ 0.25377108+2.69034232e-16j]]
    Evo final diff:
    [[-0.74622892+2.52875771e-16j]
     [ 0.06009432-1.26572029e-01j]
     [ 0.06009432+1.26572029e-01j]
     [ 0.74622892-2.69034232e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14412235331786286
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.01586381333505983 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.74520604-1.76819258e-16j]
     [-0.05994057+1.28941675e-01j]
     [-0.05994057-1.28941675e-01j]
     [ 0.25479396+1.73642552e-16j]]
    Evo final diff:
    [[-0.74520604+1.76819258e-16j]
     [ 0.05994057-1.28941675e-01j]
     [ 0.05994057+1.28941675e-01j]
     [ 0.74520604-1.73642552e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14388771644252965
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.016178186421546756 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.74091634-2.87190550e-16j]
     [-0.05929076+1.38338465e-01j]
     [-0.05929076-1.38338465e-01j]
     [ 0.25908366+2.73786612e-16j]]
    Evo final diff:
    [[-0.74091634+2.87190550e-16j]
     [ 0.05929076-1.38338465e-01j]
     [ 0.05929076+1.38338465e-01j]
     [ 0.74091634-2.73786612e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14290248850119172
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.017428168981806554 
    DEBUG:qutip_qtrl.tslotcomp:225 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.74062063-3.15718456e-16j]
     [-0.05924575+1.38957016e-01j]
     [-0.05924575-1.38957016e-01j]
     [ 0.25937937+3.14054176e-16j]]
    Evo final diff:
    [[-0.74062063+3.15718456e-16j]
     [ 0.05924575-1.38957016e-01j]
     [ 0.05924575+1.38957016e-01j]
     [ 0.74062063-3.14054176e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1428345054298097
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.017510614819729996 
    DEBUG:qutip_qtrl.tslotcomp:224 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.73937414-4.15681474e-16j]
     [-0.05907625+1.41533582e-01j]
     [-0.05907625-1.41533582e-01j]
     [ 0.26062586+4.06640596e-16j]]
    Evo final diff:
    [[-0.73937414+4.15681474e-16j]
     [ 0.05907625-1.41533582e-01j]
     [ 0.05907625+1.41533582e-01j]
     [ 0.73937414-4.06640596e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14254896842965076
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.01785388130329608 
    DEBUG:qutip_qtrl.tslotcomp:224 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.73414975-1.61735124e-16j]
     [-0.05836421+1.51726404e-01j]
     [-0.05836421-1.51726404e-01j]
     [ 0.26585025+1.54396872e-16j]]
    Evo final diff:
    [[-0.73414975+1.61735124e-16j]
     [ 0.05836421-1.51726404e-01j]
     [ 0.05836421+1.51726404e-01j]
     [ 0.73414975-1.54396872e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1413507852618395
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.019214241062380285 
    DEBUG:qutip_qtrl.tslotcomp:224 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.73369778-5.24209189e-16j]
     [-0.0583026 +1.52566418e-01j]
     [-0.0583026 -1.52566418e-01j]
     [ 0.26630222+5.27488180e-16j]]
    Evo final diff:
    [[-0.73369778+5.24209189e-16j]
     [ 0.0583026 -1.52566418e-01j]
     [ 0.0583026 +1.52566418e-01j]
     [ 0.73369778-5.27488180e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14124703274087408
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.019326496244075477 
    DEBUG:qutip_qtrl.tslotcomp:223 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.73217753-2.38035378e-16j]
     [-0.05811666+1.55351065e-01j]
     [-0.05811666-1.55351065e-01j]
     [ 0.26782247+2.39444850e-16j]]
    Evo final diff:
    [[-0.73217753+2.38035378e-16j]
     [ 0.05811666-1.55351065e-01j]
     [ 0.05811666+1.55351065e-01j]
     [ 0.73217753-2.39444850e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14089885721301582
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.019698470198813694 
    DEBUG:qutip_qtrl.tslotcomp:223 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.73137754-2.14833521e-16j]
     [-0.05801886+1.56789710e-01j]
     [-0.05801886-1.56789710e-01j]
     [ 0.26862246+2.21132447e-16j]]
    Evo final diff:
    [[-0.73137754+2.14833521e-16j]
     [ 0.05801886-1.56789710e-01j]
     [ 0.05801886+1.56789710e-01j]
     [ 0.73137754-2.21132447e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14071557611653554
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.019890710693953796 
    DEBUG:qutip_qtrl.tslotcomp:222 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.72977185-2.52048580e-16j]
     [-0.05783769+1.59626387e-01j]
     [-0.05783769-1.59626387e-01j]
     [ 0.27022815+2.45013017e-16j]]
    Evo final diff:
    [[-0.72977185+2.52048580e-16j]
     [ 0.05783769-1.59626387e-01j]
     [ 0.05783769+1.59626387e-01j]
     [ 0.72977185-2.45013017e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.14034818529570556
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.02026964795868936 
    DEBUG:qutip_qtrl.tslotcomp:222 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.72305153-4.05055452e-16j]
     [-0.05708328+1.70802420e-01j]
     [-0.05708328-1.70802420e-01j]
     [ 0.27694847+3.91369174e-16j]]
    Evo final diff:
    [[-0.72305153+4.05055452e-16j]
     [ 0.05708328-1.70802420e-01j]
     [ 0.05708328+1.70802420e-01j]
     [ 0.72305153-3.91369174e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1388088711727402
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.02176359227660934 
    DEBUG:qutip_qtrl.tslotcomp:222 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.72031644-3.06935652e-16j]
     [-0.05677887+1.75063762e-01j]
     [-0.05677887-1.75063762e-01j]
     [ 0.27968356+2.96222417e-16j]]
    Evo final diff:
    [[-0.72031644+3.06935652e-16j]
     [ 0.05677887-1.75063762e-01j]
     [ 0.05677887+1.75063762e-01j]
     [ 0.72031644-2.96222417e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13818173275480042
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.022333461854079847 
    DEBUG:qutip_qtrl.tslotcomp:221 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.71829552-2.74111542e-16j]
     [-0.05657714+1.78122920e-01j]
     [-0.05657714-1.78122920e-01j]
     [ 0.28170448+2.66503711e-16j]]
    Evo final diff:
    [[-0.71829552+2.74111542e-16j]
     [ 0.05657714-1.78122920e-01j]
     [ 0.05657714+1.78122920e-01j]
     [ 0.71829552-2.66503711e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13771930071189628
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.022742088412171034 
    DEBUG:qutip_qtrl.tslotcomp:221 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.71715224-3.37997683e-17j]
     [-0.05646361+1.79819288e-01j]
     [-0.05646361-1.79819288e-01j]
     [ 0.28284776+3.32857249e-17j]]
    Evo final diff:
    [[-0.71715224+3.37997683e-17j]
     [ 0.05646361-1.79819288e-01j]
     [ 0.05646361+1.79819288e-01j]
     [ 0.71715224-3.32857249e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13745761136318932
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.022968666713719377 
    DEBUG:qutip_qtrl.tslotcomp:220 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.71501756-5.14151823e-16j]
     [-0.05626945+1.82922485e-01j]
     [-0.05626945-1.82922485e-01j]
     [ 0.28498244+5.28637815e-16j]]
    Evo final diff:
    [[-0.71501756+5.14151823e-16j]
     [ 0.05626945-1.82922485e-01j]
     [ 0.05626945+1.82922485e-01j]
     [ 0.71501756-5.28637815e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13696924903939514
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.023382984312986955 
    DEBUG:qutip_qtrl.tslotcomp:220 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.70724479-3.80578935e-16j]
     [-0.05557519+1.93583144e-01j]
     [-0.05557519-1.93583144e-01j]
     [ 0.29275521+3.65120435e-16j]]
    Evo final diff:
    [[-0.70724479+3.80578935e-16j]
     [ 0.05557519-1.93583144e-01j]
     [ 0.05557519+1.93583144e-01j]
     [ 0.70724479-3.65120435e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13518955822137682
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0248055696744206 
    DEBUG:qutip_qtrl.tslotcomp:219 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.70476289-3.36966967e-16j]
     [-0.05537687+1.96798521e-01j]
     [-0.05537687-1.96798521e-01j]
     [ 0.29523711+3.41867197e-16j]]
    Evo final diff:
    [[-0.70476289+3.36966967e-16j]
     [ 0.05537687-1.96798521e-01j]
     [ 0.05537687+1.96798521e-01j]
     [ 0.70476289-3.41867197e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1346217477240938
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.025233893696598158 
    DEBUG:qutip_qtrl.tslotcomp:219 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.69636021-6.66154244e-16j]
     [-0.05472491+2.07074352e-01j]
     [-0.05472491-2.07074352e-01j]
     [ 0.30363979+6.55156393e-16j]]
    Evo final diff:
    [[-0.69636021+6.66154244e-16j]
     [ 0.05472491-2.07074352e-01j]
     [ 0.05472491+2.07074352e-01j]
     [ 0.69636021-6.55156393e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13269803577305936
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.026601139778172748 
    DEBUG:qutip_qtrl.tslotcomp:218 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.69351084-7.00495450e-16j]
     [-0.05452841+2.10362922e-01j]
     [-0.05452841-2.10362922e-01j]
     [ 0.30648916+7.15640631e-16j]]
    Evo final diff:
    [[-0.69351084+7.00495450e-16j]
     [ 0.05452841-2.10362922e-01j]
     [ 0.05452841+2.10362922e-01j]
     [ 0.69351084-7.15640631e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13204579882983583
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.027037884996739613 
    DEBUG:qutip_qtrl.tslotcomp:218 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.69221895-4.13943992e-16j]
     [-0.05444051+2.11823520e-01j]
     [-0.05444051-2.11823520e-01j]
     [ 0.30778105+4.07945453e-16j]]
    Evo final diff:
    [[-0.69221895+4.13943992e-16j]
     [ 0.05444051-2.11823520e-01j]
     [ 0.05444051+2.11823520e-01j]
     [ 0.69221895-4.07945453e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13175001116099017
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.02723174764585996 
    DEBUG:qutip_qtrl.tslotcomp:217 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.68924456-7.61399565e-16j]
     [-0.05425896+2.15118460e-01j]
     [-0.05425896-2.15118460e-01j]
     [ 0.31075544+7.57525576e-16j]]
    Evo final diff:
    [[-0.68924456+7.61399565e-16j]
     [ 0.05425896-2.15118460e-01j]
     [ 0.05425896+2.15118460e-01j]
     [ 0.68924456-7.57525576e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.13106951377635798
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.027668468893053456 
    DEBUG:qutip_qtrl.tslotcomp:217 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.67688921-1.05735288e-16j]
     [-0.05355021+2.27852030e-01j]
     [-0.05355021-2.27852030e-01j]
     [ 0.32311079+1.17867251e-16j]]
    Evo final diff:
    [[-0.67688921+1.05735288e-16j]
     [ 0.05355021-2.27852030e-01j]
     [ 0.05355021+2.27852030e-01j]
     [ 0.67688921-1.17867251e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1282407928102971
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.029351821888100513 
    DEBUG:qutip_qtrl.tslotcomp:217 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.67533842-3.44934309e-16j]
     [-0.05346656+2.29350225e-01j]
     [-0.05346656-2.29350225e-01j]
     [ 0.32466158+3.45994917e-16j]]
    Evo final diff:
    [[-0.67533842+3.44934309e-16j]
     [ 0.05346656-2.29350225e-01j]
     [ 0.05346656+2.29350225e-01j]
     [ 0.67533842-3.45994917e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.12788554340376415
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.02954935346053595 
    DEBUG:qutip_qtrl.tslotcomp:216 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.67185283-3.76863339e-16j]
     [-0.05329816+2.32645474e-01j]
     [-0.05329816-2.32645474e-01j]
     [ 0.32814717+3.85742587e-16j]]
    Evo final diff:
    [[-0.67185283+3.76863339e-16j]
     [ 0.05329816-2.32645474e-01j]
     [ 0.05329816+2.32645474e-01j]
     [ 0.67185283-3.85742587e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1270877097505528
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.02998299223687019 
    DEBUG:qutip_qtrl.tslotcomp:216 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.67137162-3.02515149e-16j]
     [-0.05327543+2.33092394e-01j]
     [-0.05327543-2.33092394e-01j]
     [ 0.32862838+3.12119039e-16j]]
    Evo final diff:
    [[-0.67137162+3.02515149e-16j]
     [ 0.05327543-2.33092394e-01j]
     [ 0.05327543+2.33092394e-01j]
     [ 0.67137162-3.12119039e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.12697754694141616
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.030041753953705286 
    DEBUG:qutip_qtrl.tslotcomp:215 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.66777659-3.33278998e-16j]
     [-0.05312731+2.36369924e-01j]
     [-0.05312731-2.36369924e-01j]
     [ 0.33222341+3.28784122e-16j]]
    Evo final diff:
    [[-0.66777659+3.33278998e-16j]
     [ 0.05312731-2.36369924e-01j]
     [ 0.05312731+2.36369924e-01j]
     [ 0.66777659-3.28784122e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.12615470502121254
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.030472213981305986 
    DEBUG:qutip_qtrl.tslotcomp:215 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.6566633 -6.49942158e-16j]
     [-0.05271215+2.45874191e-01j]
     [-0.05271215-2.45874191e-01j]
     [ 0.3433367 +6.15169790e-16j]]
    Evo final diff:
    [[-0.6566633 +6.49942158e-16j]
     [ 0.05271215-2.45874191e-01j]
     [ 0.05271215+2.45874191e-01j]
     [ 0.6566633 -6.15169790e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.12360984315044998
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03171612599370095 
    DEBUG:qutip_qtrl.tslotcomp:214 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.65267779-6.11073304e-16j]
     [-0.05259357+2.49068998e-01j]
     [-0.05259357-2.49068998e-01j]
     [ 0.34732221+6.02183311e-16j]]
    Evo final diff:
    [[-0.65267779+6.11073304e-16j]
     [ 0.05259357-2.49068998e-01j]
     [ 0.05259357+2.49068998e-01j]
     [ 0.65267779-6.02183311e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.12269743680533629
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0321323084875968 
    DEBUG:qutip_qtrl.tslotcomp:214 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.64250517-6.31996656e-16j]
     [-0.05233125+2.56747951e-01j]
     [-0.05233125-2.56747951e-01j]
     [ 0.35749483+6.45417945e-16j]]
    Evo final diff:
    [[-0.64250517+6.31996656e-16j]
     [ 0.05233125-2.56747951e-01j]
     [ 0.05233125+2.56747951e-01j]
     [ 0.64250517-6.45417945e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.1203677399457908
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.033128544257186636 
    DEBUG:qutip_qtrl.tslotcomp:213 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.63817142-3.74367938e-16j]
     [-0.05224773+2.59822030e-01j]
     [-0.05224773-2.59822030e-01j]
     [ 0.36182858+3.68719064e-16j]]
    Evo final diff:
    [[-0.63817142+3.74367938e-16j]
     [ 0.05224773-2.59822030e-01j]
     [ 0.05224773+2.59822030e-01j]
     [ 0.63817142-3.68719064e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.11937501701153475
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03352562557188911 
    DEBUG:qutip_qtrl.tslotcomp:213 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.63638735-7.47434162e-16j]
     [-0.0522164 +2.61055047e-01j]
     [-0.0522164 -2.61055047e-01j]
     [ 0.36361265+7.61622335e-16j]]
    Evo final diff:
    [[-0.63638735+7.47434162e-16j]
     [ 0.0522164 -2.61055047e-01j]
     [ 0.0522164 +2.61055047e-01j]
     [ 0.63638735-7.61622335e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.11896628721909538
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03368456960336331 
    DEBUG:qutip_qtrl.tslotcomp:212 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.63192967-3.41137291e-16j]
     [-0.05216594+2.64056561e-01j]
     [-0.05216594-2.64056561e-01j]
     [ 0.36807033+3.44847895e-16j]]
    Evo final diff:
    [[-0.63192967+3.41137291e-16j]
     [ 0.05216594-2.64056561e-01j]
     [ 0.05216594+2.64056561e-01j]
     [ 0.63192967-3.44847895e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.11794556487170456
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.034070275835932376 
    DEBUG:qutip_qtrl.tslotcomp:212 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.61471667-7.23037961e-16j]
     [-0.05207953+2.74628085e-01j]
     [-0.05207953-2.74628085e-01j]
     [ 0.38528333+7.15646926e-16j]]
    Evo final diff:
    [[-0.61471667+7.23037961e-16j]
     [ 0.05207953-2.74628085e-01j]
     [ 0.05207953+2.74628085e-01j]
     [ 0.61471667-7.15646926e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.11400236221754698
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03541748065730033 
    DEBUG:qutip_qtrl.tslotcomp:211 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.60982546-7.32252542e-16j]
     [-0.05209479+2.77356154e-01j]
     [-0.05209479-2.77356154e-01j]
     [ 0.39017454+7.37513289e-16j]]
    Evo final diff:
    [[-0.60982546+7.32252542e-16j]
     [ 0.05209479-2.77356154e-01j]
     [ 0.05209479+2.77356154e-01j]
     [ 0.60982546-7.37513289e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.11288184881395368
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03576168589295334 
    DEBUG:qutip_qtrl.tslotcomp:210 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.6048515 -5.91742838e-16j]
     [-0.05213982+2.80009487e-01j]
     [-0.05213982-2.80009487e-01j]
     [ 0.3951485 +5.95413362e-16j]]
    Evo final diff:
    [[-0.6048515 +5.91742838e-16j]
     [ 0.05213982-2.80009487e-01j]
     [ 0.05213982+2.80009487e-01j]
     [ 0.6048515 -5.95413362e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.11174230260470346
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03609484772033449 
    DEBUG:qutip_qtrl.tslotcomp:210 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.59190718-5.22138295e-16j]
     [-0.05232739+2.86375879e-01j]
     [-0.05232739-2.86375879e-01j]
     [ 0.40809282+4.98319718e-16j]]
    Evo final diff:
    [[-0.59190718+5.22138295e-16j]
     [ 0.05232739-2.86375879e-01j]
     [ 0.05232739+2.86375879e-01j]
     [ 0.59190718-4.98319718e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.10877585250548147
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03688639218966988 
    DEBUG:qutip_qtrl.tslotcomp:209 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.58665514-6.93706741e-16j]
     [-0.05244071+2.88747292e-01j]
     [-0.05244071-2.88747292e-01j]
     [ 0.41334486+7.16098749e-16j]]
    Evo final diff:
    [[-0.58665514+6.93706741e-16j]
     [ 0.05244071-2.88747292e-01j]
     [ 0.05244071+2.88747292e-01j]
     [ 0.58665514-7.16098749e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.10757231878717691
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03717784815173896 
    DEBUG:qutip_qtrl.tslotcomp:209 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.57551747-2.43829410e-16j]
     [-0.05273933+2.93387292e-01j]
     [-0.05273933-2.93387292e-01j]
     [ 0.42448253+2.54354300e-16j]]
    Evo final diff:
    [[-0.57551747+2.43829410e-16j]
     [ 0.05273933-2.93387292e-01j]
     [ 0.05273933+2.93387292e-01j]
     [ 0.57551747-2.54354300e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.10501947583332127
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.037741307722651544 
    DEBUG:qutip_qtrl.tslotcomp:208 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.57004556-5.38289960e-16j]
     [-0.05291526+2.95477594e-01j]
     [-0.05291526-2.95477594e-01j]
     [ 0.42995444+5.29817841e-16j]]
    Evo final diff:
    [[-0.57004556+5.38289960e-16j]
     [ 0.05291526-2.95477594e-01j]
     [ 0.05291526+2.95477594e-01j]
     [ 0.57004556-5.29817841e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.10376474223168573
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03799210756936777 
    DEBUG:qutip_qtrl.tslotcomp:208 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.56819923-7.01294011e-16j]
     [-0.05297896+2.96155687e-01j]
     [-0.05297896-2.96155687e-01j]
     [ 0.43180077+7.07312924e-16j]]
    Evo final diff:
    [[-0.56819923+7.01294011e-16j]
     [ 0.05297896-2.96155687e-01j]
     [ 0.05297896+2.96155687e-01j]
     [ 0.56819923-7.07312924e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.10334133261595366
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03807290894205984 
    DEBUG:qutip_qtrl.tslotcomp:207 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.56266696-8.23316283e-16j]
     [-0.0532038 +2.98107309e-01j]
     [-0.0532038 -2.98107309e-01j]
     [ 0.43733304+8.25991929e-16j]]
    Evo final diff:
    [[-0.56266696+8.23316283e-16j]
     [ 0.0532038 -2.98107309e-01j]
     [ 0.0532038 +2.98107309e-01j]
     [ 0.56266696-8.25991929e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.10207317997115345
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.038303405315408705 
    DEBUG:qutip_qtrl.tslotcomp:207 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.54380031-5.48640185e-16j]
     [-0.05412485+3.03875599e-01j]
     [-0.05412485-3.03875599e-01j]
     [ 0.45619969+5.63006095e-16j]]
    Evo final diff:
    [[-0.54380031+5.48640185e-16j]
     [ 0.05412485-3.03875599e-01j]
     [ 0.05412485+3.03875599e-01j]
     [ 0.54380031-5.63006095e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.09774716357938441
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03896463799501109 
    DEBUG:qutip_qtrl.tslotcomp:206 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.53806598-7.59563749e-16j]
     [-0.05445063+3.05366055e-01j]
     [-0.05445063-3.05366055e-01j]
     [ 0.46193402+7.61282559e-16j]]
    Evo final diff:
    [[-0.53806598+7.59563749e-16j]
     [ 0.05445063-3.05366055e-01j]
     [ 0.05445063+3.05366055e-01j]
     [ 0.53806598-7.61282559e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.09643207429105892
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03912916463139644 
    DEBUG:qutip_qtrl.tslotcomp:205 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.5323011 -2.72829793e-16j]
     [-0.05480985+3.06741722e-01j]
     [-0.05480985-3.06741722e-01j]
     [ 0.4676989 +2.51845903e-16j]]
    Evo final diff:
    [[-0.5323011 +2.72829793e-16j]
     [ 0.05480985-3.06741722e-01j]
     [ 0.05480985+3.06741722e-01j]
     [ 0.5323011 -2.51845903e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.09510976729548923
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03927756575439362 
    DEBUG:qutip_qtrl.tslotcomp:205 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.5190676 -7.01774783e-16j]
     [-0.05572398+3.09449290e-01j]
     [-0.05572398-3.09449290e-01j]
     [ 0.4809324 +7.10992040e-16j]]
    Evo final diff:
    [[-0.5190676 +7.01774783e-16j]
     [ 0.05572398-3.09449290e-01j]
     [ 0.05572398+3.09449290e-01j]
     [ 0.5190676 -7.10992040e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.09207379958468942
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.039554883929803615 
    DEBUG:qutip_qtrl.tslotcomp:204 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.51323489-6.20733383e-16j]
     [-0.05616719+3.10447731e-01j]
     [-0.05616719-3.10447731e-01j]
     [ 0.48676511+6.23231245e-16j]]
    Evo final diff:
    [[-0.51323489+6.20733383e-16j]
     [ 0.05616719-3.10447731e-01j]
     [ 0.05616719+3.10447731e-01j]
     [ 0.51323489-6.23231245e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.09073564913180734
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.039650126687170585 
    DEBUG:qutip_qtrl.tslotcomp:204 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.50159327-6.42306204e-16j]
     [-0.05712893+3.12086894e-01j]
     [-0.05712893-3.12086894e-01j]
     [ 0.49840673+6.49211321e-16j]]
    Evo final diff:
    [[-0.50159327+6.42306204e-16j]
     [ 0.05712893-3.12086894e-01j]
     [ 0.05712893+3.12086894e-01j]
     [ 0.50159327-6.49211321e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0880644368823737
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03979051992069293 
    DEBUG:qutip_qtrl.tslotcomp:203 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.49572858-3.39014563e-16j]
     [-0.05763815+3.12734644e-01j]
     [-0.05763815-3.12734644e-01j]
     [ 0.50427142+3.31741027e-16j]]
    Evo final diff:
    [[-0.49572858+3.39014563e-16j]
     [ 0.05763815-3.12734644e-01j]
     [ 0.05763815+3.12734644e-01j]
     [ 0.49572858-3.31741027e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.08671798497037973
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03983741481561637 
    DEBUG:qutip_qtrl.tslotcomp:203 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.49419982-7.39012248e-16j]
     [-0.05777526+3.12884150e-01j]
     [-0.05777526-3.12884150e-01j]
     [ 0.50580018+7.32231909e-16j]]
    Evo final diff:
    [[-0.49419982+7.39012248e-16j]
     [ 0.05777526-3.12884150e-01j]
     [ 0.05777526+3.12884150e-01j]
     [ 0.49419982-7.32231909e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0863669839333387
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03984692665362219 
    DEBUG:qutip_qtrl.tslotcomp:202 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.4883668 -4.56155982e-16j]
     [-0.05833534+3.13381498e-01j]
     [-0.05833534-3.13381498e-01j]
     [ 0.5116332 +4.61293259e-16j]]
    Evo final diff:
    [[-0.4883668 +4.56155982e-16j]
     [ 0.05833534-3.13381498e-01j]
     [ 0.05833534+3.13381498e-01j]
     [ 0.4883668 -4.61293259e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0850282766126049
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.039872746372552934 
    DEBUG:qutip_qtrl.tslotcomp:202 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.46935112-4.54854539e-16j]
     [-0.06035473+3.14198420e-01j]
     [-0.06035473-3.14198420e-01j]
     [ 0.53064888+4.24758436e-16j]]
    Evo final diff:
    [[-0.46935112+4.54854539e-16j]
     [ 0.06035473-3.14198420e-01j]
     [ 0.06035473+3.14198420e-01j]
     [ 0.46935112-4.24758436e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0806634542809313
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.03984426682584268 
    DEBUG:qutip_qtrl.tslotcomp:201 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.31559106+2.18470835e-16j]
     [-0.29201069-1.19719160e-01j]
     [-0.29201069+1.19719160e-01j]
     [ 0.68440894-2.12766239e-16j]]
    Evo final diff:
    [[-0.31559106-2.18470835e-16j]
     [ 0.29201069+1.19719160e-01j]
     [ 0.29201069-1.19719160e-01j]
     [ 0.31559106+2.12766239e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.049800159556787876
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.033387858399689184 
    DEBUG:qutip_qtrl.tslotcomp:171 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.20131376+1.90780925e-16j]
     [-0.13083837+1.38723248e-01j]
     [-0.13083837-1.38723248e-01j]
     [ 0.79868624-1.98580734e-16j]]
    Evo final diff:
    [[-0.20131376-1.90780925e-16j]
     [ 0.13083837-1.38723248e-01j]
     [ 0.13083837+1.38723248e-01j]
     [ 0.20131376+1.98580734e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.019222512288685996
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.017491353438115798 
    DEBUG:qutip_qtrl.tslotcomp:152 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.1816594 +6.02748488e-17j]
     [-0.15067819+6.84261618e-02j]
     [-0.15067819-6.84261618e-02j]
     [ 0.8183406 -4.15704533e-17j]]
    Evo final diff:
    [[-0.1816594 -6.02748488e-17j]
     [ 0.15067819-6.84261618e-02j]
     [ 0.15067819+6.84261618e-02j]
     [ 0.1816594 +4.15704533e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.015096548611431919
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.013670082182459433 
    DEBUG:qutip_qtrl.tslotcomp:156 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.16849715+2.50135878e-16j]
     [-0.13264625+3.34976180e-02j]
     [-0.13264625-3.34976180e-02j]
     [ 0.83150285-2.56424104e-16j]]
    Evo final diff:
    [[-0.16849715-2.50135878e-16j]
     [ 0.13264625-3.34976180e-02j]
     [ 0.13264625+3.34976180e-02j]
     [ 0.16849715+2.56424104e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.011777101837078275
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.011480990049517794 
    DEBUG:qutip_qtrl.tslotcomp:153 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.15493998+3.51125072e-17j]
     [-0.07813057+9.91910478e-03j]
     [-0.07813057-9.91910478e-03j]
     [ 0.84506002-4.09579686e-17j]]
    Evo final diff:
    [[-0.15493998-3.51125072e-17j]
     [ 0.07813057-9.91910478e-03j]
     [ 0.07813057+9.91910478e-03j]
     [ 0.15493998+4.09579686e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.007552292958213012
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0062833948937121315 
    DEBUG:qutip_qtrl.tslotcomp:137 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.15316377-1.43451255e-16j]
     [-0.04009852+1.57617808e-02j]
     [-0.04009852-1.57617808e-02j]
     [ 0.84683623+1.56843612e-16j]]
    Evo final diff:
    [[-0.15316377+1.43451255e-16j]
     [ 0.04009852-1.57617808e-02j]
     [ 0.04009852+1.57617808e-02j]
     [ 0.15316377-1.56843612e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.006328866328369624
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.003038745925995049 
    DEBUG:qutip_qtrl.tslotcomp:119 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.15306883-4.86970890e-18j]
     [-0.030905  +1.50597807e-02j]
     [-0.030905  -1.50597807e-02j]
     [ 0.84693117+5.39952689e-18j]]
    Evo final diff:
    [[-0.15306883+4.86970890e-18j]
     [ 0.030905  -1.50597807e-02j]
     [ 0.030905  +1.50597807e-02j]
     [ 0.15306883-5.39952689e-18j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.006152995343819839
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.002508356866757705 
    DEBUG:qutip_qtrl.tslotcomp:113 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.15257059-2.86266035e-16j]
     [-0.01992457+8.56547658e-03j]
     [-0.01992457-8.56547658e-03j]
     [ 0.84742941+2.86191075e-16j]]
    Evo final diff:
    [[-0.15257059+2.86266035e-16j]
     [ 0.01992457-8.56547658e-03j]
     [ 0.01992457+8.56547658e-03j]
     [ 0.15257059-2.86191075e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.005937035062464446
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0015587091954753587 
    DEBUG:qutip_qtrl.tslotcomp:115 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.15179587-3.25979993e-17j]
     [-0.01463539+3.39500158e-04j]
     [-0.01463539-3.39500158e-04j]
     [ 0.84820413+3.23200765e-17j]]
    Evo final diff:
    [[-0.15179587+3.25979993e-17j]
     [ 0.01463539-3.39500158e-04j]
     [ 0.01463539+3.39500158e-04j]
     [ 0.15179587-3.23200765e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.00581407385674651
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0007632515218465383 
    DEBUG:qutip_qtrl.tslotcomp:133 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.15103612-1.64565330e-16j]
     [-0.01316383-4.05847194e-03j]
     [-0.01316383+4.05847194e-03j]
     [ 0.84896388+1.74983659e-16j]]
    Evo final diff:
    [[-0.15103612+1.64565330e-16j]
     [ 0.01316383+4.05847194e-03j]
     [ 0.01316383-4.05847194e-03j]
     [ 0.15103612-1.74983659e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.005750416728963727
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0007035771752061577 
    DEBUG:qutip_qtrl.tslotcomp:146 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.14983243-2.07060409e-16j]
     [-0.01266315-6.78354613e-03j]
     [-0.01266315+6.78354613e-03j]
     [ 0.85016757+2.01010929e-16j]]
    Evo final diff:
    [[-0.14983243+2.07060409e-16j]
     [ 0.01266315+6.78354613e-03j]
     [ 0.01266315-6.78354613e-03j]
     [ 0.14983243-2.01010929e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0056640325098776785
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.000875533443023031 
    DEBUG:qutip_qtrl.tslotcomp:147 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.14736865-1.98101729e-17j]
     [-0.01395029-7.58517534e-03j]
     [-0.01395029+7.58517534e-03j]
     [ 0.85263135+2.46437306e-17j]]
    Evo final diff:
    [[-0.14736865+1.98101729e-17j]
     [ 0.01395029+7.58517534e-03j]
     [ 0.01395029-7.58517534e-03j]
     [ 0.14736865-2.46437306e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.005492416455265949
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0009792637379427893 
    DEBUG:qutip_qtrl.tslotcomp:151 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.13543044-9.90346214e-17j]
     [-0.031025  +2.26361283e-02j]
     [-0.031025  -2.26361283e-02j]
     [ 0.86456956+1.25741960e-16j]]
    Evo final diff:
    [[-0.13543044+9.90346214e-17j]
     [ 0.031025  -2.26361283e-02j]
     [ 0.031025  +2.26361283e-02j]
     [ 0.13543044-1.25741960e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.004954087169670477
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.002816855705100954 
    DEBUG:qutip_qtrl.tslotcomp:135 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.13554058-1.42501810e-16j]
     [-0.0281265 +1.65553447e-02j]
     [-0.0281265 -1.65553447e-02j]
     [ 0.86445942+1.43220318e-16j]]
    Evo final diff:
    [[-0.13554058+1.42501810e-16j]
     [ 0.0281265 -1.65553447e-02j]
     [ 0.0281265 +1.65553447e-02j]
     [ 0.13554058-1.43220318e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.004859107182026605
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0021941396970016616 
    DEBUG:qutip_qtrl.tslotcomp:131 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.13575185-3.37779973e-16j]
     [-0.02081153+3.77437686e-03j]
     [-0.02081153-3.77437686e-03j]
     [ 0.86424815+3.10875498e-16j]]
    Evo final diff:
    [[-0.13575185+3.37779973e-16j]
     [ 0.02081153-3.77437686e-03j]
     [ 0.02081153+3.77437686e-03j]
     [ 0.13575185-3.10875498e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0047189827899690034
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0010073578183123135 
    DEBUG:qutip_qtrl.tslotcomp:130 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.13530186-7.27119118e-17j]
     [-0.01764337+5.43691276e-04j]
     [-0.01764337-5.43691276e-04j]
     [ 0.86469814+6.12490329e-17j]]
    Evo final diff:
    [[-0.13530186+7.27119118e-17j]
     [ 0.01764337-5.43691276e-04j]
     [ 0.01764337+5.43691276e-04j]
     [ 0.13530186-6.12490329e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.004654544041536687
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0007592681905472547 
    DEBUG:qutip_qtrl.tslotcomp:126 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.13403186+6.86479517e-17j]
     [-0.01393748+1.08562636e-04j]
     [-0.01393748-1.08562636e-04j]
     [ 0.86596814-7.29814388e-17j]]
    Evo final diff:
    [[-0.13403186-6.86479517e-17j]
     [ 0.01393748-1.08562636e-04j]
     [ 0.01393748+1.08562636e-04j]
     [ 0.13403186+7.29814388e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0045397014376240725
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0005692317333776652 
    DEBUG:qutip_qtrl.tslotcomp:125 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 1.27670784e-01-2.16234204e-16j]
     [-7.46907037e-04+2.71856059e-02j]
     [-7.46907037e-04-2.71856059e-02j]
     [ 8.72329216e-01+2.12601372e-16j]]
    Evo final diff:
    [[-0.12767078+2.16234204e-16j]
     [ 0.00074691-2.71856059e-02j]
     [ 0.00074691+2.71856059e-02j]
     [ 0.12767078-2.12601372e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.004259861039623093
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0036164830842216594 
    DEBUG:qutip_qtrl.tslotcomp:143 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.12674848-2.02792245e-16j]
     [-0.00620768+1.95245106e-02j]
     [-0.00620768-1.95245106e-02j]
     [ 0.87325152+2.07998536e-16j]]
    Evo final diff:
    [[-0.12674848+2.02792245e-16j]
     [ 0.00620768-1.95245106e-02j]
     [ 0.00620768+1.95245106e-02j]
     [ 0.12674848-2.07998536e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.004121229805969434
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.002423605818773954 
    DEBUG:qutip_qtrl.tslotcomp:142 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.12513358-2.73738416e-16j]
     [-0.0156496 +7.86097283e-03j]
     [-0.0156496 -7.86097283e-03j]
     [ 0.87486642+2.70488348e-16j]]
    Evo final diff:
    [[-0.12513358+2.73738416e-16j]
     [ 0.0156496 -7.86097283e-03j]
     [ 0.0156496 +7.86097283e-03j]
     [ 0.12513358-2.70488348e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0039912794285803434
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.000918881455787781 
    DEBUG:qutip_qtrl.tslotcomp:136 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.12454703-1.08713181e-16j]
     [-0.01718828+5.79852866e-03j]
     [-0.01718828-5.79852866e-03j]
     [ 0.87545297+1.21106065e-16j]]
    Evo final diff:
    [[-0.12454703+1.08713181e-16j]
     [ 0.01718828-5.79852866e-03j]
     [ 0.01718828+5.79852866e-03j]
     [ 0.12454703-1.21106065e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0039602556103499175
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0008779527858675085 
    DEBUG:qutip_qtrl.tslotcomp:132 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.1236505 -1.23465683e-16j]
     [-0.01811325+3.97205392e-03j]
     [-0.01811325-3.97205392e-03j]
     [ 0.8763495 +1.24040755e-16j]]
    Evo final diff:
    [[-0.1236505 +1.23465683e-16j]
     [ 0.01811325-3.97205392e-03j]
     [ 0.01811325+3.97205392e-03j]
     [ 0.1236505 -1.24040755e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.003908328371381658
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0009257862736929979 
    DEBUG:qutip_qtrl.tslotcomp:128 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.11777994-2.76891385e-16j]
     [-0.01467044+1.94547748e-02j]
     [-0.01467044-1.94547748e-02j]
     [ 0.88222006+2.87209455e-16j]]
    Evo final diff:
    [[-0.11777994+2.76891385e-16j]
     [ 0.01467044-1.94547748e-02j]
     [ 0.01467044+1.94547748e-02j]
     [ 0.11777994-2.87209455e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0036164558427296887
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0017534099117774652 
    DEBUG:qutip_qtrl.tslotcomp:110 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.11755998+5.74296773e-18j]
     [-0.01423278+1.59465433e-02j]
     [-0.01423278-1.59465433e-02j]
     [ 0.88244002+9.24094171e-19j]]
    Evo final diff:
    [[-0.11755998-5.74296773e-18j]
     [ 0.01423278-1.59465433e-02j]
     [ 0.01423278+1.59465433e-02j]
     [ 0.11755998-9.24094171e-19j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0035693033154546795
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0014386769254665266 
    DEBUG:qutip_qtrl.tslotcomp:105 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.1166713 -1.45522484e-17j]
     [-0.01310825+9.04460212e-04j]
     [-0.01310825-9.04460212e-04j]
     [ 0.8833287 +4.77845036e-17j]]
    Evo final diff:
    [[-0.1166713 +1.45522484e-17j]
     [ 0.01310825-9.04460212e-04j]
     [ 0.01310825+9.04460212e-04j]
     [ 0.1166713 -4.77845036e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0034462091781149434
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0007094621487583083 
    DEBUG:qutip_qtrl.tslotcomp:114 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.1157596 -1.06205488e-17j]
     [-0.00838469+6.11512969e-03j]
     [-0.00838469-6.11512969e-03j]
     [ 0.8842404 +1.63217820e-17j]]
    Evo final diff:
    [[-0.1157596 +1.06205488e-17j]
     [ 0.00838469-6.11512969e-03j]
     [ 0.00838469+6.11512969e-03j]
     [ 0.1157596 -1.63217820e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0033769958899527034
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0006002112724568193 
    DEBUG:qutip_qtrl.tslotcomp:124 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.11540556-1.05495365e-16j]
     [-0.00723593+6.14208109e-03j]
     [-0.00723593-6.14208109e-03j]
     [ 0.88459444+1.28837745e-16j]]
    Evo final diff:
    [[-0.11540556+1.05495365e-16j]
     [ 0.00723593-6.14208109e-03j]
     [ 0.00723593+6.14208109e-03j]
     [ 0.11540556-1.28837745e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.003352131968892376
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0006206540278559121 
    DEBUG:qutip_qtrl.tslotcomp:127 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.11730567-6.28545002e-17j]
     [0.04132207+5.54769529e-02j]
     [0.04132207-5.54769529e-02j]
     [0.88269433+5.88631082e-17j]]
    Evo final diff:
    [[-0.11730567+6.28545002e-17j]
     [-0.04132207-5.54769529e-02j]
     [-0.04132207+5.54769529e-02j]
     [ 0.11730567-5.88631082e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.004636456654879235
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.008466177084199153 
    DEBUG:qutip_qtrl.tslotcomp:127 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.11479238-2.16773149e-16j]
     [-0.00190573+1.16936871e-02j]
     [-0.00190573-1.16936871e-02j]
     [ 0.88520762+1.96527670e-16j]]
    Evo final diff:
    [[-0.11479238+2.16773149e-16j]
     [ 0.00190573-1.16936871e-02j]
     [ 0.00190573+1.16936871e-02j]
     [ 0.11479238-1.96527670e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0033294161351953753
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00140652457734953 
    DEBUG:qutip_qtrl.tslotcomp:161 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.11345567-2.64803713e-16j]
     [0.02230965+3.05856202e-02j]
     [0.02230965-3.05856202e-02j]
     [0.88654433+2.51732420e-16j]]
    Evo final diff:
    [[-0.11345567+2.64803713e-16j]
     [-0.02230965-3.05856202e-02j]
     [-0.02230965+3.05856202e-02j]
     [ 0.11345567-2.51732420e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.003576347575750469
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004832367011452715 
    DEBUG:qutip_qtrl.tslotcomp:161 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.11434582-1.97546972e-16j]
     [0.0014966 +1.44532218e-02j]
     [0.0014966 -1.44532218e-02j]
     [0.88565418+1.82179483e-16j]]
    Evo final diff:
    [[-0.11434582+1.97546972e-16j]
     [-0.0014966 -1.44532218e-02j]
     [-0.0014966 +1.44532218e-02j]
     [ 0.11434582-1.82179483e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0033215252734320284
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0018847109570573012 
    DEBUG:qutip_qtrl.tslotcomp:165 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.11311129-1.78769439e-16j]
     [0.0192881 +2.70178745e-02j]
     [0.0192881 -2.70178745e-02j]
     [0.88688871+1.80945132e-16j]]
    Evo final diff:
    [[-0.11311129+1.78769439e-16j]
     [-0.0192881 -2.70178745e-02j]
     [-0.0192881 +2.70178745e-02j]
     [ 0.11311129-1.80945132e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.003474040083892313
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.004284021748206093 
    DEBUG:qutip_qtrl.tslotcomp:165 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.11415438-2.96243998e-16j]
     [0.00294656+1.55132375e-02j]
     [0.00294656-1.55132375e-02j]
     [0.88584562+3.04878725e-16j]]
    Evo final diff:
    [[-0.11415438+2.96243998e-16j]
     [-0.00294656-1.55132375e-02j]
     [-0.00294656+1.55132375e-02j]
     [ 0.11415438-3.04878725e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0033201410759716036
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0020800187801302576 
    DEBUG:qutip_qtrl.tslotcomp:165 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[1.13780965e-01-9.45985399e-17j]
     [3.83794035e-04+1.30577224e-02j]
     [3.83794035e-04-1.30577224e-02j]
     [8.86219035e-01+1.12315878e-16j]]
    Evo final diff:
    [[-0.11378096+9.45985399e-17j]
     [-0.00038379-1.30577224e-02j]
     [-0.00038379+1.30577224e-02j]
     [ 0.11378096-1.12315878e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.003279189843179375
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0016500071806924944 
    DEBUG:qutip_qtrl.tslotcomp:159 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.11140081-3.26953161e-16j]
     [0.00983386+2.28050448e-02j]
     [0.00983386-2.28050448e-02j]
     [0.88859919+3.21754966e-16j]]
    Evo final diff:
    [[-0.11140081+3.26953161e-16j]
     [-0.00983386-2.28050448e-02j]
     [-0.00983386+2.28050448e-02j]
     [ 0.11140081-3.21754966e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0032567287401730326
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0030620299087130392 
    DEBUG:qutip_qtrl.tslotcomp:143 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.11027288-2.60080742e-16j]
     [-0.00211515+1.01988537e-02j]
     [-0.00211515-1.01988537e-02j]
     [ 0.88972712+2.51328848e-16j]]
    Evo final diff:
    [[-0.11027288+2.60080742e-16j]
     [ 0.00211515-1.01988537e-02j]
     [ 0.00211515+1.01988537e-02j]
     [ 0.11027288-2.51328848e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.003067149372482811
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0010635270647882651 
    DEBUG:qutip_qtrl.tslotcomp:142 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10986347-1.52044276e-16j]
     [-0.00676178+3.81635360e-03j]
     [-0.00676178-3.81635360e-03j]
     [ 0.89013653+1.84026165e-16j]]
    Evo final diff:
    [[-0.10986347+1.52044276e-16j]
     [ 0.00676178-3.81635360e-03j]
     [ 0.00676178+3.81635360e-03j]
     [ 0.10986347-1.84026165e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.00303256689509992
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00043067781094844736 
    DEBUG:qutip_qtrl.tslotcomp:156 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10966187-1.51373545e-16j]
     [-0.00722021+2.20990232e-03j]
     [-0.00722021-2.20990232e-03j]
     [ 0.89033813+1.81611675e-16j]]
    Evo final diff:
    [[-0.10966187+1.51373545e-16j]
     [ 0.00722021-2.20990232e-03j]
     [ 0.00722021+2.20990232e-03j]
     [ 0.10966187-1.81611675e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0030206850268025697
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0004206778835279385 
    DEBUG:qutip_qtrl.tslotcomp:144 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10881918-1.05117998e-16j]
     [-0.00818667-9.98658967e-04j]
     [-0.00818667+9.98658967e-04j]
     [ 0.89118082+8.63042319e-17j]]
    Evo final diff:
    [[-0.10881918+1.05117998e-16j]
     [ 0.00818667+9.98658967e-04j]
     [ 0.00818667-9.98658967e-04j]
     [ 0.10881918-8.63042319e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002977408062166688
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0006179572985145115 
    DEBUG:qutip_qtrl.tslotcomp:150 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.10893369-5.09188722e-17j]
     [0.01915619+3.34754429e-02j]
     [0.01915619-3.34754429e-02j]
     [0.89106631+6.30446511e-17j]]
    Evo final diff:
    [[-0.10893369+5.09188722e-17j]
     [-0.01915619-3.34754429e-02j]
     [-0.01915619+3.34754429e-02j]
     [ 0.10893369-6.30446511e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0033385283684678937
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00440432211260146 
    DEBUG:qutip_qtrl.tslotcomp:150 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.1082078 -7.62810704e-18j]
     [-0.00188368+6.88378461e-03j]
     [-0.00188368-6.88378461e-03j]
     [ 0.8917922 -2.12907249e-17j]]
    Evo final diff:
    [[-0.1082078 +7.62810704e-18j]
     [ 0.00188368-6.88378461e-03j]
     [ 0.00188368+6.88378461e-03j]
     [ 0.1082078 +2.12907249e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0029399658007447775
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0007275766472791512 
    DEBUG:qutip_qtrl.tslotcomp:150 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10752116-2.11858619e-16j]
     [-0.00168463+6.00524977e-03j]
     [-0.00168463-6.00524977e-03j]
     [ 0.89247884+1.98456153e-16j]]
    Evo final diff:
    [[-0.10752116+2.11858619e-16j]
     [ 0.00168463-6.00524977e-03j]
     [ 0.00168463+6.00524977e-03j]
     [ 0.10752116-1.98456153e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002899925081967757
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0006452058698532823 
    DEBUG:qutip_qtrl.tslotcomp:130 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.10569875-3.00077911e-16j]
     [0.0059499 +1.39251675e-02j]
     [0.0059499 -1.39251675e-02j]
     [0.89430125+3.16129435e-16j]]
    Evo final diff:
    [[-0.10569875+3.00077911e-16j]
     [-0.0059499 -1.39251675e-02j]
     [-0.0059499 +1.39251675e-02j]
     [ 0.10569875-3.16129435e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002850384225861206
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0016855236265835014 
    DEBUG:qutip_qtrl.tslotcomp:132 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10497416-7.39664158e-17j]
     [-0.00243508+3.63835198e-03j]
     [-0.00243508-3.63835198e-03j]
     [ 0.89502584+6.54204692e-17j]]
    Evo final diff:
    [[-0.10497416+7.39664158e-17j]
     [ 0.00243508-3.63835198e-03j]
     [ 0.00243508+3.63835198e-03j]
     [ 0.10497416-6.54204692e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002759685324237138
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.000435543446037828 
    DEBUG:qutip_qtrl.tslotcomp:134 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10479938-1.17661387e-17j]
     [-0.00446064+1.59815368e-04j]
     [-0.00446064-1.59815368e-04j]
     [ 0.89520062+3.25578489e-17j]]
    Evo final diff:
    [[-0.10479938+1.17661387e-17j]
     [ 0.00446064-1.59815368e-04j]
     [ 0.00446064+1.59815368e-04j]
     [ 0.10479938-3.25578489e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002750708233412382
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0004896581077905727 
    DEBUG:qutip_qtrl.tslotcomp:202 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.10436742-6.90043500e-17j]
     [0.00328362+5.30833354e-03j]
     [0.00328362-5.30833354e-03j]
     [0.89563258+6.90910003e-17j]]
    Evo final diff:
    [[-0.10436742+6.90043500e-17j]
     [-0.00328362-5.30833354e-03j]
     [-0.00328362+5.30833354e-03j]
     [ 0.10436742-6.90910003e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0027328799606105027
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.000760130975106369 
    DEBUG:qutip_qtrl.tslotcomp:198 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 1.04107186e-01-9.29898879e-17j]
     [-3.96115525e-04-5.37137207e-04j]
     [-3.96115525e-04+5.37137207e-04j]
     [ 8.95892814e-01+1.13300683e-16j]]
    Evo final diff:
    [[-0.10410719+9.29898879e-17j]
     [ 0.00039612+5.37137207e-04j]
     [ 0.00039612-5.37137207e-04j]
     [ 0.10410719-1.13300683e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0027096879175113045
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0003222966255689542 
    DEBUG:qutip_qtrl.tslotcomp:202 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10399701+1.78748013e-16j]
     [-0.00143715-2.01419268e-03j]
     [-0.00143715+2.01419268e-03j]
     [ 0.89600299-1.71911082e-16j]]
    Evo final diff:
    [[-0.10399701-1.78748013e-16j]
     [ 0.00143715+2.01419268e-03j]
     [ 0.00143715-2.01419268e-03j]
     [ 0.10399701+1.71911082e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0027053751425556963
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00040868454197329196 
    DEBUG:qutip_qtrl.tslotcomp:206 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[1.03602593e-01+7.41523511e-17j]
     [7.76199379e-04-3.33318139e-03j]
     [7.76199379e-04+3.33318139e-03j]
     [8.96397407e-01-6.98430227e-17j]]
    Evo final diff:
    [[-0.10360259-7.41523511e-17j]
     [-0.0007762 +3.33318139e-03j]
     [-0.0007762 -3.33318139e-03j]
     [ 0.10360259+6.98430227e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0026863024750416607
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00039887171896492055 
    DEBUG:qutip_qtrl.tslotcomp:212 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[1.02801167e-01+3.76611693e-17j]
     [8.77013768e-04+4.08904284e-03j]
     [8.77013768e-04-4.08904284e-03j]
     [8.97198833e-01-5.32629335e-17j]]
    Evo final diff:
    [[-0.10280117-3.76611693e-17j]
     [-0.00087701-4.08904284e-03j]
     [-0.00087701+4.08904284e-03j]
     [ 0.10280117+5.32629335e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0026463923436669114
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0005547734083537043 
    DEBUG:qutip_qtrl.tslotcomp:208 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.10190409-1.15163006e-16j]
     [-0.00172282-1.78044665e-04j]
     [-0.00172282+1.78044665e-04j]
     [ 0.89809591+1.24903520e-16j]]
    Evo final diff:
    [[-0.10190409+1.15163006e-16j]
     [ 0.00172282+1.78044665e-04j]
     [ 0.00172282-1.78044665e-04j]
     [ 0.10190409-1.24903520e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002596860968248521
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0005226078214042328 
    DEBUG:qutip_qtrl.tslotcomp:206 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.09794285+3.98467394e-17j]
     [0.00184585-1.04772205e-03j]
     [0.00184585+1.04772205e-03j]
     [0.90205715-3.81075707e-17j]]
    Evo final diff:
    [[-0.09794285-3.98467394e-17j]
     [-0.00184585+1.04772205e-03j]
     [-0.00184585-1.04772205e-03j]
     [ 0.09794285+3.81075707e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002399326797379077
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00043545868627036346 
    DEBUG:qutip_qtrl.tslotcomp:199 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.09581463+1.18848982e-16j]
     [-0.00724779-6.72702891e-03j]
     [-0.00724779+6.72702891e-03j]
     [ 0.90418537-1.24877144e-16j]]
    Evo final diff:
    [[-0.09581463-1.18848982e-16j]
     [ 0.00724779+6.72702891e-03j]
     [ 0.00724779-6.72702891e-03j]
     [ 0.09581463+1.24877144e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002319556861186033
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0013023651442190495 
    DEBUG:qutip_qtrl.tslotcomp:206 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.09053693+7.21333751e-17j]
     [-0.01521012+2.06478190e-03j]
     [-0.01521012-2.06478190e-03j]
     [ 0.90946307-5.38625646e-17j]]
    Evo final diff:
    [[-0.09053693-7.21333751e-17j]
     [ 0.01521012-2.06478190e-03j]
     [ 0.01521012+2.06478190e-03j]
     [ 0.09053693+5.38625646e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.002108136579583669
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0013974311451801593 
    DEBUG:qutip_qtrl.tslotcomp:204 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.08740393+1.13699959e-16j]
     [0.02516675+7.75704131e-03j]
     [0.02516675-7.75704131e-03j]
     [0.91259607-1.14147599e-16j]]
    Evo final diff:
    [[-0.08740393-1.13699959e-16j]
     [-0.02516675-7.75704131e-03j]
     [-0.02516675+7.75704131e-03j]
     [ 0.08740393+1.14147599e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0020832461766792203
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.002506533779775255 
    DEBUG:qutip_qtrl.tslotcomp:204 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[8.86222058e-02+4.97268253e-17j]
     [2.56588382e-04+4.76433737e-03j]
     [2.56588382e-04-4.76433737e-03j]
     [9.11377794e-01-4.72413309e-17j]]
    Evo final diff:
    [[-0.08862221-4.97268253e-17j]
     [-0.00025659-4.76433737e-03j]
     [-0.00025659+4.76433737e-03j]
     [ 0.08862221+4.72413309e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0019691650267841965
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0005952728968143743 
    DEBUG:qutip_qtrl.tslotcomp:204 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.08495534-3.29007930e-17j]
     [0.01528129-5.66222785e-03j]
     [0.01528129+5.66222785e-03j]
     [0.91504466+3.46683616e-17j]]
    Evo final diff:
    [[-0.08495534+3.29007930e-17j]
     [-0.01528129+5.66222785e-03j]
     [-0.01528129-5.66222785e-03j]
     [ 0.08495534-3.46683616e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0018707471382189164
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.001120846566928445 
    DEBUG:qutip_qtrl.tslotcomp:197 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.08263591-1.08670839e-16j]
     [-0.0021057 -8.19285612e-03j]
     [-0.0021057 +8.19285612e-03j]
     [ 0.91736409+1.20622806e-16j]]
    Evo final diff:
    [[-0.08263591+1.08670839e-16j]
     [ 0.0021057 +8.19285612e-03j]
     [ 0.0021057 -8.19285612e-03j]
     [ 0.08263591-1.20622806e-16j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.001725062577736425
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0010739051187731142 
    DEBUG:qutip_qtrl.tslotcomp:198 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.0807071 +3.36395492e-17j]
     [-0.00798595-3.76762132e-03j]
     [-0.00798595+3.76762132e-03j]
     [ 0.9192929 -3.54514155e-17j]]
    Evo final diff:
    [[-0.0807071 -3.36395492e-17j]
     [ 0.00798595+3.76762132e-03j]
     [ 0.00798595-3.76762132e-03j]
     [ 0.0807071 +3.54514155e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0016479017720151489
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0010189445700085143 
    DEBUG:qutip_qtrl.tslotcomp:198 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.07375042+6.28185036e-17j]
     [-0.01443816+1.75836628e-02j]
     [-0.01443816-1.75836628e-02j]
     [ 0.92624958-5.14006535e-17j]]
    Evo final diff:
    [[-0.07375042-6.28185036e-17j]
     [ 0.01443816-1.75836628e-02j]
     [ 0.01443816+1.75836628e-02j]
     [ 0.07375042+5.14006535e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0014891926475058604
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.001541352580157223 
    DEBUG:qutip_qtrl.tslotcomp:207 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.07304985-5.66968551e-17j]
     [-0.00896955+8.59571602e-03j]
     [-0.00896955-8.59571602e-03j]
     [ 0.92695015+4.40306800e-17j]]
    Evo final diff:
    [[-0.07304985+5.66968551e-17j]
     [ 0.00896955-8.59571602e-03j]
     [ 0.00896955+8.59571602e-03j]
     [ 0.07304985-4.40306800e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0013726548989253327
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0007931689912542538 
    DEBUG:qutip_qtrl.tslotcomp:208 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.07120002-5.50391068e-17j]
     [0.00097728-4.14710189e-03j]
     [0.00097728+4.14710189e-03j]
     [0.92879998+5.72663573e-17j]]
    Evo final diff:
    [[-0.07120002+5.50391068e-17j]
     [-0.00097728+4.14710189e-03j]
     [-0.00097728-4.14710189e-03j]
     [ 0.07120002-5.72663573e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0012718991600032883
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0005914141449148123 
    DEBUG:qutip_qtrl.tslotcomp:213 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[0.06815425-7.70364037e-17j]
     [0.00448965+1.11383284e-03j]
     [0.00448965-1.11383284e-03j]
     [0.93184575+8.04728568e-17j]]
    Evo final diff:
    [[-0.06815425+7.70364037e-17j]
     [-0.00448965-1.11383284e-03j]
     [-0.00448965+1.11383284e-03j]
     [ 0.06815425-8.04728568e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0011665998232497808
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.00059355174364513 
    DEBUG:qutip_qtrl.tslotcomp:216 amplitudes changed
    Level 8:qutip_qtrl.dynamics:Computing evolution
    Level 8:qutip_qtrl.fidcomp:Calculating TraceDiff fidelity...
     Target:
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [1.+0.j]]
     Evo final:
    [[ 0.05542137+7.22898608e-17j]
     [-0.00407353+1.49163418e-02j]
     [-0.00407353-1.49163418e-02j]
     [ 0.94457863-6.48865312e-17j]]
    Evo final diff:
    [[-0.05542137-7.22898608e-17j]
     [ 0.00407353-1.49163418e-02j]
     [ 0.00407353+1.49163418e-02j]
     [ 0.05542137+6.48865312e-17j]]
    DEBUG:qutip_qtrl.fidcomp:Fidelity error: 0.0008276547544061342
    DEBUG:qutip_qtrl.tslotcomp:No amplitudes changed
    DEBUG:qutip_qtrl.fidcomp:Gradient norm: 0.0015358534376677783 
    

    Infidelity:  0.0008276547544061342
    


    
![png](GRAPE_state_open_files/GRAPE_state_open_7_2.png)
    



```python
H_result = [Hd, [Hc, res_grape.optimized_controls[0]]]
evolution = qt.mesolve(H_result, initial_state, times, c_ops)

plt.plot(times, [np.abs(state.overlap(target_state)) for state in evolution.states])
plt.plot(times, [qt.fidelity(state, target_state) for state in evolution.states], '--', label="Fidelity")
plt.title("CRAB performance")
plt.xlabel('Time')
plt.legend()
plt.ylim(0, 1)
plt.show()
```


    
![png](GRAPE_state_open_files/GRAPE_state_open_8_0.png)
    


## Validation


```python
assert res_grape.infidelity < 0.01
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
