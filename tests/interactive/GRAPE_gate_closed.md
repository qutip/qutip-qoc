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

# GRAPE algorithm for a closed system (gate synthesis)

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import gates, qeye, sigmax, sigmay, sigmaz
import qutip as qt
from qutip_qoc import Objective, optimize_pulses

def fidelity(gate, target_gate):
    """
    Fidelity used for unitary gates in qutip-qtrl and qutip-qoc
    """
    return np.abs(gate.overlap(target_gate) / target_gate.norm())
```

## Problem setup

```python
omega = 0.1  # energy splitting
sx, sy, sz = sigmax(), sigmay(), sigmaz()

Hd = 1 / 2 * omega * sz
Hc = [sx, sy, sz]
H = [Hd, Hc[0], Hc[1], Hc[2]]

# objective for optimization
initial_gate = qeye(2)
target_gate = gates.hadamard_transform()

times = np.linspace(0, np.pi / 2, 250)
```

## Guess

```python
guess_pulse_x = np.sin(times)
guess_pulse_y = np.cos(times)
guess_pulse_z = np.tanh(times)

H_guess = [Hd, [Hc[0], guess_pulse_x], [Hc[1], guess_pulse_y], [Hc[2], guess_pulse_z]]
evolution_guess = qt.sesolve(H_guess, initial_gate, times)

print('Fidelity: ', fidelity(evolution_guess.states[-1], target_gate))

plt.plot(times, [fidelity(gate, initial_gate) for gate in evolution_guess.states], label="Overlap with initial gate")
plt.plot(times, [fidelity(gate, target_gate) for gate in evolution_guess.states], label="Overlap with target gate")
plt.title("Guess performance")
plt.xlabel('Time')
plt.legend()
plt.show()
```

## GRAPE algorithm

```python
control_params = {
    "ctrl_x": {"guess": guess_pulse_x, "bounds": [-1, 1]},
    "ctrl_y": {"guess": guess_pulse_y, "bounds": [-1, 1]},
    "ctrl_z": {"guess": guess_pulse_z, "bounds": [-1, 1]},
}

res_grape = optimize_pulses(
    objectives = Objective(initial_gate, H, target_gate),
    control_parameters = control_params,
    tlist = times,
    algorithm_kwargs = {
        "alg": "GRAPE",
        "fid_err_targ": 0.001
    },
)

print('Infidelity: ', res_grape.infidelity)

plt.plot(times, guess_pulse_x, 'b--', label='guess pulse sx')
plt.plot(times, res_grape.optimized_controls[0], 'b', label='optimized pulse sx')
plt.plot(times, guess_pulse_y, 'g--', label='guess pulse sy')
plt.plot(times, res_grape.optimized_controls[1], 'g', label='optimized pulse sy')
plt.plot(times, guess_pulse_z, 'r--', label='guess pulse sz')
plt.plot(times, res_grape.optimized_controls[2], 'r', label='optimized pulse sz')
plt.title('GRAPE pulses')
plt.xlabel('Time')
plt.ylabel('Pulse amplitude')
plt.legend()
plt.show()
```

```python
H_result = [Hd,
            [Hc[0], np.array(res_grape.optimized_controls[0])],
            [Hc[1], np.array(res_grape.optimized_controls[1])],
            [Hc[2], np.array(res_grape.optimized_controls[2])]]
evolution = qt.sesolve(H_result, initial_gate, times)

plt.plot(times, [fidelity(gate, initial_gate) for gate in evolution.states], label="Overlap with initial gate")
plt.plot(times, [fidelity(gate, target_gate) for gate in evolution.states], label="Overlap with target gate")

plt.title('GRAPE performance')
plt.xlabel('Time')
plt.legend()
plt.show()
```

## Validation

```python
assert res_grape.infidelity < 0.001
assert np.allclose(fidelity(evolution.states[-1], target_gate), 1 - res_grape.infidelity, atol=1e-3)
```

```python
qt.about()
```
