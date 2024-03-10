import qutip as qt
import numpy as np
import qutip_qoc as qoc

# Identity ---> Hadamard
initial = qt.qeye(2)
target = qt.gates.hadamard_transform()

# convert to superoperator
initial = qt.sprepost(initial, initial.dag())
target = qt.sprepost(target, target.dag())

# control Hamiltonian
sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
Hc = [sx, sy, sz]
Hc = [qt.liouvillian(H) for H in Hc]

# energy splitting, tunneling, amplitude damping
omega, delta, gamma = 0.1, 1.0, 0.1

# drift Hamiltonian
Hd = 1 / 2 * (omega * sz + delta * sx)  # drift
Hd = qt.liouvillian(H=Hd, c_ops=[np.sqrt(gamma) * qt.sigmam()])
# full Hamiltonian
H = [Hd, Hc[0], Hc[1], Hc[2]]

# time interval
pi, N = 3.14, 100
interval = qoc.TimeInterval(evo_time=2 * pi, n_tslots=N)
# initial control amplitudes
initial_x = np.sin(interval())
initial_y = np.cos(interval())
initial_z = np.tan(interval())

res_grape = qoc.optimize_pulses(
    objectives=[qoc.Objective(initial, H, target)],
    pulse_options={
        "ctrl_x": {"guess": initial_x, "bounds": [-1, 1]},
        "ctrl_y": {"guess": initial_y, "bounds": [-1, 1]},
        "ctrl_z": {"guess": initial_z, "bounds": [-1, 1]},
    },
    time_interval=interval,
    algorithm_kwargs={
        "alg": "GRAPE",
        "fid_err_targ": 1,
    },
)

print(res_grape)

# initial control amplitudes
initial_x = np.sin(interval())
initial_y = np.cos(interval())
initial_z = np.tan(interval())

n_coeffs = 4
res_grape = qoc.optimize_pulses(
    objectives=qoc.Objective(initial, H, target),
    pulse_options={
        "ctrl_x": {
            "guess": [0.5 for _ in range(3 * n_coeffs)],
            "bounds": [(-1, 1) for _ in range(3 * n_coeffs)],
        },
        "ctrl_y": {
            "guess": [0.5 for _ in range(3 * n_coeffs)],
            "bounds": [(-1, 1) for _ in range(3 * n_coeffs)],
        },
        "ctrl_z": {
            "guess": [0.5 for _ in range(3 * n_coeffs)],
            "bounds": [(-1, 1) for _ in range(3 * n_coeffs)],
        },
    },
    time_interval=interval,
    algorithm_kwargs={
        "alg": "CRAB",
        "fid_err_targ": 0.5,
        "fix_frequency": False,
    },
)

print(res_grape.final_states[-1])
