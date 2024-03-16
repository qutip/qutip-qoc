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

# res_grape = qoc.optimize_pulses(
# objectives=qoc.Objective(initial, H, target),
# pulse_options={
#        "ctrl_x": {
#            "guess": initial_x,
#            "bounds": [-1, 1]},
#        "ctrl_y": {
#            "guess": initial_y,
#            "bounds": [-1, 1]},
#        "ctrl_z": {
#            "guess": initial_z,
#            "bounds": [-1, 1]},
#    },
#    time_interval=interval,
#    algorithm_kwargs={
#        "alg": "CRAB",
#        "fid_err_targ": 0.01,
#    })


# number of control parameters (multiple of 3)
n_coeffs = 3  # c0 * sin(c2*t) + c1 * cos(c2*t)

# res_crab = qoc.optimize_pulses(
# objectives=qoc.Objective(initial, H, target),
# pulse_options={
#  "ctrl_x": {
#   "guess":  [0 for _ in range(n_coeffs)],
#   "bounds": [(-1, 1) for _ in range(n_coeffs)],
#  },
#  "ctrl_y": {
#   "guess":  [0 for _ in range(n_coeffs)],
#   "bounds": [(-1, 1) for _ in range(n_coeffs)],
#  },
#  "ctrl_z": {
#   "guess":  [0 for _ in range(n_coeffs)],
#   "bounds": [(-1, 1) for _ in range(n_coeffs)],
#  },
# },
# time_interval=interval,
# algorithm_kwargs={
#  "alg": "CRAB",
#  "fid_err_targ": 0.1,
#  "fix_frequency": False,
# },
# )


def sin(t, c):
    return c[0] * np.sin(c[1] * t)


def grad_sin(t, c, idx):
    if idx == 0:  # w.r.t. c0
        return np.sin(c[1] * t)
    if idx == 1:  # w.r.t. c1
        return c[0] * np.cos(c[1] * t) * t
    if idx == 2:  # w.r.t. time
        return c[0] * np.cos(c[1] * t) * c[1]


sin_x = lambda t, c: sin(t, c)
sin_y = lambda t, d: sin(t, d)
sin_z = lambda t, e: sin(t, e)

H = [Hd] + [
    [Hc[0], sin_x, {"grad": grad_sin}],
    [Hc[1], sin_y, {"grad": grad_sin}],
    [Hc[2], sin_z, {"grad": grad_sin}],
]

# ‚res_goat = qoc.optimize_pulses(
# ‚ objectives=qoc.Objective(initial, H, target),
# ‚ pulse_options={
# ‚     id: {
# ‚         "guess":  [0 , 0],
# ‚         "bounds": [(-1, 1),(0, 2*pi)],
# ‚     } for id in range(len(Hc))
# ‚ },
# ‚ time_interval=interval,
# ‚ time_options={
# ‚     "guess": 1 / 2 * interval.evo_time,
# ‚     "bounds": (0, interval.evo_time),
# ‚ },
# ‚ algorithm_kwargs={
# ‚     "alg": "GOAT",
# ‚     "fid_err_targ": 0.1,
# ‚ },
# ‚)

import jax


@jax.jit
def sin_x(t, c, **kwargs):
    return c[0] * jax.numpy.sin(c[1] * t)


@jax.jit
def sin_y(t, d, **kwargs):
    return d[0] * jax.numpy.sin(d[1] * t)


@jax.jit
def sin_z(t, e, **kwargs):
    return e[0] * jax.numpy.sin(e[1] * t)


H = [Hd] + [[Hc[0], sin_x], [Hc[1], sin_y], [Hc[2], sin_z]]

res_jopt = qoc.optimize_pulses(
    objectives=qoc.Objective(initial, H, target),
    pulse_options={
        id: {
            "guess": [0, 0],
            "bounds": [(-1, 1), (0, 2 * pi)],
        }
        for id in range(len(Hc))
    },
    time_interval=interval,
    time_options={
        "guess": 1 / 2 * interval.evo_time,
        "bounds": (0, interval.evo_time),
    },
    algorithm_kwargs={
        "alg": "JOPT",
        "fid_err_targ": 0.1,
    },
)

print(res_jopt)
