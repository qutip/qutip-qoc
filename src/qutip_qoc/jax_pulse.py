import jax
import jax.numpy as jnp


class Pulse:
    """
    Base class for pulse generation.
    """

    def __init__(self, n_sup, n_var):
        self.n_sup = n_sup  # num of superpositions, 'm' in paper
        self.n_var = n_var  # num of paras for each control function
        self.n_par = n_sup * n_var  # total num of paras
        self.pulse_vmap = jax.vmap(self.pulse, in_axes=(None, 0))

    def __call__(self, t, paras):
        return self.gen_pulse(t, paras)

    def gen_pulse(self, time, paras):
        """
        Generate a supposition of pulses.
        A set of self.n_var parameters defines a single pulse.
        Summing over self.n_sup pulses gives the total pulse.
        """
        paras = jnp.reshape(paras, (self.n_sup, self.n_var))
        pulse = jnp.sum(self.pulse_vmap(time, paras), axis=0)
        return pulse

    def gen_grad(self, time, paras, idx):
        """
        Calculate the derivative of gen_pulse w.r.t. idx-th parameter.
        """
        dt, dp = jax.grad(self.gen_pulse, argnums=(0, 1))(time, paras)
        return jnp.concatenate((dp, dt), axis=None)[idx]


class SinPulse(Pulse):
    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var=3)

    def pulse(self, time, paras):
        """
        Generate a pulse with sinusoidal shape.
        """
        return paras[0] * jnp.sin(paras[1] * time + paras[2])


class CosPulse(Pulse):
    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var=3)

    def pulse(self, time, paras):
        """
        Generate a pulse with sinusoidal shape.
        """
        return paras[0] * jnp.cos(paras[1] * time + paras[2])


class FourierPulse(Pulse):
    def __init__(self, n_sup=1, n_var=4):
        if n_var % 2 != 0 or n_var < 4:
            raise ValueError("n_var must be an even number >= 4")
        super().__init__(n_sup, n_var)
        self.n = jnp.arange(1, self.n_par // 2)

    def pulse(self, time, paras):
        """
        Compute the Fourier series for a given set of parameters at point t.

        A0 +
        A1 * cos(2 * pi / period * 1 * t) + B1 * sin(2 * pi / period * 1 * t) +
        A2 * cos(2 * pi / period * 2 * t) + B2 * sin(2 * pi / period * 2 * t) +
        ...
        """
        pulse = paras[1] + jnp.sum(
            paras[2 * self.n] * jnp.cos(2 * jnp.pi / paras[0] * self.n * time)
            + paras[2 * self.n + 1] * jnp.sin(2 * jnp.pi / paras[0] * self.n * time)
        )
        return pulse


class FluxPulse(Pulse):
    # implementation of pulse from GOAT paper
    # NOTICE: there might be errors in the paper
    #         the amplitudes seem to be off

    def __init__(self, times, n_sup, n_var):
        super().__init__(times, n_sup=1, n_var=18)
        self.T = times[-1]  # pulse duration

    def L(self, x, a, b, a2, b2):
        i = (b2 - a2) / (b - a)
        ii = x - (b + a) / 2
        iii = (b2 + a2) / 2
        return i * ii + iii

    def C(self, x, a, b):
        b_p_a = (b + a) / 2
        b_m_a = (b - a) / 2
        return b_m_a * jnp.sin((x - b_p_a) / b_m_a) + b_p_a

    def rescale(self, paras):
        p = paras.reshape(6, 3)
        L, b = jnp.zeros((6, 3)), jnp.zeros((6, 3))
        # linearly rescacle parameters
        L = L.at[:, 0].set(self.L(p[:, 0], -1, 1, -0.3, 0.3))
        L = L.at[:, 1].set(self.L(p[:, 1], -1, 1, -2.0944e9, 2.0944e9))
        L = L.at[:, 2].set(self.L(p[:, 2], -1, 1, -3141.59, 3141.59))
        # bound parameters
        b = b.at[:, 0].set(self.C(L[:, 0], -0.3, 0.3))
        b = b.at[:, 1].set(self.C(L[:, 1], -2.0944e9, 2.0944e9))
        b = b.at[:, 2].set(L[:, 2])  # no bound
        return b.reshape(18)

    def S_down_bar(self, x, g0):
        return 1 / (1 + jnp.exp((4 * g0) * x))

    def S_up_bar(self, x, g0):
        return 1 - self.S_down_bar(x, g0)

    def S_up(self, a, b, x):
        b_p_a = (b + a) / 2
        b_m_a = (b - a) / 2
        S = self.S_up_bar((x - b_p_a) / b_m_a, 1 / 2)
        return (2 * S - 1) * b_m_a + b_p_a

    def A(self, f, tau, a, b, g, Delta):
        b_m_a = (b - a) / 2
        i = self.S_up_bar(tau - Delta, g)
        ii = self.S_down_bar(tau - (1 - Delta), g)
        iii = self.S_up(-b_m_a, b_m_a, f)
        return i * ii * iii

    def d(self, paras, t):
        p = paras.reshape(6, 3)
        d = p[:, 0] * jnp.sin(p[:, 1] * t + p[:, 2])
        return d.sum()

    def delta(self, paras, t):
        d = self.d(paras, t)
        return self.A(d, t / self.T, -0.3, 0.3, 40, 0.075)

    def G(self, f, t, Theta, omega):
        return Theta + jnp.cos(omega * t) * f

    def Phi(self, paras, t):
        delta = self.delta(paras, t)
        return self.G(delta, t, -0.108, 5.34448e9)

    def W(self, Phi, t, omega_0):
        return omega_0 * jnp.sqrt(jnp.abs(jnp.cos(jnp.pi * Phi)))

    def omega(self, paras, t):
        Phi = self.Phi(paras, t)
        return self.W(Phi, t, 4.67783e10)

    def omega_direct(self, paras, t):
        f = self.delta(paras, t)  # = delta
        Theta = -0.108
        omega = 5.34448e9
        Phi = Theta + jnp.cos(omega * t) * f  # = G
        omega_0 = 4.67783e10
        W = omega_0 * jnp.sqrt(jnp.abs(jnp.cos(jnp.pi * Phi)))
        return W

    def pulse(self, time, paras):
        """
        Generate a pulse with polynomial shape.
        """
        # rescaled_paras = self.rescale(paras)
        pulse = self.omega_direct(paras, time)
        return pulse
