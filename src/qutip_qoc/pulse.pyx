"""
This module contains classes for analytically defined pulse shapes for use with GOAT.
It is intended to provide common pulse functions implemented in Cython.
"""

from libc.math cimport sin, cos, exp, pi
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Pulse:
    """
    Base class for analytically defined pulse shapes for use with GOAT and JOPT.

    Attributes:
        n_sup (int):
            number of superpositions i.e. summands

        n_var (int):
            number of parameters for each summand

        n_par (int):
            total number of parameters
    """
    cdef public int n_sup
    cdef public int n_var
    cdef public int n_par

    def __init__(self, n_sup, n_var):
        self.n_sup = n_sup
        self.n_var = n_var
        self.n_par = n_sup * n_var

    def __call__(self, time, paras):
        return self.gen_pulse(time, paras)

    cpdef double gen_pulse(self, double time, double[:] paras):
        """
        Generate a supposition of pulses.
        A set of self.n_var parameters defines a single pulse.
        Summing over self.n_sup pulses gives the total pulse.
        """
        cdef double pulse_val = 0.0
        cdef int i

        for i in range(0, self.n_par, self.n_var):
            pulse_val += self.pulse(time, paras[i: i + self.n_var])
        return pulse_val

    cpdef double gen_grad(self, double time, double[:] paras, int idx):
        """
        Generate the gradient of a superposition of pulses.
        Index runs over all parameters 0, 1, ..., (self.n_par-1).
        If time is also a parameter, then idx = self.n_par refers to
        the derivative with respect to time.
        """
        cdef int i
        cdef double grad = 0.0
        cdef double[:] par

        if idx == self.n_par:
            for i in range(0, self.n_par, self.n_var):
                grad += self.pulse_grad(time, paras[i: i + self.n_var], idx)
        else:
            i = idx % self.n_var # single summand parameters
            par = paras[(idx - i): (idx - i) + self.n_var]
            grad = self.pulse_grad(time, par, i)

        return grad

    cdef double pulse(self, double time, double[:] paras):
        # to be implemented by subclass
        return 0.

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        # to be implemented by subclass
        return 0.



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GaussianPulse(Pulse):
    """
    Gaussian pulse with summands of shape:
    s * exp(-1 * ((time - m) ** 2) / (v ** 2))
    Number of parameters per summand is fixed: 3
    """
    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var=3)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with Gaussian shape. The peak is centred around the
        mean and the variance determines the breadth.
        """
        cdef double s, m, v, pulse_val
        s, m, v = paras[0], paras[1], paras[2]

        pulse_val = s * exp(-1 * ((time - m) ** 2) / (v ** 2))
        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the gradient of a pulse with Gaussian shape
        for the idx-th parameter.
        """
        cdef double s, m, v, grad
        s, m, v = paras[0], paras[1], paras[2]

        grad = exp(-1 * ((time - m)**2) / (v**2))
        if idx == 0:
            return grad
        elif idx == 1:
            grad = s * grad * 2 * (time - m) / (v**2)
            return grad
        elif idx == 2:
            grad = s * grad * 2 * (time - m)**2 / (v**3)
            return grad
        elif idx == 3:
            grad = s * grad * (-2) * (time - m) / (v**2)
            return grad



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SinPulse(Pulse):
    """
    Sine pulse with summands of shape:
    p0 * sin(p1 * time + p2)
    Number of parameters per summand is fixed: 3
    """

    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var=3)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with sinusoidal shape.
        """
        cdef double p0, p1, p2, pulse_val
        p0, p1, p2 = paras[0], paras[1], paras[2]

        pulse_val = p0 * sin(p1 * time + p2)

        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of pulse with sinusoidal shape
        for idx-th parameter.
        """
        cdef double p0, p1, p2, grad
        p0, p1, p2 = paras[0], paras[1], paras[2]

        if idx == 0:
            grad = sin(p1 * time + p2)
            return grad
        elif idx == 1:
            grad = p0 * cos(p1 * time + p2) * time
            return grad
        elif idx == 2:
            grad = p0 * cos(p1 * time + p2)
            return grad
        elif idx == 3:
            grad = p0 * cos(p1 * time + p2) * p1
            return grad



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CosPulse(Pulse):
    """
    Cosine pulse with summands of shape:
    p0 * cos(p1 * time + p2)
    Number of parameters per summand is fixed: 3
    """

    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var=3)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with sinusoidal shape.
        """
        cdef double p0, p1, p2, pulse_val
        p0, p1, p2 = paras[0], paras[1], paras[2]

        pulse_val = p0 * cos(p1 * time + p2)

        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of pulse with sinusoidal shape
        for idx-th parameter.
        """
        cdef double p0, p1, p2, grad
        p0, p1, p2 = paras[0], paras[1], paras[2]

        if idx == 0:
            grad = cos(p1 * time + p2)
            return grad
        elif idx == 1:
            grad = p0 * (-1) * sin(p1 * time + p2) * time
            return grad
        elif idx == 2:
            grad = p0 * (-1) * sin(p1 * time + p2)
            return grad
        elif idx == 3:
            grad = p0 * (-1) * sin(p1 * time + p2) * p1
            return grad



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CosCosPulse(Pulse):
    """
    Modulated cosine pulse with summands of shape:
    p0 * cos(p1 * time - p2) * cos(p3 * time - p4)
    Number of parameters per summand is fixed: 5
    """

    def __init__(self, n_sup=1, n_var=5):
        super().__init__(n_sup, n_var=5)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with modulated sinusoidal shape.
        Cosine envelope for the amplitude of the other cosine.
        """
        cdef double p0, p1, p2, p3, p4, pulse_val
        p0, p1, p2, p3, p4 = paras[0], paras[1], paras[2], paras[3], paras[4]
        cdef double t = time

        pulse_val = p0 * cos(p1 * t - p2) * cos(p3 * t - p4)
        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of pulse with modulated sinusoidal shape
        for the idx-th parameter.
        """
        cdef double p0, p1, p2, p3, p4, grad
        p0, p1, p2, p3, p4 = paras[0], paras[1], paras[2], paras[3], paras[4]
        cdef double t = time

        if idx == 0:
            grad = cos(p1 * t - p2) * cos(p3 * t - p4)
            return grad
        elif idx == 1:
            grad = -p0 * t * cos(p3 * t - p4) * sin(p1 * t - p2)
            return grad
        elif idx == 2:
            grad = -p0 * cos(p3 * t - p4) * sin(p2 - p1 * t)
            return grad
        elif idx == 3:
            grad = -p0 * t * cos(p1 * t - p2) * sin(p3 * t - p4)
            return grad
        elif idx == 4:
            grad = -p0 * cos(p1 * t - p2) * sin(p4 - p3 * t)
            return grad
        elif idx == 5:
            grad = -p0 * (p1 * sin(p1 * t - p2) * cos(p3 * t - p4) +\
                         p3 * cos(p1 * t - p2) * sin(p3 * t - p4))
            return grad


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class PolynomialPulse(Pulse):
    """
    Polynomial pulse with summands of shape:
    p0 + p1 * time + p2 * time**2 + p3 * time**3 + ...
    """

    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with polynomial shape.
        """
        cdef double pulse_val = 0.0
        cdef int i

        for i in range(self.n_par):
            pulse_val += paras[i] * time**i
        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of the polynomial pulse.
        """
        cdef double grad = 0.
        cdef int i
        if idx == self.n_par:
            for i in range(1, self.n_par):
                grad += paras[i] * i * time**(i-1)
        else:
            grad = time**idx
        return grad


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FourierPulse(Pulse):
    """
    Fourier pulse with summands of shape:
    A0 + A1 * cos(2 * pi / period * 1 * t) + B1 * sin(2 * pi / period * 1 * t) + A2 * cos(2 * pi / period * 2 * t) + B2 * sin(2 * pi / period * 2 * t) + ...
    where the period is the first parameter.
    """

    def __init__(self, n_sup=1, n_var=4):
        if n_var % 2 != 0 or n_var < 4:
            raise ValueError("n_var must be an even number >= 4")
        super().__init__(n_sup, n_var)

    cdef double pulse(self, double t, double[:] paras):
        """
        Compute the Fourier series for a given set of parameters at point t.
        """
        cdef double period = paras[0]
        cdef double result = paras[1] # A0
        cdef int n

        for n in range(1, self.n_par // 2):
            result += paras[2*n] * cos(2 * (pi / period) * n * t) + \
                  paras[2*n + 1] * sin(2 * (pi / period) * n * t)

        return result

    cdef double pulse_grad(self, double t, double[:] paras,  int idx):
        """
        Compute the gradient of the Fourier series with respect to a specific parameter at a given index.
        """
        cdef double period = paras[0]
        cdef double result = 0.0
        cdef int n

        if idx == self.n_par: # wrt time
            for n in range(1, self.n_par // 2):
                result += (2 * (pi / period) * n ) *\
                    (paras[2*n] * (-1) * sin(2 * (pi / period) * n * t) +\
                     paras[2*n + 1] * cos(2 * (pi / period) * n * t))

        elif idx == 0: # wrt period
            for n in range(1, self.n_par // 2):
                result += (-2 * pi / (period**2) * n * t) *\
                    (paras[2*n] * (-1) * sin(2 * (pi / period) * n * t) +\
                     paras[2*n + 1] * cos(2 * (pi / period) * n * t))

        elif idx==1: # wrt A0
            result = 1.

        elif idx % 2 == 0: # wrt A
            n = idx // 2
            result = cos(2 * (pi / period) * n * t)

        elif idx % 2 == 1: # wrt B
            n = idx // 2
            result = sin(2 * (pi / period) * n * t)

        return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class PWCPulse(Pulse):
    """
    Piecewise constant pulse with parameters specifying the value of the pulse
    during equidistant segments of lenth (interval[-1] - interval[0]) / n_var.
    n_var is the number of segments (parameters).
    """

    cdef double[:] interval
    cdef double step
    cdef int max_idx

    def __init__(self, interval, n_var=3):
        super().__init__(n_var, n_sup=1)
        self.interval = interval
        self.step = (interval[-1] - interval[0]) / n_var
        self.max_idx = n_var - 1

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with piecewise constant shape.
        Assumes that interval is equidistantly spaced.
        """
        cdef int i = int((time - self.interval[0]) / self.step)
        cdef int idx = min(i, self.max_idx)
        return paras[idx]

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of pulse with piecewise constant shape
        for the idx-th parameter.
        """
        cdef int i = int((time - self.interval[0]) / self.step)
        cdef int index = min(i, self.max_idx)
        if idx == self.n_par: # wrt time
            return 0.
        return 0. if index != idx else 1.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class PWLPulse(Pulse):
    """
    Piecewise linear pulse with parameters specifying the value of the pulse
    during equidistant segments of lenth (interval[-1] - interval[0]) / n_var.
    Parameters are the slope and intercept of each segment:
    slope_1, inter_1, slope_2, inter_2, ...
    There are 2x(n_segments) parameters.
    """

    cdef double[:] interval
    cdef double step
    cdef int max_idx

    def __init__(self, interval, n_var=3):
        super().__init__(n_var=n_var, n_sup=1)
        self.interval = interval
        self.step = (interval[-1] - interval[0]) / (n_var / 2)
        self.max_idx = (n_var / 2) - 1

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with piecewise linear shape.
        Assumes that interval is equidistantly spaced in segments.
        Parameters are the slope and intercept of each segment:
        slope_1, inter_1, slope_2, inter_2, ...
        There are 2x(n_segments) parameters.
        """
        # calculate segment index on interval
        cdef int i = int((time - self.interval[0]) / self.step)
        cdef int idx = min(i, self.max_idx)

        # get slope and intercept
        cdef double slope = paras[idx * 2]
        cdef double inter = paras[idx * 2 + 1]

        return slope * (time - (idx * self.step)) + inter

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of pulse with piecewise linear shape
        for the idx-th parameter.
        """
        cdef int i = int((time - self.interval[0]) / self.step)
        cdef int index = min(i, self.max_idx)
        if idx == self.n_par: # wrt time
             return paras[index * 2]
        elif idx == index * 2: # wrt slope
            return time - (index * self.step)
        elif idx == index * 2 + 1: # wrt intercept
            return 1.
        else: # wrt parameters on different segment
            return 0.
