from libc.math cimport sin, cos, exp, pi
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Pulse:
    cdef public int n_sup
    cdef public int n_var
    cdef public int n_par

    def __init__(self, n_sup, n_var):
        self.n_sup = n_sup  # num of superpositions, 'm' in paper
        self.n_var = n_var  # num of paras for each control function
        self.n_par = n_sup * n_var  # total num of paras

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
        return 0.

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        return 0.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GaussianPulse(Pulse):
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
    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var=3)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with sinusoidal shape.
        """
        cdef double s, m, v, pulse_val
        s, m, v = paras[0], paras[1], paras[2]

        pulse_val = s * sin(m * time + v)

        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of pulse with sinusoidal shape
        for idx-th parameter.
        """
        cdef double s, m, v, grad
        s, m, v = paras[0], paras[1], paras[2]

        if idx == 0:
            grad = sin(m * time + v)
            return grad
        elif idx == 1:
            grad = s * cos(m * time + v) * time
            return grad
        elif idx == 2:
            grad = s * cos(m * time + v)
            return grad
        elif idx == 3:
            grad = s * cos(m * time + v) * m
            return grad


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CosPulse(Pulse):
    def __init__(self, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var=3)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with sinusoidal shape.
        """
        cdef double s, m, v, pulse_val
        s, m, v = paras[0], paras[1], paras[2]

        pulse_val = s * cos(m * time + v)

        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of a pulse with sinusoidal shape
        for the idx-th parameter.
        """
        cdef double s, m, v, grad
        s, m, v = paras[0], paras[1], paras[2]

        if idx == 0:
            grad = cos(m * time - v)
            return grad
        elif idx == 1:
            grad = s * (-1) * sin(m * time + v) * time
            return grad
        elif idx == 2:
            grad = s * (-1) * sin(m * time + v)
            return grad
        elif idx == 3:
            grad = s * (-1) * sin(m * time + v) * m
            return grad


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CosCosPulse(Pulse):
    def __init__(self, n_sup=1, n_var=5):
        super().__init__(n_sup, n_var=5)

    cdef double pulse(self, double time, double[:] paras):
        """
        Generate a pulse with modulated sinusoidal shape.
        Cosine envelope for the amplitude of the other cosine.
        """
        cdef double s, m1, v1, m2, v2, pulse_val
        s, m1, v1, m2, v2 = paras[0], paras[1], paras[2], paras[3], paras[4]
        cdef double t = time

        pulse_val = s * cos(m1 * t - v1) * cos(m2 * t - v2)
        return pulse_val

    cdef double pulse_grad(self, double time, double[:] paras, int idx):
        """
        Generate the derivative of pulse with modulated sinusoidal shape
        for the idx-th parameter.
        """
        cdef double s, m1, v1, m2, v2, grad
        s, m1, v1, m2, v2 = paras[0], paras[1], paras[2], paras[3], paras[4]
        cdef double t = time

        if idx == 0:
            grad = cos(m1 * t - v1) * cos(m2 * t - v2)
            return grad
        elif idx == 1:
            grad = -s * t * cos(m2 * t - v2) * sin(m1 * t - v1)
            return grad
        elif idx == 2:
            grad = -s * cos(m2 * t - v2) * sin(v1 - m1 * t)
            return grad
        elif idx == 3:
            grad = -s * t * cos(m1 * t - v1) * sin(m2 * t - v2)
            return grad
        elif idx == 4:
            grad = -s * cos(m1 * t - v1) * sin(v2 - m2 * t)
            return grad
        elif idx == 5:
            grad = -s * (m1 * sin(m1 * t - v1) * cos(m2 * t - v2) +\
                         m2 * cos(m1 * t - v1) * sin(m2 * t - v2))
            return grad


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class PolynomialPulse(Pulse):
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
    def __init__(self, n_sup=1, n_var=4):
        if n_var % 2 != 0 or n_var < 4:
            raise ValueError("n_var must be an even number >= 4")
        super().__init__(n_sup, n_var)

    cdef double pulse(self, double t, double[:] paras):
        """
        Compute the Fourier series for a given set of parameters at point t.

        A0 + 
        A1 * cos(2 * pi / period * 1 * t) + B1 * sin(2 * pi / period * 1 * t) +
        A2 * cos(2 * pi / period * 2 * t) + B2 * sin(2 * pi / period * 2 * t) +
        ...
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
    cdef double[:] interval
    cdef double step
    cdef int max_idx

    def __init__(self, interval, n_sup=1, n_var=3):
        super().__init__(n_sup, n_var)
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
    cdef double[:] interval
    cdef double step
    cdef int max_idx

    def __init__(self, interval, n_sup=1, n_var=3):
        super().__init__(n_sup=1, n_var=n_var)
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
