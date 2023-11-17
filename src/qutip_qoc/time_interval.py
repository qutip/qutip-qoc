import numpy as np


class TimeInterval:
    """
    Class for storing a time interval and deriving its attributes.

    Attributes
    ----------
    tslots : array_like, optional
        List of time slots at which the control pulse is evaluated.
        The last element of tslots is the total evolution time.
        Can be unevenly spaced.

    evo_time : float, optional
        Total evolution time.
        If given together with n_tslots, tslots is derived from evo_time
        and assumed to be evenly spaced.

    n_tslots : int, optional
        Number of time slots. Length of tslots is n_tslots.

    tdiffs : array_like, optional
        List of time intervals between time slots.
        Can be unevenly spaced.
        Length of tdiffs is n_tslots - 1.
        Sum over all elements of tdiffs is evo_time.
    """

    def __init__(self, tslots=None, evo_time=None, n_tslots=None, tdiffs=None):
        self._tslots = tslots
        self._evo_time = evo_time
        self._n_tslots = n_tslots
        self._tdiffs = tdiffs

    @property
    def tslots(self):
        if self._tslots is None:
            n_tslots = self.n_tslots
            if self._evo_time:  # derive from evo_time
                self._tslots = np.linspace(0., self._evo_time, n_tslots)
            elif self._tdiffs:  # derive from tdiffs
                self._tslots = [sum(self._tdiffs[:i])
                                for i in range(n_tslots - 1)]
        return self._tslots

    @property
    def tdiffs(self):
        if self._tdiffs is None:
            tslots = self.tslots
            self._tdiffs = np.diff(tslots)
        return self._tdiffs

    @property
    def evo_time(self):
        if self._evo_time is None:
            tslots = self.tslots
            self._evo_time = tslots[-1]
        return self._evo_time

    @property
    def n_tslots(self):
        if self._n_tslots is None:
            if self._tslots:
                self._n_tslots = len(self._tslots)
            elif self._tdiffs:
                self._n_tslots = len(self._tdiffs) - 1
            else:
                raise ValueError(
                    "Either tslots, tdiffs, or evo_time + n_tslots must be specified."
                )
        return self._n_tslots
