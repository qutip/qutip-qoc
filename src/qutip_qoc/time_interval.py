import numpy as np


class TimeInterval:
    """
    """

    def __init__(self, tlist=None, evo_time=None, num_tslots=None,
                 tau=None, bounds=None, guess=None):
        self._tlist = tlist
        self._tau = tau
        self._evo_time = evo_time
        self._num_tslots = num_tslots

    @property
    def tlist(self):
        if self._tlist is None:
            n_tslots = self.num_tslots
            if self._evo_time:  # derive from evo_time
                self._tlist = np.linspace(0., self._evo_time, n_tslots)
            elif self._tau:  # derive from tau
                self._tlist = [sum(self._tau[:i]) for i in range(n_tslots - 1)]
        return self._tlist

    @property
    def tau(self):
        if self._tau is None:
            tlist = self.tlist
            self._tau = np.diff(tlist)
        return self._tau

    @property
    def evo_time(self):
        if self._evo_time is None:
            tlist = self.tlist
            self._evo_time = tlist[-1]
        return self._evo_time

    @property
    def num_tslots(self):
        if self._num_tslots is None:
            if self._tlist:
                self._num_tslots = len(self._tlist)
            elif self._tau:
                self._num_tslots = len(self._tau) - 1
            else:
                raise ValueError(
                    "Either tlist, tau, or evo_time + num_tslots must be specified."
                )
        return self._num_tslots
