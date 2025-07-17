import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from qutip_qoc.q2.grape import grape_unitary
from qutip import Qobj, sigmaz, sigmax
import numpy as np

def test_grape_unitary():
    U_target = Qobj([[1, 1], [1, -1]]) / np.sqrt(2)
    H0 = sigmaz()
    H_ops = [sigmax()]
    times = np.linspace(0, 10, 100)
    
    result = grape_unitary(
        U=U_target,
        H0=H0,
        H_ops=H_ops,
        R=5,  # Reduced iterations for testing
        times=times
    )
    assert isinstance(result.U_f, Qobj)