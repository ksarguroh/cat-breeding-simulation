import numpy as np
import qutip as qt


def apply_beamsplitter(psi, N, theta=np.pi / 4):
    a = qt.destroy(N)

    a1 = qt.tensor(a, qt.qeye(N))
    a2 = qt.tensor(qt.qeye(N), a)

    generator = a1.dag() * a2 - a1 * a2.dag()
    U = (theta * generator).expm()

    return U * psi