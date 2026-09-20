import qutip as qt


def make_cat(N, alpha, parity=1):
    ket_plus = qt.coherent(N, alpha)
    ket_minus = qt.coherent(N, -alpha)

    return (ket_plus + parity * ket_minus).unit()


def make_squeezed_cat(N, alpha, r, parity=1):
    cat = make_cat(N, alpha, parity)
    S = qt.squeeze(N, r)

    return S * cat