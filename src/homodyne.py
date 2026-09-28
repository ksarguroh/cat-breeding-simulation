import numpy as np


def momentum_fock_wavefunctions(p_grid, N):
    """
    Calculate the momentum-space wavefunctions <p|n>
    for n = 0, ..., N-1.
    """

    p_grid = np.asarray(p_grid, dtype=float)

    phi = np.zeros(
        (len(p_grid), N),
        dtype=complex
    )

    # Fock state |0>
    phi[:, 0] = (
        np.pi ** (-0.25)
        * np.exp(-p_grid**2 / 2)
    )

    # Fock state |1>
    if N > 1:
        phi[:, 1] = (
            -1j
            * np.sqrt(2)
            * p_grid
            * phi[:, 0]
        )

    # Higher Fock states
    for n in range(1, N - 1):
        phi[:, n + 1] = (
            -1j
            * np.sqrt(2 / (n + 1))
            * p_grid
            * phi[:, n]
            + np.sqrt(n / (n + 1))
            * phi[:, n - 1]
        )

    return phi

def conditional_states(psi_out, p_grid):
    """
    Calculate the unnormalised conditional state of mode B
    for every momentum value in p_grid.

    Parameters
    ----------
    psi_out : qutip.Qobj
        Two-mode output state after the beamsplitter.
    p_grid : array-like
        Momentum measurement grid.

    Returns
    -------
    conditional : ndarray
        Shape (len(p_grid), N).

        conditional[i, :] contains the Fock-basis coefficients
        of the unnormalised conditional state of mode B
        corresponding to p_grid[i].
    """

    if not psi_out.isket:
        raise ValueError("psi_out must be a ket.")

    N_A = psi_out.dims[0][0]
    N_B = psi_out.dims[0][1]

    if N_A != N_B:
        raise ValueError(
            "This implementation assumes equal Fock-space dimensions."
        )

    # Two-mode Fock-basis coefficient matrix
    C = psi_out.full().reshape(N_A, N_B)

    # Momentum-space Fock wavefunctions
    phi = momentum_fock_wavefunctions(
        p_grid,
        N_A
    )

    # Project mode A onto <p|
    conditional = phi @ C

    return conditional

def homodyne_pdf(psi_out, p_grid):
    """
    Calculate the momentum homodyne probability density.

    Parameters
    ----------
    psi_out : qutip.Qobj
        Two-mode output ket.
    p_grid : array-like
        Momentum measurement grid.

    Returns
    -------
    pdf : ndarray
        Normalised homodyne probability density.
    conditional : ndarray
        Unnormalised conditional states of mode B.
    """

    conditional = conditional_states(
        psi_out,
        p_grid
    )

    pdf = np.sum(
        np.abs(conditional)**2,
        axis=1
    )

    # Normalise the numerical PDF
    integral = np.trapezoid(
        pdf,
        p_grid
    )

    if integral <= 0:
        raise ValueError(
            "Homodyne probability density has a non-positive integral."
        )

    pdf /= integral

    return pdf, conditional