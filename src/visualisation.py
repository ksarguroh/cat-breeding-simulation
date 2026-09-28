import matplotlib.pyplot as plt
import qutip as qt


def plot_wigner(state, x_grid, p_grid, title=None):
    """
    Calculate and plot the Wigner function of a single-mode state.

    Parameters
    ----------
    state : qutip.Qobj
        Single-mode quantum state.
    x_grid : array-like
        Position-axis grid.
    p_grid : array-like
        Momentum-axis grid.
    title : str, optional
        Plot title.
    """

    W = qt.wigner(
        state,
        x_grid,
        p_grid
    )

    plt.figure(figsize=(7, 6))

    plt.contourf(
        x_grid,
        p_grid,
        W,
        levels=100
    )

    plt.xlabel("x")
    plt.ylabel("p")
    plt.colorbar(label="W(x,p)")

    if title is not None:
        plt.title(title)

    plt.show()