"""Control implementation based on state-space methods."""
# ABCD are tiny; ~4x4, no GPU here
import numpy as np


class StateSpaceFilter:
    """A state-space representation of a discrete time filter / controller."""

    def __init__(self, A, B, C, D, x=None):
        """Create a new State-Space Filter.

        Parameters
        ----------
        A : numpy.ndarray
            shape (n,n) square array; state transition matrix
        B : numpy.ndarray
            shape (n,) column vector, "input matrix"
            note (n,) duck-types column or row vector in numpy
        C : numpy.ndarray
            shape (n,) row vector, "input matrix"
        D : float
            feedthrough constant
        x : numpy.ndarray
            initial state of the filter

        """
        if x is None:
            x = np.zeros_like(B)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x = x

    def update(self, y):
        """Iterate the filter one step in time.

        Parameters
        ----------
        y : float
            input variable, "measurand"

        Returns
        -------
        u : float
            control variable

        """
        xhat = self.A @ self.x + self.B * y
        u = np.dot(self.C, self.x) + self.D * y
        self.x = xhat
        return u

    def reset(self):
        """Reset the filter's state to 0."""
        self.x[:] = 0
        return


class NoOpFilter:
    """No-op used to place (something) in public examples in lieu of a real state space filter."""

    def __init__(self):
        return

    def update(self, y):
        return 0

    def reset(self):
        return
