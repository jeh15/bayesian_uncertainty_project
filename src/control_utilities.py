import numpy as np
import numpy.typing as npt
import scipy


def LQR(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    Q: npt.ArrayLike,
    R: npt.ArrayLike,
    x: npt.ArrayLike,
) -> np.ndarray:
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    F = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    u = -F @ x
    return u
