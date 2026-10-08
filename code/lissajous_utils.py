import numpy as np
from scipy.integrate import quad


def lissajous_arc_length(A: float, B: float, a: float, b: float) -> float:
    """Computes the arc length of the Lissajous curve:

    x = A * sin(a * t)
    y = B * sin(b * t)
    for t in [0, 2 * pi].

    Parameters
    ----------
    A : float
        Amplitude along x-axis.
    B : float
        Amplitude along y-axis.
    a : float
        Frequency constant for x (a >= 1).
    b : float
        Frequency constant for y (b >= 1, |a - b| = 1).

    Returns
    -------
    float
        The total arc length of the curve over t in [0, 2*pi].
    """

    def integrand(t):
        dxdt = A * a * np.cos(a * t)
        dydt = B * b * np.cos(b * t)
        return np.sqrt(dxdt**2 + dydt**2)

    length, abserr = quad(integrand, 0, 2 * np.pi)
    return length


# Example Usage:
# if __name__ == "__main__":
#     A_val, B_val = 2.0, 3.0
#     a_val, b_val = 3.0, 2.0  # |a - b| = 1
# 
#     length = lissajous_arc_length(A_val, B_val, a_val, b_val)
#     print(f"Arc Length: {length:.6f}")