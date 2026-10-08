import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from typing import Tuple, Union


class LissajousConstantSpeed:
    """
    Reparameterizes a Lissajous curve by arc length so that the object moves
    along the curve at a constant linear speed.

    Parametric equations:
        x(tau) = A * sin(a * tau)
        y(tau) = B * sin(b * tau)
    where tau in [0, 2*pi].
    """

    def __init__(
        self,
        A: float,
        B: float,
        a: float,
        b: float,
        table_resolution: int = 2000
    ):
        """
        Initializes the Lissajous constant speed solver.

        Parameters
        ----------
        A : float
            Amplitude along the x-axis.
        B : float
            Amplitude along the y-axis.
        a : float
            Frequency multiplier for x (a >= 1).
        b : float
            Frequency multiplier for y (b >= 1, |a - b| = 1).
        table_resolution : int, default=2000
            Number of points used to construct the monotonic spline lookup table.
        """
        if a < 1 or b < 1:
            raise ValueError("Frequencies 'a' and 'b' must be >= 1.")
        if abs(a - b) != 1:
            raise ValueError("|a - b| must equal 1.")

        self.A = float(A)
        self.B = float(B)
        self.a = float(a)
        self.b = float(b)
        self.table_resolution = table_resolution

        self._build_arc_length_table()

    def speed(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates the instantaneous linear speed ||dpos/dtau||."""
        dxdtau = self.A * self.a * np.cos(self.a * tau)
        dydtau = self.B * self.b * np.cos(self.b * tau)
        return np.sqrt(dxdtau**2 + dydtau**2)

    def _build_arc_length_table(self):
        """
        Computes the total arc length and creates a high-precision inverse mapping
        from arc-length distance s to the original phase parameter tau.
        """
        # Calculate total arc length L over [0, 2*pi] with high precision
        self.total_length, _ = quad(self.speed, 0.0, 2.0 * np.pi, limit=200)

        # Dense sample grid for tau
        tau_samples = np.linspace(0.0, 2.0 * np.pi, self.table_resolution)
        speed_samples = self.speed(tau_samples)

        # Cumulative numerical integration for arc length s(tau)
        s_samples = cumulative_trapezoid(speed_samples, tau_samples, initial=0.0)

        # Force exact boundary matching to minimize end-point drift
        s_samples[-1] = self.total_length

        # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves monotonicity
        # Inverse mapping: s (distance) -> tau (original parameter)
        self._s_to_tau_spline = PchipInterpolator(s_samples, tau_samples)

    def get_tau_from_distance(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Maps arc length distance s in [0, total_length] to original phase tau in [0, 2*pi].
        Supports scalar and vectorized inputs. Wraps periodically for s outside bounds.
        """
        s_mod = np.mod(s, self.total_length)
        return self._s_to_tau_spline(s_mod)

    def position_at_distance(self, s: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Returns (x, y) coordinates for a distance s along the curve.

        Equispaced inputs s_i (s[i+1] - s[i] = ds) produce equispaced arc lengths
        and uniform step sizes ||pos[i+1] - pos[i]|| ≈ ds.
        """
        tau = self.get_tau_from_distance(s)
        x = self.A * np.sin(self.a * tau)
        y = self.B * np.sin(self.b * tau)
        return x, y

    def position_at_time(self, t: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Returns (x, y) coordinates for t in [0, 2*pi], where t is normalized to total curve length.
        Equispaced inputs t_i result in constant linear speed along the curve.
        """
        # Convert t in [0, 2*pi] to distance s in [0, total_length]
        s = (np.asarray(t) / (2.0 * np.pi)) * self.total_length
        return self.position_at_distance(s)


def lissajous_constant_speed_point(
    t: float,
    A: float,
    B: float,
    a: float,
    b: float
) -> Tuple[float, float]:
    """
    Returns the (x, y) position on the Lissajous curve for a parameter t in [0, 2*pi]
    such that linear speed is constant (i.e. equispaced t yields equispaced positions).

    Parameters
    ----------
    t : float
        Normalized parameter in [0, 2*pi].
    A, B : float
        Curve amplitudes along x and y axes.
    a, b : float
        Frequencies (a, b >= 1 and |a - b| = 1).

    Returns
    -------
    (x, y) : Tuple[float, float]
        2D spatial position on the curve.
    """
    curve = LissajousConstantSpeed(A=A, B=B, a=a, b=b)
    return curve.position_at_time(t)


# if __name__ == "__main__":
#     # Define parameters
#     A, B = 2.0, 3.0
#     a, b = 3.0, 2.0  # |a - b| = 1
#     N_points = 60# 

#     print("--- Lissajous Constant Linear Speed Reparameterization ---")
#     curve = LissajousConstantSpeed(A=A, B=B, a=a, b=b)
#     print(f"Total curve length L = {curve.total_length:.6f}")# 

#     # Equispaced parameter values t in [0, 2*pi]
#     t_equispaced = np.linspace(0, 2 * np.pi, N_points, endpoint=False)# 

#     # 1. Standard (variable speed) parameterization
#     x_std = A * np.sin(a * t_equispaced)
#     y_std = B * np.sin(b * t_equispaced)
#     step_lengths_std = np.sqrt(np.diff(x_std, append=x_std[0])**2 + np.diff(y_std, append=y_std[0])**2)# 

#     # 2. Constant speed reparameterization
#     x_cs, y_cs = curve.position_at_time(t_equispaced)
#     step_lengths_cs = np.sqrt(np.diff(x_cs, append=x_cs[0])**2 + np.diff(y_cs, append=y_cs[0])**2)# 

#     print("\n--- Step Size Statistics (Consecutive Point Distances) ---")
#     print(f"Standard Parameterization  : Mean = {step_lengths_std.mean():.4f}, Std Dev = {step_lengths_std.std():.4f}")
#     print(f"Constant Speed Reparameter : Mean = {step_lengths_cs.mean():.4f}, Std Dev = {step_lengths_cs.std():.4f}")# 

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))# 

#     # High-res background curve
#     t_dense = np.linspace(0, 2 * np.pi, 1000)
#     x_dense = A * np.sin(a * t_dense)
#     y_dense = B * np.sin(b * t_dense)# 

#     # Plot 1: Standard parameterization
#     axes[0].plot(x_dense, y_dense, 'k--', alpha=0.3, label='Curve')
#     axes[0].scatter(x_std, y_std, c='red', s=30, zorder=3, label='Sampled Points')
#     axes[0].set_title(f"Standard Parameterization\n(Std Dev of step size = {step_lengths_std.std():.4f})")
#     axes[0].set_xlabel("x")
#     axes[0].set_ylabel("y")
#     axes[0].grid(True, linestyle=':', alpha=0.6)
#     axes[0].axis('equal')
#     axes[0].legend()# 

#     # Plot 2: Constant speed parameterization
#     axes[1].plot(x_dense, y_dense, 'k--', alpha=0.3, label='Curve')
#     axes[1].scatter(x_cs, y_cs, c='blue', s=30, zorder=3, label='Constant Speed Points')
#     axes[1].set_title(f"Constant Speed Reparameterization\n(Std Dev of step size = {step_lengths_cs.std():.4f})")
#     axes[1].set_xlabel("x")
#     axes[1].set_ylabel("y")
#     axes[1].grid(True, linestyle=':', alpha=0.6)
#     axes[1].axis('equal')
#     axes[1].legend()# 

#     plt.tight_layout()
#     plt.show()