
from scipy.integrate import solve_ivp
import math as m
import numpy as np

def compute_Esp_VR(
    phi_diff_matrix: np.ndarray,  # 形状 (num_electrons, num_electrons)
    denominator: np.ndarray,  # 形状 (num_electrons,)
    delta_phi0: float,
    C,
    b,
    num_electrons: int,
    beam_params: list,
) -> np.ndarray:
    # 参数初始化
    F1z_integral = np.zeros(num_electrons)
    # beam_params初始化
    Rn_sqr_values = beam_params[0]
    wp_w = beam_params[1]
    Space_CUT = beam_params[2]

    n_values = np.arange(1, Space_CUT)  # n ∈ [1, Space_cut)

    for n in n_values:
        Rn_sqr = Rn_sqr_values
        term = (np.sin(n * phi_diff_matrix) * Rn_sqr) / (2 * np.pi * n)
        sum_term = np.sum(term / denominator[np.newaxis, :], axis=0) * delta_phi0
        F1z_integral += sum_term

    Esp_vector = (
        -((wp_w / C) ** 2) / (1 + C * b) * F1z_integral
    )  # 形状 (num_electrons,)

    return Esp_vector
