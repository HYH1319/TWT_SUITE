# -*- coding: utf-8 -*-
"""
最终优化版 TWT 非线性带状束流求解器 (Scipy RK45 优化版)

优化策略:
1. 使用 scipy.integrate.solve_ivp 替换手写的 for 循环 RK4。
2. 保持所有公共函数的接口、参数和返回值与原始版本完全一致。
3. 物理方程和求解逻辑保持一字不变。

依赖:
- numpy
- scipy
"""

import numpy as np
from scipy.integrate import solve_ivp


def compute_Esp_VR(
    phi_diff_matrix: np.ndarray,
    denominator: np.ndarray,
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


# ==============================================================================
# 公共 API 函数 (接口与原始版本完全一致)
# ==============================================================================


def solveTWTNOLINE_INIT(C, b, d, beam_params, m, A0, y_end, N_steps=1000):
    """四阶龙格库塔法求解行波管非线性方程组（Scipy优化版）"""
    # ========================= 初始化设置 =========================
    y_start = 0
    y_span = (y_start, y_end)
    t_eval = np.linspace(y_start, y_end, N_steps + 1)

    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # ========================= 初始条件 =========================
    state0 = np.zeros(4 + 2 * m)
    state0[0] = A0
    state0[1] = 0.0
    state0[2] = 0.0
    state0[3] = -b
    state0[4 : 4 + m] = 0.0
    state0[4 + m : 4 + 2 * m] = phi0_grid

    # ========================= ODE系统定义 =========================
    def ode_system(y, state):
        """ODE系统定义，逻辑与原始 compute_derivatives 完全相同"""
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]

        denominator = 1 + 2 * C * u
        sum_cos = np.sum(np.cos(phi) / denominator) * delta_phi0
        sum_sin = np.sum(np.sin(phi) / denominator) * delta_phi0

        # 方程1: d²A/dy²
        term1 = A * ((1 / C - dtheta_dy) ** 2 - ((1 + C * b) / C) ** 2)
        rhs1 = (1 + C * b) / (np.pi * C) * (sum_cos + 2 * C * d * sum_sin)
        d2A_dy2 = term1 + rhs1

        # 方程2: d²θ/dy²
        term_theta = (1 + C * b) / (np.pi * C) * (sum_sin - 2 * C * d * sum_cos)
        d2theta_dy2 = (term_theta - 2 * dA_dy * (dtheta_dy - 1 / C)) / A + 2 * d / C * (
            1 + C * b
        ) ** 2

        # 方程3: ∂u/∂y (含空间电荷项)
        phi_diff = phi[:, np.newaxis] - phi
        Ez = compute_Esp_VR(phi_diff, denominator, delta_phi0, C, b, m, beam_params)
        term_u = A * (1 - C * dtheta_dy) * np.sin(phi) - C * dA_dy * np.cos(phi) + Ez
        du_dy = term_u / denominator

        # 方程4: ∂ϕ/∂y
        dphi_dy = (2 * u) / (1 + 2 * C * u) - dtheta_dy

        # 组合导数向量
        d_state = np.zeros_like(state)
        d_state[0] = dA_dy
        d_state[1] = d2A_dy2
        d_state[2] = dtheta_dy
        d_state[3] = d2theta_dy2
        d_state[4 : 4 + m] = du_dy
        d_state[4 + m : 4 + 2 * m] = dphi_dy

        return d_state

    # ========================= Scipy求解 =========================
    sol = solve_ivp(
        ode_system,
        y_span,
        state0,
        t_eval=t_eval,
        method="RK45",  # 使用经典的 RK45 方法
        rtol=1e-6,
        atol=1e-8,
        vectorized=False,
    )

    # ========================= 提取结果 =========================
    y_values = sol.t
    A_values = sol.y[0, :]
    theta_values = sol.y[2, :]
    dA_dy_values = sol.y[1, :]
    dtheta_dy_values = sol.y[3, :]
    u_now = sol.y[4 : 4 + m, :].T

    # 计算最终的导数值
    final_derivatives = ode_system(y_values[-1], sol.y[:, -1])
    dphi_dy_final = final_derivatives[4 + m : 4 + 2 * m]

    return {
        "y": y_values,
        "A": A_values,
        "theta": theta_values,
        "u_now": u_now,
        "A_Ends": A_values[-1],
        "theta_Ends": theta_values[-1],
        "dA_dy_Ends": dA_dy_values[-1],
        "dtheta_dy_Ends": dtheta_dy_values[-1],
        "u_final": sol.y[4 : 4 + m, -1],
        "phi_final": sol.y[4 + m : 4 + 2 * m, -1],
        "phi0_grid": phi0_grid,
        "dphi_dy": dphi_dy_final,
    }


def solveTWTNOLINE_OUTPUT(
    C,
    b,
    d,
    beam_params,
    m,
    result_y_ends,
    result_A_ends,
    result_dA_dy,
    result_theta,
    result_dtheta_dy,
    result_u_finnal,
    result_phi_finnal,
    y_end,
    N_steps=1000,
):
    """四阶龙格库塔法求解行波管非线性方程组（Scipy优化版）"""
    y_start = result_y_ends
    y_span = (y_start, y_end)
    t_eval = np.linspace(y_start, y_end, N_steps + 1)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    state0 = np.zeros(4 + 2 * m)
    state0[0] = result_A_ends
    state0[1] = result_dA_dy
    state0[2] = result_theta
    state0[3] = result_dtheta_dy
    state0[4 : 4 + m] = result_u_finnal
    state0[4 + m : 4 + 2 * m] = result_phi_finnal

    def ode_system(y, state):
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]
        denominator = 1 + 2 * C * u
        sum_cos = np.sum(np.cos(phi) / denominator) * delta_phi0
        sum_sin = np.sum(np.sin(phi) / denominator) * delta_phi0
        term1 = A * ((1 / C - dtheta_dy) ** 2 - ((1 + C * b) / C) ** 2)
        rhs1 = (1 + C * b) / (np.pi * C) * (sum_cos + 2 * C * d * sum_sin)
        d2A_dy2 = term1 + rhs1
        term_theta = (1 + C * b) / (np.pi * C) * (sum_sin - 2 * C * d * sum_cos)
        d2theta_dy2 = (term_theta - 2 * dA_dy * (dtheta_dy - 1 / C)) / A + 2 * d / C * (
            1 + C * b
        ) ** 2
        phi_diff = phi[:, np.newaxis] - phi
        Ez = compute_Esp_VR(phi_diff, denominator, delta_phi0, C, b, m, beam_params)
        term_u = A * (1 - C * dtheta_dy) * np.sin(phi) - C * dA_dy * np.cos(phi) + Ez
        du_dy = term_u / denominator
        dphi_dy = (2 * u) / (1 + 2 * C * u) - dtheta_dy
        d_state = np.zeros_like(state)
        d_state[0] = dA_dy
        d_state[1] = d2A_dy2
        d_state[2] = dtheta_dy
        d_state[3] = d2theta_dy2
        d_state[4 : 4 + m] = du_dy
        d_state[4 + m : 4 + 2 * m] = dphi_dy
        return d_state

    sol = solve_ivp(
        ode_system,
        y_span,
        state0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        vectorized=False,
    )

    y_values = sol.t
    A_values = sol.y[0, :]
    theta_values = sol.y[2, :]
    dA_dy_values = sol.y[1, :]
    dtheta_dy_values = sol.y[3, :]
    u_now = sol.y[4 : 4 + m, :].T
    final_derivatives = ode_system(y_values[-1], sol.y[:, -1])
    dphi_dy_final = final_derivatives[4 + m : 4 + 2 * m]

    return {
        "y": y_values,
        "A": A_values,
        "theta": theta_values,
        "u_now": u_now,
        "A_Ends": A_values[-1],
        "theta_Ends": theta_values[-1],
        "dA_dy_Ends": dA_dy_values[-1],
        "dtheta_dy_Ends": dtheta_dy_values[-1],
        "u_final": sol.y[4 : 4 + m, -1],
        "phi_final": sol.y[4 + m : 4 + 2 * m, -1],
        "phi0_grid": phi0_grid,
        "dphi_dy": dphi_dy_final,
    }


def solveTWTNOLINE_Drift(
    C,
    b,
    d,
    beam_params,
    m,
    result_y_ends,
    result_A_ends,
    result_dA_dy,
    result_theta,
    result_dtheta_dy,
    result_u_finnal,
    result_phi_finnal,
    y_end,
    N_steps=1000,
):
    """四阶龙格库塔法求解行波管非线性方程组（Scipy优化版）"""
    y_start = result_y_ends
    y_span = (y_start, y_end)
    t_eval = np.linspace(y_start, y_end, N_steps + 1)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    state0 = np.zeros(4 + 2 * m)
    state0[0] = result_A_ends
    state0[1] = result_dA_dy
    state0[2] = result_theta
    state0[3] = result_dtheta_dy
    state0[4 : 4 + m] = result_u_finnal
    state0[4 + m : 4 + 2 * m] = result_phi_finnal

    def ode_system(y, state):
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]
        denominator = 1 + 2 * C * u
        d2A_dy2 = 0.0
        d2theta_dy2 = 0.0
        phi_diff = phi[:, np.newaxis] - phi
        Ez = compute_Esp_VR(phi_diff, denominator, delta_phi0, C, b, m, beam_params)
        term_u = Ez
        du_dy = term_u / denominator
        dphi_dy = (2 * u) / (1 + 2 * C * u) - dtheta_dy
        d_state = np.zeros_like(state)
        d_state[0] = dA_dy
        d_state[1] = d2A_dy2
        d_state[2] = dtheta_dy
        d_state[3] = d2theta_dy2
        d_state[4 : 4 + m] = du_dy
        d_state[4 + m : 4 + 2 * m] = dphi_dy
        return d_state

    sol = solve_ivp(
        ode_system,
        y_span,
        state0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        vectorized=False,
    )

    y_values = sol.t
    A_values = sol.y[0, :]
    theta_values = sol.y[2, :]
    dA_dy_values = sol.y[1, :]
    dtheta_dy_values = sol.y[3, :]
    u_now = sol.y[4 : 4 + m, :].T
    final_derivatives = ode_system(y_values[-1], sol.y[:, -1])
    dphi_dy_final = final_derivatives[4 + m : 4 + 2 * m]

    return {
        "y": y_values,
        "A": A_values,
        "theta": theta_values,
        "u_now": u_now,
        "A_Ends": A_values[-1],
        "theta_Ends": theta_values[-1],
        "dA_dy_Ends": dA_dy_values[-1],
        "dtheta_dy_Ends": dtheta_dy_values[-1],
        "u_final": sol.y[4 : 4 + m, -1],
        "phi_final": sol.y[4 + m : 4 + 2 * m, -1],
        "phi0_grid": phi0_grid,
        "dphi_dy": dphi_dy_final,
    }
