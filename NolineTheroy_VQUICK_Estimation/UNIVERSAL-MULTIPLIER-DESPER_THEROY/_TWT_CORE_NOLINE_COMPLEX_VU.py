import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import math as m
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from _TWT_CORE_NOLINE_CALC_Esp import compute_Esp_VR


def solveTWTNOLINE_INIT(C, b, d, beam_params, m, A0, y_end, N_steps=1000):
    """四阶龙格库塔法求解行波管非线性方程组

    参数：
        C: 增益参量
        b: 速度非同步参量
        d: 线路损耗参量
        wp_w: 相对等离子体频率
        Rn: 空间电荷系数
        m: 电子离散数量
        A0: 初始振幅
        y_end: 模拟终止位置 (默认10)
        N_steps: 总步数 (默认1000)

    返回：
        result: 包含求解结果的字典
    """
    # ========================= 初始化设置 =========================
    y_start = 0
    h = (y_end - y_start) / N_steps  # 计算步长

    # 离散初始相位ϕ₀ (式18.3.8)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # ========================= 初始条件 =========================
    state = np.zeros(4 + 2 * m)
    state[0] = A0  # A(0) = A₀ (式18.3.1)
    state[1] = 0.0  # dA/dy(0) = 0 (式18.3.2)
    state[2] = 0.0  # θ(0) = 0 (式18.3.3)
    state[3] = -b  # dθ/dy(0) = -b (式18.3.4)
    state[4 : 4 + m] = 0.0  # u(y=0, ϕ₀) = 0 (式18.3.6)
    state[4 + m : 4 + 2 * m] = phi0_grid  # ϕ(y=0, ϕ₀) = ϕ₀ (式18.3.7)

    # ========================= 辅助函数 =========================
    def compute_derivatives(y, state):
        """计算状态向量导数的嵌套函数"""
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]

        # 计算积分项
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
        phi_diff = phi[:, np.newaxis] - phi  # 相位差矩阵

        Ez = compute_Esp_VR(phi_diff, denominator, delta_phi0, C, b, m, beam_params)
        term_u = (
            A * (1 - C * dtheta_dy) * np.sin(phi) - C * dA_dy * np.cos(phi) + Ez
        )  # ((wp_w/C)**2)/(1 + C*b) * F1z_integral
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

    # ========================= RK4 求解 =========================
    y = y_start
    y_values = np.linspace(y_start, y_end, N_steps + 1)
    A_values = np.zeros(N_steps + 1)
    theta_values = np.zeros(N_steps + 1)
    dA_dy_values = np.zeros(N_steps + 1)
    dtheta_dy_values = np.zeros(N_steps + 1)
    u_now = np.zeros((N_steps + 1, m))

    # 存储初始值
    A_values[0] = state[0]
    theta_values[0] = state[2]
    dA_dy_values[0] = state[1]
    dtheta_dy_values[0] = state[3]
    u_now[0, :] = state[4 : 4 + m]

    for i in range(1, N_steps + 1):
        # RK4步骤
        k1 = compute_derivatives(y, state)
        k2 = compute_derivatives(y + h / 2, state + h * k1 / 2)
        k3 = compute_derivatives(y + h / 2, state + h * k2 / 2)
        k4 = compute_derivatives(y + h, state + h * k3)

        state += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y += h

        # 存储结果
        A_values[i] = state[0]
        theta_values[i] = state[2]

        dA_dy_values[i] = state[1]
        dtheta_dy_values[i] = state[3]
        u_now[i, :] = state[4 : 4 + m]

    return {
        "y": y_values,
        "A": A_values,
        "theta": theta_values,
        "u_now": u_now,
        "A_Ends": A_values[-1],
        "theta_Ends": theta_values[-1],
        "dA_dy_Ends": dA_dy_values[-1],
        "dtheta_dy_Ends": dtheta_dy_values[-1],
        "u_final": state[4 : 4 + m],
        "phi_final": state[4 + m : 4 + 2 * m],
        "phi0_grid": phi0_grid,
        "dphi_dy": k1[4 + m : 4 + 2 * m],
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
    """四阶龙格库塔法求解行波管非线性方程组

    参数：
        C: 增益参量
        b: 速度非同步参量
        d: 线路损耗参量
        wp_w: 相对等离子体频率
        Rn: 空间电荷系数
        m: 电子离散数量
        A0: 初始振幅
        y_end: 模拟终止位置 (默认10)
        N_steps: 总步数 (默认1000)

    返回：
        result: 包含求解结果的字典
    """
    # ========================= 初始化设置 =========================
    y_start = result_y_ends
    h = (y_end - y_start) / N_steps  # 计算步长

    # 离散初始相位ϕ₀ (式18.3.8)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # ========================= 初始条件 =========================
    state = np.zeros(4 + 2 * m)
    state[0] = result_A_ends  # A(0) = A₀ (式18.3.1)
    state[1] = result_dA_dy  # dA/dy(0) = 0 (式18.3.2)
    state[2] = result_theta  # θ(0) = 0 (式18.3.3)
    state[3] = result_dtheta_dy  # dθ/dy(0) = -b (式18.3.4)
    state[4 : 4 + m] = result_u_finnal  # u(y=0, ϕ₀) = 0 (式18.3.6)
    state[4 + m : 4 + 2 * m] = result_phi_finnal  # ϕ(y=0, ϕ₀) = ϕ₀ (式18.3.7)

    # ========================= 辅助函数 =========================
    def compute_derivatives(y, state):
        """计算状态向量导数的嵌套函数"""
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]

        # 计算积分项
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
        phi_diff = phi[:, np.newaxis] - phi  # 相位差矩阵

        Ez = compute_Esp_VR(phi_diff, denominator, delta_phi0, C, b, m, beam_params)
        term_u = (
            A * (1 - C * dtheta_dy) * np.sin(phi) - C * dA_dy * np.cos(phi) + Ez
        )  # ((wp_w/C)**2)/(1 + C*b) * F1z_integral
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

    # ========================= RK4 求解 =========================
    y = y_start
    y_values = np.linspace(y_start, y_end, N_steps + 1)
    A_values = np.zeros(N_steps + 1)
    theta_values = np.zeros(N_steps + 1)
    dA_dy_values = np.zeros(N_steps + 1)
    dtheta_dy_values = np.zeros(N_steps + 1)
    u_now = np.zeros((N_steps + 1, m))

    # 存储初始值
    A_values[0] = state[0]
    theta_values[0] = state[2]
    dA_dy_values[0] = state[1]
    dtheta_dy_values[0] = state[3]
    u_now[0, :] = state[4 : 4 + m]

    for i in range(1, N_steps + 1):
        # RK4步骤
        k1 = compute_derivatives(y, state)
        k2 = compute_derivatives(y + h / 2, state + h * k1 / 2)
        k3 = compute_derivatives(y + h / 2, state + h * k2 / 2)
        k4 = compute_derivatives(y + h, state + h * k3)

        state += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y += h

        # 存储结果
        A_values[i] = state[0]
        theta_values[i] = state[2]

        dA_dy_values[i] = state[1]
        dtheta_dy_values[i] = state[3]
        u_now[i, :] = state[4 : 4 + m]

    return {
        "y": y_values,
        "A": A_values,
        "theta": theta_values,
        "u_now": u_now,
        "A_Ends": A_values[-1],
        "theta_Ends": theta_values[-1],
        "dA_dy_Ends": dA_dy_values[-1],
        "dtheta_dy_Ends": dtheta_dy_values[-1],
        "u_final": state[4 : 4 + m],
        "phi_final": state[4 + m : 4 + 2 * m],
        "phi0_grid": phi0_grid,
        "dphi_dy": k1[4 + m : 4 + 2 * m],
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
    """四阶龙格库塔法求解行波管非线性方程组

    参数：
        C: 增益参量
        b: 速度非同步参量
        d: 线路损耗参量
        wp_w: 相对等离子体频率
        Rn: 空间电荷系数
        m: 电子离散数量
        A0: 初始振幅
        y_end: 模拟终止位置 (默认10)
        N_steps: 总步数 (默认1000)

    返回：
        result: 包含求解结果的字典
    """
    # ========================= 初始化设置 =========================
    y_start = result_y_ends
    h = (y_end - y_start) / N_steps  # 计算步长

    # 离散初始相位ϕ₀ (式18.3.8)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # ========================= 初始条件 =========================
    state = np.zeros(4 + 2 * m)
    state[0] = result_A_ends  # A(0) = A₀ (式18.3.1)
    state[1] = result_dA_dy  # dA/dy(0) = 0 (式18.3.2)
    state[2] = result_theta  # θ(0) = 0 (式18.3.3)
    state[3] = result_dtheta_dy  # dθ/dy(0) = -b (式18.3.4)
    state[4 : 4 + m] = result_u_finnal  # u(y=0, ϕ₀) = 0 (式18.3.6)
    state[4 + m : 4 + 2 * m] = result_phi_finnal  # ϕ(y=0, ϕ₀) = ϕ₀ (式18.3.7)

    # ========================= 辅助函数 =========================
    def compute_derivatives(y, state):
        """计算状态向量导数的嵌套函数"""
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]

        # 计算积分项
        denominator = 1 + 2 * C * u

        # 方程1: d²A/dy²
        d2A_dy2 = 0

        # 方程2: d²θ/dy²
        d2theta_dy2 = 0

        # 方程3: ∂u/∂y (含空间电荷项)
        phi_diff = phi[:, np.newaxis] - phi  # 相位差矩阵

        Ez = compute_Esp_VR(phi_diff, denominator, delta_phi0, C, b, m, beam_params)
        term_u = 0 + Ez  # ((wp_w/C)**2)/(1 + C*b) * F1z_integral
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

    # ========================= RK4 求解 =========================
    y = y_start
    y_values = np.linspace(y_start, y_end, N_steps + 1)
    A_values = np.zeros(N_steps + 1)
    theta_values = np.zeros(N_steps + 1)
    dA_dy_values = np.zeros(N_steps + 1)
    dtheta_dy_values = np.zeros(N_steps + 1)
    u_now = np.zeros((N_steps + 1, m))

    # 存储初始值
    A_values[0] = state[0]
    theta_values[0] = state[2]
    dA_dy_values[0] = state[1]
    dtheta_dy_values[0] = state[3]
    u_now[0, :] = state[4 : 4 + m]

    for i in range(1, N_steps + 1):
        # RK4步骤
        k1 = compute_derivatives(y, state)
        k2 = compute_derivatives(y + h / 2, state + h * k1 / 2)
        k3 = compute_derivatives(y + h / 2, state + h * k2 / 2)
        k4 = compute_derivatives(y + h, state + h * k3)

        state += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y += h

        # 存储结果
        A_values[i] = state[0]
        theta_values[i] = state[2]

        dA_dy_values[i] = state[1]
        dtheta_dy_values[i] = state[3]
        u_now[i, :] = state[4 : 4 + m]

    return {
        "y": y_values,
        "A": A_values,
        "theta": theta_values,
        "u_now": u_now,
        "A_Ends": A_values[-1],
        "theta_Ends": theta_values[-1],
        "dA_dy_Ends": dA_dy_values[-1],
        "dtheta_dy_Ends": dtheta_dy_values[-1],
        "u_final": state[4 : 4 + m],
        "phi_final": state[4 + m : 4 + 2 * m],
        "phi0_grid": phi0_grid,
        "dphi_dy": k1[4 + m : 4 + 2 * m],
    }

