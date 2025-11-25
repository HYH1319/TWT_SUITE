import numpy as np
from scipy.integrate import solve_ivp
from TWT_CORE_SIMP import calc_Rn_sqr_values


def solveTWTNOLINE_INIT(
    C,
    b,
    d,
    wp_w,
    beta_space,
    r_beam,
    Fill_Rate,
    m,
    A0,
    y_end,
    Space_cut,
    N_steps=1000,
):
    """四阶龙格库塔法求解行波管非线性方程组（scipy优化版）

    参数：
        C: 增益参量
        b: 速度非同步参量
        d: 线路损耗参量
        wp_w: 相对等离子体频率
        beta_space: 电子波数
        r_beam：束流半径
        Fill_Rate：填充率倒数
        p_SWS：高频周期长度
        m: 电子离散数量
        A0: 初始振幅
        y_end: 模拟终止位置 (默认10)
        N_steps: 总步数 (默认1000)

    返回：
        result: 包含求解结果的字典
    """
    # ========================= 初始化设置 =========================
    y_start = 0
    y_span = (y_start, y_end)
    t_eval = np.linspace(y_start, y_end, N_steps + 1)

    # 离散初始相位ϕ₀ (式18.3.8)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # 预计算空间电荷相关参数
    n_values = np.arange(1, Space_cut)
    Rn_sqr_values = calc_Rn_sqr_values(
        beta_space, Space_cut, r_beam, Fill_Rate
    )

    # ========================= 初始条件 =========================
    state0 = np.zeros(4 + 2 * m)
    state0[0] = A0  # A(0) = A₀ (式18.3.1)
    state0[1] = 0.0  # dA/dy(0) = 0 (式18.3.2)
    state0[2] = 0.0  # θ(0) = 0 (式18.3.3)
    state0[3] = -b  # dθ/dy(0) = -b (式18.3.4)
    state0[4 : 4 + m] = 0.0  # u(y=0, ϕ₀) = 0 (式18.3.6)
    state0[4 + m : 4 + 2 * m] = phi0_grid  # ϕ(y=0, ϕ₀) = ϕ₀ (式18.3.7)

    # ========================= ODE系统 =========================
    def ode_system(y, state):
        """ODE系统定义"""
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
        F1z_integral = np.zeros(m)

        for n_idx, n in enumerate(n_values):
            Rn_sqr = Rn_sqr_values[n_idx]
            term = (np.sin(n * phi_diff) * Rn_sqr) / (2 * np.pi * n)
            sum_term = np.sum(term / denominator[np.newaxis, :], axis=0) * delta_phi0
            F1z_integral += sum_term

        term_u = (
            A * (1 - C * dtheta_dy) * np.sin(phi)
            - C * dA_dy * np.cos(phi)
            - ((wp_w / C) ** 2) / (1 + C * b) * F1z_integral
        )
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

    # ========================= scipy求解 =========================
    # 使用DOP853算法（8阶Runge-Kutta），精度高且稳定
    sol = solve_ivp(
        ode_system,
        y_span,
        state0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-4,
        atol=1e-5,
        vectorized=False
    )

    # ========================= 提取结果 =========================
    y_values = sol.t
    A_values = sol.y[0, :]
    theta_values = sol.y[2, :]
    dA_dy_values = sol.y[1, :]
    dtheta_dy_values = sol.y[3, :]
    u_now = sol.y[4:4+m, :].T

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
        "u_final": sol.y[4:4+m, -1],
        "phi_final": sol.y[4+m:4+2*m, -1],
        "phi0_grid": phi0_grid,
        "dphi_dy": dphi_dy_final,
    }


def solveTWTNOLINE_OUTPUT(
    C,
    b,
    d,
    wp_w,
    beta_space,
    r_beam,
    Fill_Rate,
    m,
    result_y_ends,
    result_A_ends,
    result_dA_dy,
    result_theta,
    result_dtheta_dy,
    result_u_finnal,
    result_phi_finnal,
    y_end,
    Space_cut,
    N_steps=1000,
):
    """四阶龙格库塔法求解行波管非线性方程组（scipy优化版）

    参数：
        C: 增益参量
        b: 速度非同步参量
        d: 线路损耗参量
        wp_w: 相对等离子体频率
        beta_space: 电子波数
        r_beam：束流半径
        Fill_Rate：填充率倒数
        p_SWS：高频周期长度
        m: 电子离散数量
        result_y_ends: 起始位置
        result_A_ends: 起始振幅
        result_dA_dy: 起始振幅导数
        result_theta: 起始相位
        result_dtheta_dy: 起始相位导数
        result_u_finnal: 起始u值
        result_phi_finnal: 起始phi值
        y_end: 模拟终止位置 (默认10)
        N_steps: 总步数 (默认1000)

    返回：
        result: 包含求解结果的字典
    """
    # ========================= 初始化设置 =========================
    y_start = result_y_ends
    y_span = (y_start, y_end)
    t_eval = np.linspace(y_start, y_end, N_steps + 1)

    # 离散初始相位ϕ₀ (式18.3.8)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # 预计算空间电荷相关参数
    n_values = np.arange(1, Space_cut)
    Rn_sqr_values = calc_Rn_sqr_values(
        beta_space, Space_cut, r_beam, Fill_Rate
    )

    # ========================= 初始条件 =========================
    state0 = np.zeros(4 + 2 * m)
    state0[0] = result_A_ends  # A(0) = A₀ (式18.3.1)
    state0[1] = result_dA_dy  # dA/dy(0) = 0 (式18.3.2)
    state0[2] = result_theta  # θ(0) = 0 (式18.3.3)
    state0[3] = result_dtheta_dy  # dθ/dy(0) = -b (式18.3.4)
    state0[4 : 4 + m] = result_u_finnal  # u(y=0, ϕ₀) = 0 (式18.3.6)
    state0[4 + m : 4 + 2 * m] = result_phi_finnal  # ϕ(y=0, ϕ₀) = ϕ₀ (式18.3.7)

    # ========================= ODE系统 =========================
    def ode_system(y, state):
        """ODE系统定义"""
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
        F1z_integral = np.zeros(m)

        for n_idx, n in enumerate(n_values):
            Rn_sqr = Rn_sqr_values[n_idx]
            term = (np.sin(n * phi_diff) * Rn_sqr) / (2 * np.pi * n)
            sum_term = np.sum(term / denominator[np.newaxis, :], axis=0) * delta_phi0
            F1z_integral += sum_term

        term_u = (
            A * (1 - C * dtheta_dy) * np.sin(phi)
            - C * dA_dy * np.cos(phi)
            - ((wp_w / C) ** 2) / (1 + C * b) * F1z_integral
        )
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

    # ========================= scipy求解 =========================
    sol = solve_ivp(
        ode_system,
        y_span,
        state0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-4,
        atol=1e-5,
        vectorized=False
    )

    # ========================= 提取结果 =========================
    y_values = sol.t
    A_values = sol.y[0, :]
    theta_values = sol.y[2, :]
    dA_dy_values = sol.y[1, :]
    dtheta_dy_values = sol.y[3, :]
    u_now = sol.y[4:4+m, :].T

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
        "u_final": sol.y[4:4+m, -1],
        "phi_final": sol.y[4+m:4+2*m, -1],
        "phi0_grid": phi0_grid,
        "dphi_dy": dphi_dy_final,
    }


def solveTWTNOLINE_Drift(
    C,
    b,
    d,
    wp_w,
    beta_space,
    r_beam,
    Fill_Rate,
    m,
    result_y_ends,
    result_A_ends,
    result_dA_dy,
    result_theta,
    result_dtheta_dy,
    result_u_finnal,
    result_phi_finnal,
    y_end,
    Space_cut,
    N_steps=1000,
):
    """四阶龙格库塔法求解行波管非线性方程组（scipy优化版）

    参数：
        C: 增益参量
        b: 速度非同步参量
        d: 线路损耗参量
        wp_w: 相对等离子体频率
        beta_space: 电子波数
        r_beam：束流半径
        Fill_Rate：填充率倒数
        p_SWS：高频周期长度
        m: 电子离散数量
        result_y_ends: 起始位置
        result_A_ends: 起始振幅
        result_dA_dy: 起始振幅导数
        result_theta: 起始相位
        result_dtheta_dy: 起始相位导数
        result_u_finnal: 起始u值
        result_phi_finnal: 起始phi值
        y_end: 模拟终止位置 (默认10)
        N_steps: 总步数 (默认1000)

    返回：
        result: 包含求解结果的字典
    """
    # ========================= 初始化设置 =========================
    y_start = result_y_ends
    y_span = (y_start, y_end)
    t_eval = np.linspace(y_start, y_end, N_steps + 1)

    # 离散初始相位ϕ₀ (式18.3.8)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # 预计算空间电荷相关参数
    n_values = np.arange(1, Space_cut)
    Rn_sqr_values = calc_Rn_sqr_values(
        beta_space, Space_cut, r_beam, Fill_Rate
    )

    # ========================= 初始条件 =========================
    state0 = np.zeros(4 + 2 * m)
    state0[0] = result_A_ends  # A(0) = A₀ (式18.3.1)
    state0[1] = result_dA_dy  # dA/dy(0) = 0 (式18.3.2)
    state0[2] = result_theta  # θ(0) = 0 (式18.3.3)
    state0[3] = result_dtheta_dy  # dθ/dy(0) = -b (式18.3.4)
    state0[4 : 4 + m] = result_u_finnal  # u(y=0, ϕ₀) = 0 (式18.3.6)
    state0[4 + m : 4 + 2 * m] = result_phi_finnal  # ϕ(y=0, ϕ₀) = ϕ₀ (式18.3.7)

    # ========================= ODE系统 =========================
    def ode_system(y, state):
        """ODE系统定义（漂移区）"""
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]

        # 计算积分项
        denominator = 1 + 2 * C * u

        # 方程1: d²A/dy² (漂移区为0)
        d2A_dy2 = 0

        # 方程2: d²θ/dy² (漂移区为0)
        d2theta_dy2 = 0

        # 方程3: ∂u/∂y (含空间电荷项)
        phi_diff = phi[:, np.newaxis] - phi  # 相位差矩阵
        F1z_integral = np.zeros(m)

        for n_idx, n in enumerate(n_values):
            Rn_sqr = Rn_sqr_values[n_idx]
            term = (np.sin(n * phi_diff) * Rn_sqr) / (2 * np.pi * n)
            sum_term = np.sum(term / denominator[np.newaxis, :], axis=0) * delta_phi0
            F1z_integral += sum_term

        term_u = 0 - ((wp_w / C) ** 2) / (1 + C * b) * F1z_integral
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

    # ========================= scipy求解 =========================
    sol = solve_ivp(
        ode_system,
        y_span,
        state0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-4,
        atol=1e-5,
        vectorized=False
    )

    # ========================= 提取结果 =========================
    y_values = sol.t
    A_values = sol.y[0, :]
    theta_values = sol.y[2, :]
    dA_dy_values = sol.y[1, :]
    dtheta_dy_values = sol.y[3, :]
    u_now = sol.y[4:4+m, :].T

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
        "u_final": sol.y[4:4+m, -1],
        "phi_final": sol.y[4+m:4+2*m, -1],
        "phi0_grid": phi0_grid,
        "dphi_dy": dphi_dy_final,
    }

