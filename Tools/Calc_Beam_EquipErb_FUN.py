import sys
import math as m
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")


from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas

from TWT_CORE_SIMP import Vpc_calc, calculate_R

# 物理常量
YITA = 1.7588e11  # 电子电荷质量比
ERB = 8.854e-12  # 真空介电常数 (F/m)


def calculate_dielectric(params, A_list, u_list):
    """
    计算相对介电常数

    参数:
    params (dict): 基本参数字典
    A_list (list): A值数组
    u_list (list): u值数组

    返回:
    dict: 包含所有计算结果
    """
    # 解包参数
    I = params["I"]
    U = params["U"]
    f0_Hz = params["f0_GHz"] * 1e9
    Fill_Rate = params["Fill_Rate"]
    w_beam = params["w_beam"] * 1e-3  # mm 转 m
    t_beam = params["t_beam"] * 1e-3  # mm 转 m

    # 转换为数组
    A_arr = np.array(A_list)
    u_arr = np.array(u_list)

    # ========== 核心计算逻辑（与图片严格一致） ==========

    # 计算omega和Wq （图片等离子体频率公式）
    omega = 2 * m.pi * f0_Hz

    yita = 1.7588e11
    erb = 8.854e-12
    S = (w_beam * t_beam * m.pi) / 4

    numerator = I * np.sqrt(yita)
    denominator = S * erb * np.sqrt(2 * U)
    Wp = np.sqrt(numerator / denominator)

    Vec = Vpc_calc(U)  # 归一化电子速度Vec

    # 计算电子波数Beta_e,gamma0与束流归一化尺寸
    beta_e = omega / (Vec * 299792458)  # 电子波数Beta_e
    K_wave = omega / 299792458
    gamma0 = np.sqrt(beta_e**2 - K_wave**2)

    r_beam = np.sqrt((w_beam * t_beam / 4))  # 束流归一化尺寸

    # 计算等离子体频率降Fn
    ##圆形束流等离子体频率降特征方程-特征值数值求解
    if w_beam == t_beam:
        if Fill_Rate == 1:
            Fn_tmp = 2.405 / r_beam  # 圆形束流等离子体频率降特征方程-特征值
        else:
            Fn_tmp = calculate_R(gamma_0=gamma0, a=r_beam * Fill_Rate, b=r_beam)
        Fn = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
    else:
        Fn_tmp = np.sqrt((np.pi / w_beam) ** 2 + (np.pi / t_beam) ** 2)
        Fn = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
        Fn = Fn * Fill_Rate

    epsilon_rs = 1 - (Fn * Wp / YITA) * u_arr / (A_arr)
    print(f"\n等离子体频率降低因子是{Fn},等离子体频率是{Wp}Hz,等效介电常数是{epsilon_rs}")

    # 返回结果
    return {
        "Wp": Wp,
        "Vec": Vec,
        "beta_e": beta_e,
        "epsilon_rs": epsilon_rs,
        "A": A_arr,
        "u": u_arr,
    }
