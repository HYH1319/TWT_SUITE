# twt_core.py（核心计算模块）
import math as m
import numpy as np
import Vpc_calc
import sympy
from sympy import *
from scipy.special import iv, kv, jv  # 仅导入贝塞尔函数
from scipy.optimize import root


def detailed_calculation(I, V, Kc, Loss_perunit, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc):
    """返回包含中间结果和最终增益的字典"""
    try:
        Lineparam = simple_calculation(
            I, V, Kc, Loss_perunit, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc
        )
        #         return {
        #     "小信号增益因子C": C,
        #     "互作用长度N": N,
        #     "慢波线单位互作用长度损耗L": L,
        #     "损耗因子d": d,
        #     "等离子体频率降低因子Fn": Fn,
        #     "等离子体频率Wp": Wp,
        #     "空间电荷参量4QC": Wq_over_omegaC_sq,
        #     "非同步参量b": b,
        #     "归一化电子速度Vec": Vec,
        #     "束流归一化尺寸r_beam": r_beam,
        #     "beta_Space": beta_Space,
        #     "Rowe特征值R": R,
        # }
        C = Lineparam["小信号增益因子C"]
        N = Lineparam["互作用长度N"]
        L = Lineparam["慢波线单位互作用长度损耗L"]
        d = Lineparam["损耗因子d"]
        Fn = Lineparam["等离子体频率降低因子Fn"]
        Wp = Lineparam["等离子体频率Wp"]
        Wq_over_omegaC_sq = Lineparam["空间电荷参量4QC"]
        b = Lineparam["非同步参量b"]
        Vec = Lineparam["归一化电子速度Vec"]
        Q = Wq_over_omegaC_sq / (4 * C)

        # ========== 行波管三次方程求解（图片公式） ==========
        # 原方程 δ² = 1/(-b+jd +jδ) - 4QC
        # 重排为三次方程: jδ³ + (−b+jd)δ² + (4QC)jδ + (−1−4QCb+4QCjd) = 0
        # ========== 替换4QC为Wq_over_omegaC_sq ==========
        # 重排为三次方程: jδ³ + jdδ² + (Wq_over_omegaC_sq)jδ + (Wq_over_omegaC_sq)*(j*d-b) -1 = 0
        j = 1j
        coeffs = [
            j,
            j * d - b,
            Wq_over_omegaC_sq * j,
            Wq_over_omegaC_sq * (j * d - b) - 1,
        ]
        roots = np.roots(coeffs)
        sorted_roots = sorted(roots, key=lambda x: x.real, reverse=True)
        sorted_roots[1],sorted_roots[2]=sorted_roots[2],sorted_roots[1]
        print(sorted_roots)
        x1, y1 = sorted_roots[0].real, sorted_roots[0].imag
        x2, y2 = sorted_roots[1].real, sorted_roots[1].imag
        x3, y3 = sorted_roots[2].real, sorted_roots[2].imag

        # 截断附加衰减
        delta1 = sorted_roots[0]
        delta2 = sorted_roots[1]
        delta3 = sorted_roots[2]

        # ========== 核心增益计算与振荡判断 ==========

        # 计算Gmax = A + 54.6*x1*C*N （图片公式5）
        numeratorA = (
            (1 + j * C * delta2)
            * (1 + j * C * delta3)
            * (delta1**2 + 4 * Q * C * (1 + j * C * delta1) ** 2)
        )
        denominatorA = (delta1 - delta2) * (delta1 - delta3)
        A = 20 * np.log10(abs(numeratorA / denominatorA))
        A = -abs(A)
        # A=-9.54
        Gmax = A + 54.6 * x1 * C * N
        if Gmax < 0:
            Gmax = 0
        else:
            Gmax = Gmax

        # 计算Tao = (Gmax - Loss)/2 （图片公式6）
        Loss=Loss_perunit*N_unit
        Tao = (Gmax - Loss) / 2
        # return
        return {
            "小信号增益因子C": C,
            "互作用长度N": N,
            "慢波线单位互作用长度损耗L": L,
            "损耗因子d": d,
            "等离子体频率降低因子Fn": Fn,
            "等离子体频率Wp": Wp,
            "等离子体频率降Fn": Fn,
            "空间电荷参量4QC": Wq_over_omegaC_sq,
            "非同步参量b": b,
            "归一化电子速度Vec": Vec,
            "增幅波第一实数解x1": x1,
            "Gmax": Gmax,
            "慢波线最大反射Tao": Tao,
        }

    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")


def simple_calculation(I, V, Kc, Loss_perunit, p_SWS, N_unit, w, t, Fill_Rate, f0_GHz, Vpc):
    try:
        # 单位转换
        p_SWS *= 1e-3  # mm转m
        w *= 1e-3
        t *= 1e-3
        # ========== 核心计算逻辑（与图片严格一致） ==========

        # 计算C = (I*Kc/(4V))^(1/3) （图片公式4）
        C = m.pow(I * Kc / (4 * V), 1 / 3)

        # 计算omega和Wq （图片等离子体频率公式）
        f0_Hz = f0_GHz * 1e9
        omega = 2 * m.pi * f0_Hz

        yita = 1.7588e11
        erb = 8.854e-12
        S = (w * t * m.pi) / 4

        numerator = I * np.sqrt(yita)
        denominator = S * erb * np.sqrt(2 * V)
        Wp = np.sqrt(numerator / denominator)

        Vec = Vpc_calc(V)  # 归一化电子速度Vec

        # 计算电子波数Beta_e,gamma0与束流归一化尺寸
        beta_e = omega / (Vec * 299792458)  # 电子波数Beta_e
        beta_Space = beta_e  # (omega-Wp) / (Vec * 299792458)# 慢空间电荷波数Beta_Space
        K_wave = omega / 299792458
        gamma0 = np.sqrt(beta_e**2 - K_wave**2)

        r_beam = np.sqrt((w * t / 4))  # 束流归一化尺寸

        # 计算非同步参量b
        b = 1 / C * (Vec - Vpc) / Vpc  # 非同步参量b

        # 计算等离子体频率降Fn
        ##圆形束流等离子体频率降特征方程-特征值数值求解
        if w == t:
            if Fill_Rate == 1:
                Fn_tmp = 2.405 / r_beam  # 圆形束流等离子体频率降特征方程-特征值
            else:
                Fn_tmp = calculate_R(gamma_0=gamma0, a=r_beam * Fill_Rate, b=r_beam)
            Fn = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
        else:
            Fn_tmp = np.sqrt((np.pi / w) ** 2 + (np.pi / t) ** 2)
            Fn = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
            Fn = Fn * Fill_Rate

        R = np.sqrt(
            calc_Rn_sqr_values(beta_Space, 2, p_SWS, r_beam, Fill_Rate)[0]
        )  # R=Fn

        # 计算N=beta_e*p_SWS/(2*m.pi) * N_unit （图片公式3）
        N = beta_e * p_SWS / (2 * m.pi) * N_unit

        # 计算L = Loss/N （图片公式2）
        Loss=Loss_perunit*N_unit
        L = Loss/N
        # 计算d = 0.01836*L/C （图片公式2）
        d = 0.01836 * L / C

        # 空间电荷因子4QC
        Wq_over_omegaC_sq = (Fn * Wp / (omega * C)) ** 2

        return {
            "小信号增益因子C": C,
            "互作用长度N": N,
            "慢波线单位互作用长度损耗L": L,
            "损耗因子d": d,
            "等离子体频率降低因子Fn": Fn,
            "等离子体频率Wp": Wp,
            "空间电荷参量4QC": Wq_over_omegaC_sq,
            "非同步参量b": b,
            "归一化电子速度Vec": Vec,
            "束流归一化尺寸r_beam": r_beam,
            "beta_Space": beta_Space,
            "Rowe特征值R": R,
        }

    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")


def calculate_R(gamma_0, a, b):
    def equation(T, b, a, gamma0):
        Tb = T * b
        gamma0b = gamma0 * b
        gamma0a = gamma0 * a

        # 左侧计算
        J0 = jv(0, Tb)
        J1 = jv(1, Tb)
        lhs = -Tb * J1 / J0

        # 右侧计算
        numerator = gamma0b * (
            iv(1, gamma0b) * kv(0, gamma0a) + iv(0, gamma0a) * kv(1, gamma0b)
        )
        denominator = iv(0, gamma0b) * kv(0, gamma0a) - iv(0, gamma0a) * kv(0, gamma0b)
        rhs = numerator / denominator

        return lhs - rhs

    initial_guess = 1.0  # 初始猜测值

    # 求解
    sol = root(equation, initial_guess, args=(b, a, gamma_0))
    return sol.x[0]


def calc_Rn_sqr_values(beta_space, Space_cut, p_SWS, r_beam, Fill_Rate):
    n_values = np.arange(1, Space_cut)
    beta_n_values = beta_space + 2 * np.pi * (n_values - 1) / p_SWS
    a = r_beam * Fill_Rate

    Iv0_beta_n_a = iv(0, beta_n_values * a)
    Iv1_beta_n_r_beam = iv(1, beta_n_values * r_beam)
    Kv0_beta_n_a = kv(0, beta_n_values * a)
    Kv1_beta_n_r_beam = kv(1, beta_n_values * r_beam)

    Bes_values = Iv1_beta_n_r_beam * Kv0_beta_n_a + Iv0_beta_n_a * Kv1_beta_n_r_beam
    Rn_sqr_values = 1 - (n_values * beta_space * r_beam * Bes_values) / Iv0_beta_n_a
    return Rn_sqr_values


def Vpc_calc(U):
    gamma = U / 5.11e5
    Vp_c = 1 * m.pow(1 - 1 / m.pow(1 + gamma, 2), 0.5)

    return Vp_c


if __name__ == "__main__":
    # 计算物理参数
    inputP = [
        0.3,  # I: Any
        23000,  # V: Any
        3.9233239681172103,  # Kc: Any
        0,  # Loss: Any
        0.5,  # p_SWS: Any
        32,  # N_unit: Any
        0.2,  # w: Any
        0.2,  # t: Any
        1,  # Fill_Rate: Any
        211,  # f0_GHz: Any
        0.2893778968136611,  # Vpc: Any
    ]
    print(simple_calculation(*inputP))
    result = detailed_calculation(*inputP)
    print(result)
