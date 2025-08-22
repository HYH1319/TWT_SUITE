# twt_core.py（核心计算模块）
import math as m
import numpy as np
from scipy.special import jv  # 仅导入贝塞尔函数
from scipy.optimize import root_scalar
from TWT_CORE_SIMP import simple_calculation


def detailed_calculation(
    I, V, Kc, Loss_perunit, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc
):
    """返回包含中间结果和最终增益的字典"""
    try:
        # 通用参数计算
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
        beta_e=Lineparam["beta_Space"]
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
        Gmax = 54.6 * x1 * C * N
        if Gmax < 0:
            Gmax = 0
        else:
            Gmax = Gmax

        # 计算Tao = (Gmax - Loss_perunit)/2 （图片公式6）
        Tao = (Gmax - Loss_perunit) / 2

    # 截断附加衰减Ab(有限距离Lenth_att)
        Lenth_att = 0
        theta_Ab = beta_e * C * Lenth_att * np.sqrt(Wq_over_omegaC_sq)  # 衰减角
        Amp_Attu = (
            (delta2 * delta3 - delta1 * delta2 - delta1 * delta3 - 4 * Q * C)
            * m.cos(theta_Ab)
            + (
                (delta2 + delta3 - delta1) * np.sqrt(Wq_over_omegaC_sq)
                + delta1 * delta2 * delta3 / np.sqrt(Wq_over_omegaC_sq)
            )
            * m.sin(theta_Ab)
        ) / ((delta1 - delta2) * (delta1 - delta3))
        Ab = 20 * m.log10(Amp_Attu)  # 有限距离截断附加衰减Ab
    
    #return
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
            "增幅波第一实数解x1": x1,
            "线性最大增益Gmax": Gmax,
            "慢波线最大反射Tao": Tao,
            "衰减降低增益量Ab": Ab,
            "初始化调制增益降低量A": A,
        }

    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")
