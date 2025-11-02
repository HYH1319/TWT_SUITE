import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from TWT_CORE_SIMP import simple_calculation
import math as m
import matplotlib.pyplot as plt
import numpy as np
from _TWT_CORE_NOLINE_COMPLEX_VSHEETBEAM import solveTWTNOLINE_INIT

def Noline_CORE_CALC(I, V, Kc, Loss_perunit, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc, P_in=0.1):
    try:
        inputP = [
            I,
            V,
            Kc,
            Loss_perunit,
            p_SWS,
            N_unit,
            w,
            t,
            Fn_K,
            f0_GHz,
            Vpc,
        ]
        print(inputP)
        initparamater = simple_calculation(*inputP)

        C = initparamater["小信号增益因子C"]
        # 行波管增益参数C
        b = initparamater["非同步参量b"]
        # 行波管非同步参量b
        d = initparamater["损耗因子d"]
        # 损耗系数d

        #组装beam_params
        if (
            inputP[6] == inputP[7]
        ):  # 行波管等离子体频率降低系数R,ShitBeam用Fn,CicularBeam用Rowe特征值R
            R = initparamater["Rowe特征值R"]
            Rn_sqr_values=R**2
        else:
            Rn_sqr_values = initparamater["等离子体频率降低因子Fn"]

        wp_w=initparamater["等离子体频率Wp"]/(2*np.pi*inputP[9]*1e9)
        Space_CUT=10

        beam_params=[Rn_sqr_values,wp_w,Space_CUT]

        L = 2 * np.pi * initparamater["互作用长度N"] * C

        P_flux = C * inputP[0] * inputP[1] * 2
        A_0 = np.sqrt(P_in / P_flux)

        # ================================
        # 参数设置 (根据实际需求修改)

        params = {
            "C": C,  # 增益参量
            "b": b,  # 速度非同步参量
            "d": d,  # 线路损耗参量
            "beam_params": beam_params,  # 行波管束流参数
            "m": 50,  # 电子离散数量
            "A0": A_0,  # 初始振幅
            "y_end": L,
        }
        print(params)

        # 调用求解器
        result = solveTWTNOLINE_INIT(
            **params
        )  #     (function) def solveTWTNOLINE_INIT(C: Any,b: Any,d: Any,wp_w: Any,beta_e: Any,r_beam: Any,m: Any,A0: Any,y_end: Any,N_steps: int = 1000# ) -> dict[str, Any]

        # ========================= 结果后处理 =========================
        P_Out = C * inputP[0] * inputP[1] * 2 * (result["A"][-1]) ** 2
        Eff = 2 * C * max(result["A"]) ** 2 * 100
        P_max = C * inputP[0] * inputP[1] * 2 * (max(result["A"])) ** 2
        return {"Pmax": P_max, "P_out": P_Out, "Maxiumum_eff": Eff}

    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")


if __name__ == "__main__":
    # 计算物理参数
    inputP = [
        0.3,  # I: Any
        23000,  # V: Any
        3.6,  # Kc: Any
        0,  # Loss_perunit: Any
        0.5,  # p_SWS: Any
        50,  # N_unit: Any
        0.2,  # w: Any
        0.2,  # t: Any
        1,  # Fill_Rate: Any
        211,  # f0_GHz: Any
        0.288,  # Vpc: Any
    ]
    TWTresult = Noline_CORE_CALC(*inputP)
    print(TWTresult)
