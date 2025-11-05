import numpy as np
import sys

from TWT_CORE_SIMP import simple_calculation

from _TWT_CORE_NOLINE_COMPLEX_VSUPER_MIX import (
    solveTWTNOLINE_OUTPUT,
    solveTWTNOLINE_INIT,
    solveTWTNOLINE_Drift,
)


def calculate_SEGMENT_TWT_NOLINE(
    I,
    V,
    Kc,
    Loss_perunit,
    SectionedSEGMENT_IDX,
    p_SWS,
    N_unit,
    w,
    t,
    Fn_K,
    f0_GHz,
    Vpc,
    para_func,
    P_in=0.1,
    Loss_attu=20,
):
    # ========================= 方法配置 =========================
    def build_input_params(common_params, seg):
        """构建输入参数列表"""
        return [
            common_params["I"],
            common_params["V"],
            seg["Kc"],
            seg["Loss_perunit"],
            seg["p_SWS"],
            seg["len"],
            common_params["w"],
            common_params["t"],
            common_params["Fn_K"],
            common_params["f0_GHz"],
            seg["Vpc"],
        ]

    def handle_initial_segment(params, common_params, calc_result, results, P_in):
        """处理初始段"""
        P_in = P_in  # 输入功率Pin
        P_flux = params["C"] * common_params["I"] * common_params["V"] * 2
        params.update(
            {
                "A0": np.sqrt(P_in / P_flux),
                "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"],
            }
        )
        print(params)
        results.append(solveTWTNOLINE_INIT(**params))

    def handle_attenuator_segment(params, results, seg_idx, Loss_attu):
        """处理衰减段"""
        prev = results[seg_idx - 1]
        params.update(
            {
                "result_y_ends": prev["y"][-1],
                "result_A_ends": prev["A_Ends"] * 10 ** (-Loss_attu / 20),
                "result_dA_dy": prev["dA_dy_Ends"] * 0,
                "result_theta": prev["theta_Ends"],
                "result_dtheta_dy": prev["dtheta_dy_Ends"],
                "result_u_finnal": prev["u_final"],
                "result_phi_finnal": prev["phi_final"],
            }
        )
        results.append(solveTWTNOLINE_Drift(**params))

    def handle_normal_segment(params, results, seg_idx):
        """处理常规段"""
        prev = results[seg_idx - 1]
        params.update(
            {
                "result_y_ends": prev["y"][-1],
                "result_A_ends": prev["A_Ends"],
                "result_dA_dy": prev["dA_dy_Ends"],
                "result_theta": prev["theta_Ends"],
                "result_dtheta_dy": prev["dtheta_dy_Ends"],
                "result_u_finnal": prev["u_final"],
                "result_phi_finnal": prev["phi_final"],
            }
        )
        results.append(solveTWTNOLINE_OUTPUT(**params))

    def process_result(results, C_list, common_params, segments):
        # 功率计算
        P_Out = (
            2
            * common_params["I"]
            * common_params["V"]
            * np.concatenate(
                [C_list[i] * (results[i]["A"] ** 2) for i in range(len(segments))]
            )
        )
        return P_Out

    # ========================= 参数配置 =========================

    num_segments = len(N_unit)
    if isinstance(p_SWS, (int, float)):
        p_SWS = [p_SWS] * num_segments
    p_SWS_full = p_SWS
    print(p_SWS)  # 初始化p_SWS[]

    SEGMENTS = [
        {
            "len": n,
            "Vpc": para_func(p_SWS_full, idx, Vpc, Kc)["Vpc"],  # 传递完整列表和索引
            "Kc": para_func(p_SWS_full, idx, Vpc, Kc)["Kc"],
            "p_SWS": p_sws,
            "Loss_perunit": Loss_perunit,
        }
        for idx, (n, p_sws) in enumerate(zip(N_unit, p_SWS))
    ]
    # 公共参数配置
    COMMON_PARAMS = {
        "I": I,
        "V": V,
        "w": w,
        "t": t,
        "Fn_K": Fn_K,
        "f0_GHz": f0_GHz,
    }

    # ========================= 主计算逻辑 =========================
    results = []
    C_list = []

    for seg_idx, seg in enumerate(SEGMENTS):
        # 参数计算
        input_params = build_input_params(COMMON_PARAMS, seg)
        print(input_params)
        calc_result = simple_calculation(*input_params)

        # 缓存公共参数
        C = calc_result["小信号增益因子C"]
        L = 2 * np.pi * calc_result["互作用长度N"] * C
        C_list.append(C)

        beta_space = calc_result["beta_Space"]
        r_beam = calc_result["束流归一化尺寸r_beam"]
        wp_omega = calc_result["等离子体频率Wp"] / (2 * np.pi * input_params[9] * 1e9)
        Space_cut = 10  # 空间电荷场截断项数

        params = {
            "C": C,
            "b": calc_result["非同步参量b"],
            "d": calc_result["损耗因子d"],
            "wp_w": wp_omega,
            "beta_space": beta_space,
            "r_beam": r_beam,
            "Fill_Rate": COMMON_PARAMS["Fn_K"],
            "p_SWS": seg["p_SWS"] * 1e-3,
            "Space_cut": Space_cut,
            "m": 50,
            "y_end": L + (results[-1]["y"][-1] if seg_idx > 0 else 0),
        }

        # 分段处理（添加非法索引防护）
        if seg_idx == 0:
            handle_initial_segment(params, COMMON_PARAMS, calc_result, results, P_in)
        elif seg_idx in SectionedSEGMENT_IDX and seg_idx < len(SEGMENTS):
            handle_attenuator_segment(params, results, seg_idx, Loss_attu)
        else:
            handle_normal_segment(params, results, seg_idx)

    P_End = process_result(results, C_list, COMMON_PARAMS, SEGMENTS)
    return {"输出功率P_out": P_End[-1]}


if __name__ == "__main__":
    # 修正后的参数计算函数
    def para_func(p_SWS, idx, Vpc, Kc):
        """计算单段参数（使用完整列表和当前索引）"""
        current_p = p_SWS[idx]
        first_p = p_SWS[0]
        Vpc_adjusted = Vpc + 0.3 * (current_p - first_p) / first_p * Vpc
        Kc_adjusted = Kc + 0.9 * (current_p - first_p) / first_p * Kc
        return {"Vpc": Vpc_adjusted, "Kc": Kc_adjusted}

    # 输入参数
    inputP = [
        0.30,  # I
        23000,  # V
        3.6,  # Kc
        0,  # Loss_perunit
        [1],  # SectionedSEGMENT_IDX (衰减段索引,值一般为奇数)
        [0.50,0.495,0.49,0.485,0.48],  # p_SWS,大小为n的数组维度与N_unit保持一致
        [30, 5, 5, 5,5],  # N_unit，大小为n的数组
        0.20,  # w
        0.20,  # t
        1,  # Fill_Rate
        211,  # f0_GHz
        0.288,  # Vpc
        para_func,  # 参数计算函数
    ]

    result = calculate_SEGMENT_TWT_NOLINE(*inputP)
    print("计算结果:", result)
