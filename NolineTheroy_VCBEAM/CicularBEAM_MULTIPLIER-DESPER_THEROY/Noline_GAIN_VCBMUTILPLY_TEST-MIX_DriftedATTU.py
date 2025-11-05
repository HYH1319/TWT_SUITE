

import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.integrate import solve_ivp
from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VCBEAM_MIX import solveTWTNOLINE_OUTPUT, solveTWTNOLINE_INIT,solveTWTNOLINE_Drift

def main():
    # ========================= 多段参数配置 =========================
    SEGMENTS = [
        {"len": 30, "Vpc": 0.29, "p_SWS": 0.50, "Kc": 3.6, "Loss_perunit": 0,"Fn_K": 2, "type": "initial"},
        {"len": 5, "Vpc": 0.285244, "p_SWS": 0.490, "Kc": 3.4848, "Loss_perunit": 0.0,"Fn_K": 2, "type": "attenuator"},
        {"len": 5, "Vpc": 0.287, "p_SWS": 0.146, "Kc": 3.0, "Loss_perunit": 0.1,"Fn_K": 1.2, "type": "O"},#倍频段起始
        {"len": 5, "Vpc": 0.2773284931506849, "p_SWS": 0.140, "Kc": 2.802739726027398, "Loss_perunit": 0.1,"Fn_K": 1.2, "type": "O"},
        {"len": 5, "Vpc": 0.30150726027397257, "p_SWS": 0.155, "Kc": 3.2958904109589042, "Loss_perunit": 0.1,"Fn_K": 1.2, "type": "O"},
    ]
    Loss_attu = 15

    # ========================= 全局参数 =========================
    COMMON_PARAMS = {
        "I": 0.3, "V": 23000, 
        "w": 0.2, "t": 0.2,"f0_GHz": 211, 
    }

    # ========================= 主计算逻辑 =========================
    results = []
    C_list = []
    global_vars = {}  # 存储跨段共享的变量

    for seg_idx, seg in enumerate(SEGMENTS):
        # 参数计算
        input_params = build_input_params(COMMON_PARAMS, seg)
        calc_result = simple_calculation(*input_params)
        print(f"\n段{seg_idx}计算参数:\n{calc_result}")
        
        # 缓存公共参数
        C = calc_result["小信号增益因子C"]
        L = 2 * np.pi * calc_result["互作用长度N"] * C
        C_list.append(C)
        params = {
            "C": C, 
            "b": calc_result["非同步参量b"],
            "d": calc_result["损耗因子d"],
            "wp_w": calc_result["等离子体频率Wp"] / (2 * np.pi * COMMON_PARAMS["f0_GHz"] * 1e9),
            "beta_space": calc_result["beta_Space"],
            "r_beam": calc_result["束流归一化尺寸r_beam"],
            "Fill_Rate": seg["Fn_K"],
            "p_SWS": seg["p_SWS"]*1e-3,
            "m": 50,
            "Space_cut": 10,
            "y_end": L + (results[-1]["y"][-1] if seg_idx > 0 else 0)
        }

        # 分段处理
        if seg["type"] == "initial":
            handle_initial_segment(params, COMMON_PARAMS, calc_result, results)
        elif seg["type"] == "attenuator":
            handle_attenuator_segment(params, results, seg_idx, Loss_attu, L)
        else:
            handle_normal_segment(params, results, seg_idx)

    # ========================= 结果处理与可视化 =========================
    process_and_visualize(results, C_list, COMMON_PARAMS, SEGMENTS)

def build_input_params(common_params, seg):
    """构建输入参数列表"""
    return [
        common_params["I"], common_params["V"], seg["Kc"], seg["Loss_perunit"],
        seg["p_SWS"], seg["len"], common_params["w"], common_params["t"],
        seg["Fn_K"], common_params["f0_GHz"], seg["Vpc"]
    ]

def handle_initial_segment(params, common_params, calc_result, results):
    """处理初始段"""
    P_in = 0.1 #输入功率Pin
    P_flux = params["C"] * common_params["I"] * common_params["V"] * 2
    params.update({
        "A0": np.sqrt(P_in / P_flux),
        "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"]
    })
    print(f"\n初始段参数:\n{params}")
    results.append(solveTWTNOLINE_INIT(**params))

def handle_attenuator_segment(params, results, seg_idx, Loss_attu, L):
    """处理衰减段"""
    prev = results[seg_idx-1]
    d_attu = 0.01836 * Loss_attu / (L / (2 * np.pi))
    print(f"\n衰减段损耗系数: {d_attu}")
    params.update({
        # "d": d_attu,
        "result_y_ends": prev["y"][-1],
        "result_A_ends": prev["A_Ends"]* 10**(-Loss_attu/20),
        "result_dA_dy": prev["dA_dy_Ends"]*0,
        "result_theta": prev["theta_Ends"],
        "result_dtheta_dy": prev["dtheta_dy_Ends"],
        "result_u_finnal": prev["u_final"],
        "result_phi_finnal": prev["phi_final"],
    })
    results.append(solveTWTNOLINE_Drift(**params))

def handle_normal_segment(params, results, seg_idx):
    """处理常规段"""
    prev = results[seg_idx-1]
    params.update(get_previous_results(prev))
    results.append(solveTWTNOLINE_OUTPUT(**params))

def get_previous_results(prev_result):
    """获取前一段结果"""
    return {
        "result_y_ends": prev_result["y"][-1],
        "result_A_ends": prev_result["A_Ends"],
        "result_dA_dy": prev_result["dA_dy_Ends"],
        "result_theta": prev_result["theta_Ends"],
        "result_dtheta_dy": prev_result["dtheta_dy_Ends"],
        "result_u_finnal": prev_result["u_final"],
        "result_phi_finnal": prev_result["phi_final"],
    }

def process_and_visualize(results, C_list, common_params, segments):
    """结果处理与可视化"""
    # 数据合成
    Y_Finall = np.concatenate([r["y"] for r in results])
    A_Fianll = np.concatenate([r["A"] for r in results])
    theta_Fianll = np.concatenate([r["theta"] for r in results])
    u_Finall = np.concatenate([r["u_now"] for r in results])
    
    # 功率计算
    P_Out = 2 * common_params["I"] * common_params["V"] * np.concatenate(
        [C_list[i] * (results[i]["A"]**2) for i in range(len(segments))]
    )
    
    # 性能指标
    P_max = P_Out.max()
    Eff_max = P_max / (common_params["I"] * common_params["V"]) * 100
    Lenth = Y_Finall / (2 * np.pi * np.mean(C_list))
    
    print("\n======== 最终计算结果 ========")
    print(f"非线性理论增益: {10 * np.log10(P_Out[-1]/0.1):.4f} dB")
    print(f"输出功率: {P_Out[-1]:.4f} W")
    print(f"最大效率: {Eff_max:.4f}%")
    print(f"最大功率: {P_max:.4f} W")

    # 可视化
    plot_results(Y_Finall, A_Fianll, theta_Fianll, u_Finall, Lenth, P_Out, results[-1])

def plot_results(Y, A, theta, u, Lenth, P_Out, final_seg):
    """可视化绘图"""
    plt.figure(figsize=(12, 8), dpi=100)
    
    # 振幅演化
    plt.subplot(2, 3, 1)
    plt.plot(Y, A, 'navy')
    plt.xlabel("Position y", fontsize=10)
    plt.ylabel("Amplitude A(y)", fontsize=10)
    plt.title("Amplitude Growth", fontsize=12)
    plt.grid(alpha=0.3)

    # 相位演化
    plt.subplot(2, 3, 2)
    plt.plot(Y, theta, 'maroon')
    plt.xlabel("Position y", fontsize=10)
    plt.ylabel("Phase Shift θ(y)", fontsize=10)
    plt.title("Phase Evolution", fontsize=12)
    plt.grid(alpha=0.3)

    # 速度分布
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(
        final_seg["phi_final"], final_seg["u_final"],
        c=final_seg["phi_final"], cmap='hsv', s=20, edgecolor='k', lw=0.5
    )
    plt.colorbar(scatter, label="Final Phase ϕ(y_end)")
    plt.xlabel("Final Phase ϕ(y_end)", fontsize=10)
    plt.ylabel("Final Velocity u(y_end)", fontsize=10)
    plt.title("Velocity Distribution", fontsize=12)
    plt.grid(alpha=0.3)

    # 相位分布
    plt.subplot(2, 3, 4)
    scatter = plt.scatter(
        final_seg["phi0_grid"], final_seg["phi_final"],
        c=final_seg["phi0_grid"], cmap='hsv', s=20, edgecolor='k', lw=0.5
    )
    plt.colorbar(scatter, label="Initial Phase")
    plt.xlabel("Initial Phase ϕ₀", fontsize=10)
    plt.ylabel("Final Phase ϕ(y_end)", fontsize=10)
    plt.title("Phase Distribution", fontsize=12)
    plt.grid(alpha=0.3)

    # 电子相空间
    plt.subplot(2, 3, 5)
    plt.plot(Lenth, u, 'navy')
    plt.xlabel("Position Z(Interaction Length)", fontsize=10)
    plt.ylabel("Electron Velocity (u)", fontsize=10)
    plt.title("Electron Phase Space", fontsize=12)
    plt.grid(alpha=0.3)

    # 功率演化
    plt.subplot(2, 3, 6)
    plt.plot(Lenth, P_Out, 'darkgreen')
    plt.xlabel("Position Z(Interaction Length)", fontsize=10)
    plt.ylabel("Output Power (W)", fontsize=10)
    plt.title("Power Evolution", fontsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()