import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys

from TWT_CORE_SIMP import simple_calculation
from TWT_CORE_SIMP import detailed_calculation
import math as m
import matplotlib.pyplot as plt
import numpy as np
from _TWT_CORE_NOLINE_COMPLEX_VSUPER_MIX import solveTWTNOLINE_INIT
from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER import Noline_CORE_CALC_SUPER as Noline_CORE_CALC


if __name__ == "__main__":
    # 计算物理参数
    inputP = [
        0.03,  # I: Any
        23000,  # V: Any
        10,  # Kc: Any
        0.10,  # Loss_perunit: Any
        0.317,  # p_SWS: Any
        40,  # N_unit: Any
        0.05,  # w: Any
        0.05,  # t: Any
        1.35,  # Fill_Rate: Any
        344,  # f0_GHz: Any
        0.290,  # Vpc: Any
    ]  ##(function) def detailed_calculation(I: Any,V: Any,Kc: Any,Loss_perunit: Any,p_SWS: Any,N_unit: Any,w: Any,t: Any,Fill_Rate: Any,f0_GHz: Any,Vpc: Any# )
    initparamater = simple_calculation(*inputP)
    # return {
    #     "小信号增益因子C": C,
    #     "互作用长度N": N,
    #     "慢波线上损耗L": L,
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
    print(initparamater)

    C = initparamater["小信号增益因子C"]
    # 行波管增益参数C
    b = initparamater["非同步参量b"]
    # 行波管非同步参量b
    d = initparamater["损耗因子d"]
    # 损耗系数d
    wp_omega = initparamater["等离子体频率Wp"] / (2 * np.pi * inputP[9] * 1e9)
    # 相对等离子体频率
    beta_Space = initparamater["beta_Space"]
    r_beam = initparamater["束流归一化尺寸r_beam"]
    Fill_Rate = inputP[8]
    p_SWS = inputP[4]*1e-3
    L = 2 * np.pi * initparamater["互作用长度N"] * C

    P_in = C*inputP[0]*inputP[1]*1e-3*10
    print(f"最大输入功率是{P_in}")
    P_flux = C * inputP[0] * inputP[1] * 2
    A_0 = np.sqrt(P_in / P_flux)

    # ================================
    # 参数配置区（根据物理系统调整）
    # ================================
    # 参数设置 (根据实际需求修改)

    params = {
        "C": C,  # 增益参量
        "b": b,  # 速度非同步参量
        "d": d,  # 线路损耗参量
        "wp_w": wp_omega,  # 相对等离子体频率
        "beta_space": beta_Space,  # 电子波数beta_space
        "r_beam": r_beam,
        "Fill_Rate": Fill_Rate,
        "p_SWS": p_SWS,
        "m": 50,  # 电子离散数量
        "A0": A_0,  # 初始振幅
        "y_end": L,  # 计算终点
        "Space_cut": 10,  # 空间电荷波展开级数
    }
    print(params)

    # 调用求解器
    result = solveTWTNOLINE_INIT(
        **params
    )  #     (function) def solveTWTNOLINEM(C: Any,b: Any,d: Any,wp_w: Any,beta_e: Any,r_beam: Any,m: Any,A0: Any,y_end: Any,N_steps: int = 1000# ) -> dict[str, Any]

    # ========================= 结果后处理 =========================
    P_Out = C * inputP[0] * inputP[1] * 2 * (result["A"]) ** 2
    Eff = 2 * C * max(result["A"]) ** 2 * 100
    P_max = C * inputP[0] * inputP[1] * 2 * (max(result["A"])) ** 2

    resultLineG = detailed_calculation(*inputP)
    print(
        "The Gmax in Noline Theroy is %.4f dB "
        % (20 * np.log10((result["A_Ends"] / params["A0"])))
    )
    print("The Gain in Line Theroy is %.3f dB" % (resultLineG["Gmax"]))
    print(
        "The P_out in Noline Theroy is %.4f,The maximum Efficence is %.4f in percent,The maximum POWER is %.4f in Watt"
        % (P_Out[-1], Eff, P_max)
    )
    print(
        f"The A in the end is {result['A'][-1]}, The u in the end is {np.mean(abs(result['u_final']))}"
    )

    # ========================= 结果可视化 =========================
    plt.figure(figsize=(12, 8), dpi=100)

    # 振幅演化
    plt.subplot(2, 3, 1)
    plt.plot(result["y"], result["A"], color="navy")
    plt.xlabel("Position y", fontsize=10)
    plt.ylabel("Amplitude A(y)", fontsize=10)
    plt.title("Amplitude Growth", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 相位演化
    plt.subplot(2, 3, 2)
    plt.plot(result["y"], result["theta"], color="maroon")
    plt.xlabel("Position y", fontsize=10)
    plt.ylabel("Phase Shift θ(y)", fontsize=10)
    plt.title("Phase Evolution", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 最终速度分布

    plt.subplot(2, 3, 3)
    plt.scatter(
        result["phi_final"],
        result["u_final"],
        c=result["phi_final"],
        cmap="hsv",
        s=20,
        edgecolor="k",
        lw=0.5,
    )
    plt.colorbar(label="Final Phase ϕ(y_end)")
    plt.xlabel("Final Phase ϕ(y_end)", fontsize=10)
    plt.ylabel("Final Velocity u(y_end)", fontsize=10)
    plt.title("Velocity Distribution", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 最终相位分布
    plt.subplot(2, 3, 4)
    final_phi = result["phi_final"]
    plt.scatter(
        result["phi0_grid"],
        final_phi,
        c=result["phi0_grid"],
        cmap="hsv",
        s=20,
        edgecolor="k",
        lw=0.5,
    )
    plt.colorbar(label="Initial Phase")
    plt.xlabel("Initial Phase ϕ₀", fontsize=10)
    plt.ylabel("Final Phase ϕ(y_end)", fontsize=10)
    plt.title("Phase Distribution", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 电子相空间图
    Lenth = result["y"] / (2 * np.pi * C)
    plt.subplot(2, 3, 5)
    plt.plot(Lenth, result["u_now"], color="navy")
    plt.xlabel("Position Z(Interaction Lenth)", fontsize=10)
    plt.ylabel("Velocity of eletron (Z)", fontsize=10)
    plt.title("Velocity Distribution (Z))", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 轴向功率图
    plt.subplot(2, 3, 6)
    plt.plot(Lenth, P_Out, color="navy")
    plt.xlabel("Position Z(Interaction Lenth)", fontsize=10)
    plt.ylabel("Output Power Pout(Z)", fontsize=10)
    plt.title("Power Growth", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ========================= 参数扫描 =========================


def para_SWP(isparaSWP, isSAVERESULT, inputP):
    if isparaSWP == 1:
        import itertools
        import json
        import numpy as np

        # 原始输入参数（深拷贝避免污染原始数据）
        original_input = inputP.copy()

        # 配置扫描参数（按需求修改范围和步长）
        scan_params = {
            1: [23000,23500,24000],  # 扫描电压V
            8: [1],  # 扫描填充效率
        }

        # 生成所有参数组合
        param_indices = sorted(scan_params.keys())
        param_combinations = itertools.product(
            *(scan_params[idx] for idx in param_indices)
        )

        # 存储结果
        results = []
        for combo in param_combinations:
            current_input = original_input.copy()
            for i, idx in enumerate(param_indices):
                current_input[idx] = combo[i]

            try:
                # 执行计算
                twt_result = Noline_CORE_CALC(*current_input)
                results.append(
                    {
                        "parameters": {
                            f"param_{idx}": val
                            for idx, val in zip(param_indices, combo)
                        },
                        "result": {
                            "Pmax": twt_result["Pmax"],
                            "P_out": twt_result["P_out"],
                            "Maxiumum_eff": twt_result["Maxiumum_eff"],
                        },
                    }
                )
                print(f"参数组合 {combo} 计算完成")  # 仅打印当前组合进度
            except Exception as e:
                print(f"参数组合 {combo} 计算失败: {str(e)}")
                continue  # 跳过当前组合继续执行
        print(results)

        # 全部计算完成后统一保存
        if isSAVERESULT == 1:
            with open("param_scan_results.json", "w") as f:
                json.dump(results, f, indent=4)
            print("参数扫描完成！结果已保存至 param_scan_results.json")


para_SWP(0, 0, inputP)
