import numpy as np
from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VSUPER_MIX_VFTSSM import solveTWTNOLINE_OUTPUT, solveTWTNOLINE_INIT, solveTWTNOLINE_Drift

class TWTCalculator:
    @staticmethod
    def build_input_params(common_params, seg):
        return [
            common_params["I"], common_params["V"], seg["Kc"], seg["Loss_perunit"], 
            seg["p_SWS"], seg["len"], common_params["w"], common_params["t"], 
            seg["Fn_K"], common_params["f0"], seg["Vpc"]
        ]

    @staticmethod
    def calculate(common_params, segments, loss_attu, log_callback=None):
        # 内部日志分发函数
        def log(msg):
            if log_callback:
                log_callback(msg)

        if not segments:
            raise ValueError("请至少输入一个分段参数")
        if common_params["P_in"] <= 0:
            raise ValueError("输入功率必须大于0")

        results = []
        C_list = []

        for seg_idx, seg in enumerate(segments):
            log(f"\n段{seg_idx}: " + ", ".join(f"{k}={v}" for k, v in seg.items()))
            
            input_params = TWTCalculator.build_input_params(common_params, seg)
            calc_result = simple_calculation(*input_params)
            
            log("计算参数:")
            for k, v in calc_result.items():
                log(f"  {k}: {v}")
                
            C = calc_result["小信号增益因子C"]
            L = 2 * np.pi * calc_result["互作用长度N"] * C
            C_list.append(C)
            
            params = {
                "C": C, "b": calc_result["非同步参量b"], "d": calc_result["损耗因子d"],
                "wp_w": calc_result["等离子体频率Wp"] / (2 * np.pi * common_params["f0"] * 1e9),
                "beta_space": calc_result["beta_Space"], "r_beam": calc_result["束流归一化尺寸r_beam"],
                "Fill_Rate": seg["Fn_K"], "p_SWS": seg["p_SWS"] * 1e-3,
                "m": 64, "Space_cut": 20,
                "y_end": L + (results[-1]["y"][-1] if seg_idx > 0 and results else 0)
            }
            
            if seg["type"] == "initial":
                P_in = common_params["P_in"]
                P_flux = params["C"] * common_params["I"] * common_params["V"] * 2
                params.update({
                    "A0": np.sqrt(P_in / P_flux),
                    "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"]
                })
                # 调用前打印完整参数
                log(f"\n初始段参数:\n{params}")
                log(f"使用输入功率: {P_in} W")
                # 调用求解函数
                res = solveTWTNOLINE_INIT(**params)
                results.append(res)
                
            elif seg["type"] == "attenuator":
                prev = results[-1]
                d_attu = 0.01836 * loss_attu / (L / (2 * np.pi))
                params.update({
                    "result_y_ends": prev["y"][-1],
                    "result_A_ends": prev["A_Ends"] * 10**(-loss_attu/20),
                    "result_dA_dy": prev["dA_dy_Ends"] * 0,
                    "result_theta": prev["theta_Ends"],
                    "result_dtheta_dy": prev["dtheta_dy_Ends"],
                    "result_u_finnal": prev["u_final"],
                    "result_phi_finnal": prev["phi_final"],
                })
                # 调用前打印损耗系数
                log(f"\n衰减段损耗系数: {d_attu}")
                # 调用求解函数
                res = solveTWTNOLINE_Drift(**params)
                results.append(res)
                
            else:
                prev = results[-1]
                params.update({
                    "result_y_ends": prev["y"][-1],
                    "result_A_ends": prev["A_Ends"],
                    "result_dA_dy": prev["dA_dy_Ends"],
                    "result_theta": prev["theta_Ends"],
                    "result_dtheta_dy": prev["dtheta_dy_Ends"],
                    "result_u_finnal": prev["u_final"],
                    "result_phi_finnal": prev["phi_final"],
                })
                # 调用求解函数
                res = solveTWTNOLINE_OUTPUT(**params)
                results.append(res)

        # 汇总计算数据
        Y_Finall = np.concatenate([r["y"] for r in results])
        A_Fianll = np.concatenate([r["A"] for r in results])
        theta_Fianll = np.concatenate([r["theta"] for r in results])
        u_now = np.concatenate([r["u_now"] for r in results])
        phi_now = np.concatenate([r["phi_now"] for r in results])
        P_Out = 2 * common_params["I"] * common_params["V"] * np.concatenate([C_list[i] * (results[i]["A"]**2) for i in range(len(segments))])
        
        P_max = P_Out.max()
        Eff_max = P_max / (common_params["I"] * common_params["V"]) * 100
        Lenth = Y_Finall / (2 * np.pi * np.mean(C_list))

        final_res = {
            "Y": Y_Finall, "A": A_Fianll, "theta": theta_Fianll, "u": u_now, "phi": phi_now,
            "Lenth": Lenth, "P_Out": P_Out, "final_seg": results[-1], "C_list": C_list,
            "Gain_dB": 10 * np.log10(P_Out[-1]/common_params["P_in"]),
            "P_out_end": P_Out[-1], "Eff_max": Eff_max, "P_max": P_max
        }
        return final_res
