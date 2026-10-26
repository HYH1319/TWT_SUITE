import numpy as np
from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VSUPER_MIX_VFTSSM import (
    solveTWTNOLINE_OUTPUT,
    solveTWTNOLINE_INIT,
    solveTWTNOLINE_Drift,
)

PRECISION_PRESETS = {
    "标准精度": {"Space_cut": 20, "m": 64},
    "高精度": {"Space_cut": 50, "m": 64},
    "极高精度": {"Space_cut": 100, "m": 64},
}


class TWTCalculator:
    @staticmethod
    def build_input_params(common_params, seg):
        return [
            common_params["I"],
            common_params["V"],
            seg["Kc"],
            seg["Loss_perunit"],
            seg["p_SWS"],
            seg["len"],
            common_params["w"],
            common_params["t"],
            seg["Fn_K"],
            common_params["f0"],
            seg["Vpc"],
        ]

    @staticmethod
    def _propagate_state(params, prev, attenuate_db=None):
        """
        将上一段末态写入 params。

        attenuate_db=None → 常规段 (振幅不衰减、导数保持)
        attenuate_db=数值 → 衰减段 (振幅按 dB 衰减、导数清零)
        """
        if attenuate_db is None:
            factor = 1.0
            da_dy = prev["dA_dy_Ends"]
        else:
            factor = 10 ** (-attenuate_db / 20)
            da_dy = 0.0

        params.update(
            {
                "result_y_ends": prev["y"][-1],
                "result_A_ends": prev["A_Ends"] * factor,
                "result_dA_dy": da_dy,
                "result_theta": prev["theta_Ends"],
                "result_dtheta_dy": prev["dtheta_dy_Ends"],
                "result_u_finnal": prev["u_final"],
                "result_phi_finnal": prev["phi_final"],
            }
        )

    @staticmethod
    def calculate(
        common_params, segments, loss_attu, log_callback=None, precision="标准精度"
    ):
        # ── 1. 校验 (原有 + 显式化) ──
        if precision not in PRECISION_PRESETS:
            raise ValueError(
                f"未知精度等级 '{precision}'，可选: {list(PRECISION_PRESETS.keys())}"
            )
        if not segments:
            raise ValueError("请至少输入一个分段参数")
        if common_params["P_in"] <= 0:
            raise ValueError("输入功率必须大于0")
        if segments[0].get("type") != "initial":
            raise ValueError("第一段必须为 'initial' 类型")

        _prec = PRECISION_PRESETS[precision]
        log = log_callback or (lambda msg: None)
        I, V, P_in = common_params["I"], common_params["V"], common_params["P_in"]
        IV = I * V

        results = []
        C_list = []

        # ── 2. 逐段计算 ──
        for seg_idx, seg in enumerate(segments):
            log(f"\n段{seg_idx}: " + ", ".join(f"{k}={v}" for k, v in seg.items()))

            calc_result = simple_calculation(
                *TWTCalculator.build_input_params(common_params, seg)
            )
            log("计算参数:")
            for k, v in calc_result.items():
                log(f"  {k}: {v}")

            C = calc_result["小信号增益因子C"]
            L = 2 * np.pi * calc_result["互作用长度N"] * C
            C_list.append(C)

            y_base = results[-1]["y"][-1] if seg_idx > 0 else 0.0
            params = {
                "C": C,
                "b": calc_result["非同步参量b"],
                "d": calc_result["损耗因子d"],
                "wp_w": calc_result["等离子体频率Wp"]
                / (2 * np.pi * common_params["f0"] * 1e9),
                "beta_space": calc_result["beta_Space"],
                "r_beam": calc_result["束流归一化尺寸r_beam"],
                "Fill_Rate": seg["Fn_K"],
                "m": _prec["m"],
                "Space_cut": _prec["Space_cut"],
                "y_end": L + y_base,
            }

            seg_type = seg.get("type", "initial")

            if seg_type == "initial":
                params["A0"] = np.sqrt(P_in / (C * IV * 2))
                log(f"\n初始段参数:\n{params}")
                log(f"使用输入功率: {P_in} W")
                res = solveTWTNOLINE_INIT(**params)

            elif seg_type == "attenuator":
                TWTCalculator._propagate_state(
                    params, results[-1], attenuate_db=loss_attu
                )
                d_attu = 0.01836 * loss_attu / (L / (2 * np.pi))
                log(f"\n衰减段损耗系数: {d_attu}")
                res = solveTWTNOLINE_Drift(**params)

            else:  # "O" 或其它契约值
                TWTCalculator._propagate_state(params, results[-1])
                res = solveTWTNOLINE_OUTPUT(**params)

            results.append(res)

        # ── 3. 汇总 ──
        Y_Finall = np.concatenate([r["y"] for r in results])
        A_Fianll = np.concatenate([r["A"] for r in results])
        theta_Fianll = np.concatenate([r["theta"] for r in results])
        u_now = np.concatenate([r["u_now"] for r in results])
        phi_now = np.concatenate([r["phi_now"] for r in results])
        P_Out = (
            2
            * IV
            * np.concatenate([c * (r["A"] ** 2) for c, r in zip(C_list, results)])
        )

        P_max = P_Out.max()
        Eff_max = P_max / IV * 100
        seg_lengths = [s["p_SWS"] * s["len"] for s in segments]
        z_edges = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        Lenth = np.concatenate(
            [
                np.linspace(z_edges[i], z_edges[i + 1], len(results[i]["y"]))
                for i in range(len(results))
            ]
        )

        return {
            "Y": Y_Finall,
            "A": A_Fianll,
            "theta": theta_Fianll,
            "u": u_now,
            "phi": phi_now,
            "Lenth": Lenth,
            "P_Out": P_Out,
            "final_seg": results[-1],
            "C_list": C_list,
            "Gain_dB": 10 * np.log10(P_Out[-1] / P_in),
            "P_out_end": P_Out[-1],
            "Eff_max": Eff_max,
            "P_max": P_max,
        }
