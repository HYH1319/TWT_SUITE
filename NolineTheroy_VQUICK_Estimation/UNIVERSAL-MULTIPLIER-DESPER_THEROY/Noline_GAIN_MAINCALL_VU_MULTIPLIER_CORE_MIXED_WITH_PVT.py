import numpy as np
from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VU import (
    solveTWTNOLINE_OUTPUT,
    solveTWTNOLINE_INIT,
    solveTWTNOLINE_Drift
)

def calculate_SEGMENT_TWT_NOLINE_SHEETBEAM(
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
    P_in,
    Loss_attu,
    harmonic_start_idx=None,
    Vpc_adjust_coef=0.82,  # Vpc调整系数，默认0.82
    Kc_adjust_coef=1.6     # Kc调整系数，默认1.6
):
    """主函数：计算TWT非线性模型的分段输出功率，支持p_SWS变化的参数调整"""
    
    # ===================== 内部辅助函数定义 =====================
    
    def expand_scalar_params(params_dict, segment_count):
        """将单值参数扩展为与段数相同的列表"""
        expanded = {}
        for name, value in params_dict.items():
            if isinstance(value, (int, float)):
                expanded[name] = [value] * segment_count
            else:
                expanded[name] = value
        return expanded
    
    def handle_harmonic_parameters(param_list, harmonic_index, segment_count):
        """
        处理倍频段参数 - 将[前段值, 倍频段值]的列表扩展为完整段数列表
        """
        if isinstance(param_list, list) and len(param_list) == 2 and harmonic_index is not None:
            if harmonic_index < 0 or harmonic_index >= segment_count:
                raise ValueError(f"倍频起始索引 {harmonic_index} 必须在 0 到 {segment_count-1} 之间")
            # 返回完整列表：前段值重复harmonic_index次 + 倍频段值重复剩余次
            return [param_list[0]] * harmonic_index + [param_list[1]] * (segment_count - harmonic_index)
        return param_list
    
    def build_segment_input(common_params, seg):
        """构建单个段的计算输入参数"""
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
            common_params["f0_GHz"],
            seg["Vpc"]
        ]
    
    def initialize_first_segment(params, common, calc_result, results, input_power):
        """初始化并计算第一段"""
        flux = params["C"] * common["I"] * common["V"] * 2
        params.update({
            "A0": np.sqrt(input_power / flux),
            "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"],
        })
        results.append(solveTWTNOLINE_INIT(**params))
    
    def process_attenuator_segment(params, results, seg_idx, attenuation):
        """处理有衰减的段"""
        prev = results[seg_idx - 1]
        params.update({
            "result_y_ends": prev["y"][-1],
            "result_A_ends": prev["A_Ends"] * 10 ** (-attenuation / 20),
            "result_dA_dy": prev["dA_dy_Ends"] * 0,
            "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"],
            "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_Drift(**params))
    
    def process_normal_segment(params, results, seg_idx):
        """处理常规段（无衰减）"""
        prev = results[seg_idx - 1]
        params.update({
            "result_y_ends": prev["y"][-1],
            "result_A_ends": prev["A_Ends"],
            "result_dA_dy": prev["dA_dy_Ends"],
            "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"],
            "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_OUTPUT(**params))
    
    def calculate_final_power(results, c_list, common_params, segment_count):
        """计算并返回最终功率输出"""
        return 2 * common_params["I"] * common_params["V"] * np.concatenate(
            [c_list[i] * (results[i]["A"] ** 2) for i in range(segment_count)]
        )
    
    # ===================== 内置参数调整函数 =====================
    def para_func(p_SWS, idx, Vpc_base, Kc_base, harmonic_start_idx):
        """内置参数调整函数：根据当前段与基准段的差异调整参数"""
        # 确定基准段索引
        if harmonic_start_idx is not None and idx >= harmonic_start_idx:
            # 倍频段：以倍频段的第一段为基准
            base_idx = harmonic_start_idx
        else:
            # 倍频前：以第一段为基准
            base_idx = 0
        
        base_p = p_SWS[base_idx]  # 基准段的p_SWS值
        current_p = p_SWS[idx]    # 当前段的p_SWS值
        delta_p = (current_p - base_p) / base_p
        
        # 应用用户提供的调整系数
        Vpc_adjusted = Vpc_base + Vpc_adjust_coef * delta_p * Vpc_base
        Kc_adjusted = Kc_base + Kc_adjust_coef * delta_p * Kc_base
        
        return {"Vpc": Vpc_adjusted, "Kc": Kc_adjusted}
    
    # ===================== 主计算逻辑 =====================
    
    # 1. 参数准备与校验
    num_segments = len(N_unit)
    
    # 关键修复：首先处理倍频参数，将它们扩展为完整长度
    if harmonic_start_idx is not None:
        Kc = handle_harmonic_parameters(Kc, harmonic_start_idx, num_segments)
        Loss_perunit = handle_harmonic_parameters(Loss_perunit, harmonic_start_idx, num_segments)
        p_SWS = handle_harmonic_parameters(p_SWS, harmonic_start_idx, num_segments)
        Fn_K = handle_harmonic_parameters(Fn_K, harmonic_start_idx, num_segments)
        Vpc = handle_harmonic_parameters(Vpc, harmonic_start_idx, num_segments)
    
    # 扩展所有参数为列表
    expansion_params = expand_scalar_params({
        "p_SWS": p_SWS,
        "Kc": Kc,
        "Vpc": Vpc,
        "Loss_perunit": Loss_perunit,
        "Fn_K": Fn_K
    }, num_segments)
    
    # 2. 创建分段结构定义（应用参数调整函数）
    segments = []
    for i in range(num_segments):
        # 应用para_func调整参数，传入harmonic_start_idx
        adjusted_params = para_func(
            expansion_params["p_SWS"],  # p_SWS全列表
            i,  # 当前段索引
            expansion_params["Vpc"][i],
            expansion_params["Kc"][i],
            harmonic_start_idx  # 倍频起始索引
        )
        
        segments.append({
            "len": N_unit[i],
            "Vpc": adjusted_params["Vpc"],     # 使用调整后的参数
            "Kc": adjusted_params["Kc"],       # 使用调整后的参数
            "p_SWS": expansion_params["p_SWS"][i],
            "Loss_perunit": expansion_params["Loss_perunit"][i],
            "Fn_K": expansion_params["Fn_K"][i]
        })
    
    # 3. 公共参数配置
    common_params = {
        "I": I,
        "V": V,
        "w": w,
        "t": t,
        "f0_GHz": f0_GHz,
    }
    
    # 4. 分段计算主循环
    results = []  # 存储每段计算结果
    c_values = [] # 存储每段C值
    
    for seg_idx, segment in enumerate(segments):
        # 4.1 准备当前段输入参数
        input_params = build_segment_input(common_params, segment)
        print(input_params)
        calc_result = simple_calculation(*input_params)
        
        # 4.2 计算当前段公共参数
        C = calc_result["小信号增益因子C"]
        interaction_length = 2 * np.pi * calc_result["互作用长度N"] * C
        c_values.append(C)

        if (w == t):  # 行波管等离子体频率降低系数R^2,ShitBeam用Fn,CicularBeam用Rowe特征值R^2
            R = calc_result["Rowe特征值R"]
            Rn_sqr_values=R**2
        else:
            Rn_sqr_values = calc_result["等离子体频率降低因子Fn"]**2

        wp_omega = calc_result["等离子体频率Wp"] / (2 * np.pi * input_params[9] * 1e9)
        Space_cut = 10  # 空间电荷场截断项数

        beam_params = [Rn_sqr_values, wp_omega, Space_cut]
        
        # 4.3 准备非线性求解器参数
        solver_params = {
            "C": C,
            "b": calc_result["非同步参量b"],
            "d": calc_result["损耗因子d"],  
            "beam_params": beam_params,  # beam_params
            "m": 50,          # 求解器精度参数
            "y_end": interaction_length + (results[-1]["y"][-1] if seg_idx > 0 else 0),
        }
        print(solver_params)
        # 4.4 根据段类型处理
        if seg_idx == 0:  # 第一段
            initialize_first_segment(solver_params, common_params, calc_result, results, P_in)
        elif seg_idx in SectionedSEGMENT_IDX:  # 衰减段
            process_attenuator_segment(solver_params, results, seg_idx, Loss_attu)
        else:  # 常规段
            process_normal_segment(solver_params, results, seg_idx)
    
    # 5. 计算最终输出功率
    final_power = calculate_final_power(results, c_values, common_params, num_segments)
    
    return {
        "输出功率P_out": final_power[-1],
        "各段参数": segments,  # 返回调整后的各段参数供调试
        "c_values": c_values
    }

if __name__ == "__main__":
    # 输入参数
    input_params = [
        0.30,     # I [A],公共参数
        23000,    # V [V],公共参数
        [3.6, 3.0],  # 耦合系数 (Kc)，倍频前和倍频段两组值
        [0, 0.1],   # 单位损耗，倍频前和倍频段两组值
        [1,3],      # 衰减段索引列表
        [0.50, 0.495, 0.15, 0.145, 0.140],  # 慢波结构周期
        [30, 5, 5, 5, 5],  # 各段长度（周期数）
        0.45,     # 带状注宽度 w [mm],公共参数
        0.10,     # 带状注厚度 t [mm],公共参数
        [1.2, 1], # 填充因子，倍频前和倍频段两组值
        211,      # 中心频率 f0 [GHz],公共参数
        [0.290, 0.287],  # 相位速度，倍频前和倍频段两组值
        0.1,      # 输入功率 [W],公共参数
        15,       # 衰减器损耗 [dB],公共参数
    ]

    # 倍频段起始索引（第3段开始倍频）
    harmonic_start_idx = 2
    
    # 自定义调整系数
    Vpc_adjust_coef = 0.82  # Vpc调整系数
    Kc_adjust_coef = 1.6    # Kc调整系数
    
    # 执行计算
    result = calculate_SEGMENT_TWT_NOLINE_SHEETBEAM(
        *input_params, 
        harmonic_start_idx=harmonic_start_idx,
        Vpc_adjust_coef=Vpc_adjust_coef,
        Kc_adjust_coef=Kc_adjust_coef
    )
    
    # 输出结果
    print("\n最终输出功率:", result["输出功率P_out"], "W")
    print("\n调整后的各段参数:")
    for i, seg in enumerate(result["各段参数"]):
        harmonic_type = "倍频前" if i < harmonic_start_idx else "倍频段"
        print(f"段 {i+1} ({harmonic_type}): Kc={seg['Kc']:.4f} Ω, Vpc={seg['Vpc']:.4f}c")
    print("\n各段C值:", result["c_values"])