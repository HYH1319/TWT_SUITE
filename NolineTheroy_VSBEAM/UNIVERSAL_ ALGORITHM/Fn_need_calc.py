import numpy as np
import math as m

w = float(input("请输入束流宽度(mm):")) * 1e-3
t = float(input("请输入束流厚度(mm):")) * 1e-3
f0_Hz = float(input("请输入工作频率(GHz):")) * 1e9
Ue = float(input("请输入束流电压:"))


def Vpc_calc(U):
    gamma = U / 5.11e5
    Vp_c = 1 * m.pow(1 - 1 / m.pow(1 + gamma, 2), 0.5)

    return Vp_c


Vec = Vpc_calc(Ue)
omega = 2 * m.pi * f0_Hz
beta_e = omega / (Vec * 299792458)

K_wave = omega / 299792458
gamma0 = np.sqrt(beta_e**2 - K_wave**2)
print(f"请根据此参数查表得到正确的填充校正因子:{gamma0*t}\n")

Fill_Rate_Ture = float(input("请输入正确的填充校正因子:"))

r_beam = np.sqrt((w * t / 4))  # 束流归一化尺寸

Fn_tmp = np.sqrt((np.pi / w) ** 2 + (np.pi / t) ** 2)
Fn_FullFill = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
Fn_tureFix = Fn_FullFill * Fill_Rate_Ture  # 填充矫正Fn
print(f"填充校正后绝对正确的Fn_TureFIX={Fn_tureFix}\n")

# 带状束流Rowe特征值R介于[Fn_tureFix,np.sqrt(Fn_tureFix)]之间,但没有解析方法，近似数值计算
Fn_needFix = (Fn_tureFix * (Fn_tureFix**0.5)) ** 0.5
Rowe_Sheetbeam = Fn_needFix  # 用一个修正的Fn_needFix替代Rowe_Sheetbeam

print(f"大信号分析需要的Rowe特征值Rowe_Sheetbeam={Rowe_Sheetbeam}\n")

Fill_Rate_need = Fn_needFix / Fn_FullFill
print(f"大信号分析需要填充因子Fill_Rate_need={Fill_Rate_need}\n")
