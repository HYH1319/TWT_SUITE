import tkinter as tk
from tkinter import messagebox
import math as m
import numpy as np
import sys

from scipy.special import jv  # 贝塞尔函数和双曲余切
from scipy.optimize import root_scalar  # 数值求根
from TWT_CORE_SIMP import simple_calculation 


def calculate():
    try:

        I = float(entry_i.get())
        V = float(entry_v.get())
        Kc = float(entry_kc.get())
        Loss_perunit = float(entry_loss.get())
        p_SWS = float(entry_p_SWS.get())
        N_unit = int(entry_n_unit.get())
        w = float(entry_w.get())
        t = float(entry_t.get())
        f0_GHz = float(entry_f0.get())
        Fn_K = float(entry_Fn_K.get())
        Vpc = float(entry_Vpc.get())


     # ========== 核心计算逻辑（与图片严格一致） ==========
        Lineparam = simple_calculation(
            I, V, Kc, Loss_perunit, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc
        )
        # return {
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
        r_beam=Lineparam["束流归一化尺寸r_beam"]
        beta_e=Lineparam["beta_Space"]

        K_wave = 2*np.pi*f0_GHz*1e9 / 299792458
        gamma0 = np.sqrt(beta_e**2 - K_wave**2)
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
        print(roots)
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
        # A=-9.54
        Gmax = A + 54.6 * x1 * C * N
        if Gmax < 0:
            Gmax = 0
        else:
            Gmax = Gmax

        # 计算Tao = (Gmax - Loss)/2 （图片公式6）
        Loss=Loss_perunit*N_unit
        Tao = (Gmax - Loss) / 2

        # ========== 严格按图片格式输出 ==========
        result_text = (
            f"以下是计算结果："
            f"增益因子 C = {C:.3f}\nN = {N:.3f}\n"
            f"L = Loss/N={Loss:.1f}/{N:.3f} = {L:.3f}\n"
            f"损耗因子 d = 0.01836*L/C = 0.01836*{L:.3f}/{C:.3f} = {d:.2f}\n"
            f"圆形束流等离子体频率降低因子计算参量 Gamma*r_beam ={gamma0*r_beam:.3e} \n"
            f"超宽带状束流等离子体频率降低因子计算参量 Gamma*t ={gamma0*t:.3e} \n"
            f"圆形束满填充等离子体频率降低因子 Fn ={Fn:.3e} \n"
            f"等离子体频率 Wp = {Wp:.3e} rad/s\n"
            f"空间电荷参量 4QC = {Wq_over_omegaC_sq:.3e}\n"  # 显示实际使用的QC值
            f"非同步参量 b={b:.3f}\n"
            f"归一化电子速度 Vec={Vec:.3f}\n"
            f"x₁ = {x1:.3f}\n"
            f"Gmax要来了：\n"
            f"Gmax={A:.3f} + 54.6*{x1:.3f}* {C:.3f} * {N:.3f}\n= {Gmax:.3f} dB\n"
            f"Required Return Loss = {Tao:.3f} dB\n"
        )
        lbl_result.config(text=result_text)

    except ValueError as e:
        messagebox.showerror("输入错误", str(e))
    except ZeroDivisionError:
        messagebox.showerror("计算错误", "除零错误（请检查输入参数）")
    return Gmax


# ==============================================
# ================== GUI界面（新增QC输入框） ==================
root = tk.Tk()
root.title("行波管线性增益计算器-SUPERPLUS版")
root.geometry("500x650")

# 输入参数框架（新增QC输入行）
input_frame = tk.LabelFrame(root, text="输入参数（*为频域变量）", padx=10, pady=10)
input_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

input_labels = [
    ("束流电流 I (A):", "0.30"),
    ("束流电压 U (V):", "23000"),
    ("Kc*:", "3.6"),
    ("每单元损耗Loss_perunit*:", "0"),
    ("周期长度p_SWS (mm):", "0.500"),
    ("周期个数N_unit (正整数):", "25"),
    ("宽度 w (mm):", "0.2"),
    ("厚度 t (mm):", "0.2"),
    ("频率 f0 (GHz)*:", "211"),
    ("Fn_K 参数:\n w=t表示圆形束流\n 填充率倒数必须大于1\n w!=t,仅为带状束流矫正\n", "1"),
    ("Vpc (c)*:", "0.288"),  # 新增Vpc输入行
]

for row, (label_text, default_val) in enumerate(input_labels):
    tk.Label(input_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
    entry = tk.Entry(input_frame)
    entry.insert(0, default_val)
    entry.grid(row=row, column=1, padx=5)

# 获取输入框引用（新增QC）
entries = [
    child for child in input_frame.children.values() if isinstance(child, tk.Entry)
]
(
    entry_i,
    entry_v,
    entry_kc,
    entry_loss,
    entry_p_SWS,
    entry_n_unit,
    entry_w,
    entry_t,
    entry_f0,
    entry_Fn_K,
    entry_Vpc,
) = entries

# 其他组件保持不变...
btn_calculate = tk.Button(root, text="计算", command=calculate)
btn_calculate.pack(pady=10)

lbl_result = tk.Label(root, text="", font=("Courier New", 10), justify=tk.LEFT)
lbl_result.pack()



root.mainloop()
