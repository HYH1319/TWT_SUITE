
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd  # 添加pandas库
from datetime import datetime  # 添加日期时间库
import os

from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VCBEAM_MIX import solveTWTNOLINE_OUTPUT, solveTWTNOLINE_INIT, solveTWTNOLINE_Drift

class TWT_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("行波管计算器")
        self.root.geometry("1200x800")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧参数输入区域
        self.input_frame = ttk.LabelFrame(self.main_frame, text="参数设置")
        self.input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.LEFT)
        
        # 右侧结果区域
        self.result_frame = ttk.LabelFrame(self.main_frame, text="计算结果")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.RIGHT)
        
        # 创建输入控件
        self.create_input_controls()
        
        # 创建计算结果展示区域
        self.create_result_display()
        
        # 加载默认参数
        self.default_segments = [
            {"len": 50, "Vpc": 0.2893, "p_SWS": 0.50, "Kc": 3.88, "f0_GHz": 211, "Loss_perunit": 0, "Fn_K": 1, "type": "initial"},
        ]
        
        self.load_defaults()
    
    def create_input_controls(self):
        """创建参数输入控件"""
        # 全局参数
        global_frame = ttk.LabelFrame(self.input_frame, text="全局参数")
        global_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 电流 I
        ttk.Label(global_frame, text="电流 I (A):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.I_var = tk.DoubleVar(value=0.3)
        ttk.Entry(global_frame, textvariable=self.I_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # 电压 V
        ttk.Label(global_frame, text="电压 V (V):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.V_var = tk.DoubleVar(value=23000)
        ttk.Entry(global_frame, textvariable=self.V_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # 输入功率 P_in - 新添加的全局参数
        ttk.Label(global_frame, text="输入功率 P_in (W):").grid(row=0, column=4, padx=5, pady=2, sticky=tk.W)
        self.P_in_var = tk.DoubleVar(value=0.10)  # 默认值0.004W
        ttk.Entry(global_frame, textvariable=self.P_in_var, width=10).grid(row=0, column=5, padx=5, pady=2)
        
        # 宽度 w
        ttk.Label(global_frame, text="宽度 w (mm):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.w_var = tk.DoubleVar(value=0.2)
        ttk.Entry(global_frame, textvariable=self.w_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # 厚度 t
        ttk.Label(global_frame, text="厚度 t (mm):").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.t_var = tk.DoubleVar(value=0.2)
        ttk.Entry(global_frame, textvariable=self.t_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        
        # Loss_attu 参数
        ttk.Label(global_frame, text="Loss_attu:").grid(row=1, column=4, padx=5, pady=2, sticky=tk.W)
        self.loss_attu_var = tk.DoubleVar(value=0)
        ttk.Entry(global_frame, textvariable=self.loss_attu_var, width=10).grid(row=1, column=5, padx=5, pady=2)

        
        # 频率 f0_GHz - 新增全局参数
        ttk.Label(global_frame, text="频率 f0 (GHz):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.f0_GHz_var = tk.DoubleVar(value=211)  # 默认值211 GHz
        ttk.Entry(global_frame, textvariable=self.f0_GHz_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # 分段参数表头
        segment_header = ttk.LabelFrame(self.input_frame, text="分段参数")
        segment_header.pack(fill=tk.X, padx=5, pady=5)
        
        columns = ("len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K", "type")
        headers = ("周期数(个)", "Vpc", "螺距(mm)", "耦合阻抗", "每单元损耗", "填充因子", "类型")
        
        for col, header in enumerate(headers):
            ttk.Label(segment_header, text=header, width=10).grid(row=0, column=col, padx=2, pady=2)
        
        # 分段参数输入
        self.segment_entries = []
        entry_frames = []
        
        for i in range(10):  # 最多6个分段
            entry_frame = ttk.Frame(self.input_frame)
            entry_frame.pack(fill=tk.X, padx=5, pady=2)
            entry_frames.append(entry_frame)
            row_entries = []
            
            for j, param in enumerate(columns):
                var = tk.StringVar()
                entry = ttk.Entry(entry_frame, textvariable=var, width=10)
                entry.grid(row=0, column=j, padx=2, pady=2)
                row_entries.append(var)
            
            self.segment_entries.append(row_entries)
        
        # 类型选择列添加下拉菜单
        type_options = ["initial", "attenuator", "O"]
        for i, entry_frame in enumerate(entry_frames):
            var = self.segment_entries[i][6]  # 第6列是类型（原第7列）
            combo = ttk.Combobox(entry_frame, textvariable=var, values=type_options, width=8)
            combo.grid(row=0, column=6, padx=2, pady=2)
        
        # 控制按钮
        button_frame = ttk.Frame(self.input_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="加载默认值", command=self.load_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="计算", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="重置", command=self.reset).pack(side=tk.LEFT, padx=5)
    
    def create_result_display(self):
        """创建结果显示区域"""
        # 添加标签页
        notebook = ttk.Notebook(self.result_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 文本结果标签页
        text_frame = ttk.Frame(notebook)
        self.result_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)
        notebook.add(text_frame, text="计算结果")
        
        # 图形结果标签页
        plot_frame = ttk.Frame(notebook)
        self.fig = plt.figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建空图
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Result shown Here", 
                fontsize=12, ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.axis('off')
        self.canvas.draw()
        notebook.add(plot_frame, text="可视化结果")
    
    def load_defaults(self):
        """加载默认参数"""
        # 全局参数
        self.I_var.set(0.3)
        self.V_var.set(23000)
        self.P_in_var.set(0.10)  # P_in默认值
        self.w_var.set(0.2)
        self.t_var.set(0.2)
        self.loss_attu_var.set(0)
        
        # 分段参数 - 只加载第一个分段（初始段）
        seg_data = self.default_segments[0]
        
        self.segment_entries[0][0].set(seg_data["len"])
        self.segment_entries[0][1].set(seg_data["Vpc"])
        self.segment_entries[0][2].set(seg_data["p_SWS"])
        self.segment_entries[0][3].set(seg_data["Kc"])
        self.segment_entries[0][4].set(seg_data["Loss_perunit"])
        self.segment_entries[0][5].set(seg_data["Fn_K"])
        self.segment_entries[0][6].set(seg_data["type"])
        
        # 清空其他分段
        for i in range(1, len(self.segment_entries)):
            for entry in self.segment_entries[i]:
                entry.set("")
    
    def reset(self):
        """重置所有输入"""
        # 全局参数
        self.I_var.set("")
        self.V_var.set("")
        self.P_in_var.set("")  # 重置P_in
        self.w_var.set("")
        self.t_var.set("")
        self.loss_attu_var.set("")
        
        # 分段参数
        for row in self.segment_entries:
            for entry in row:
                entry.set("")
        
        # 清空结果
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        # 重置图形
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "计算结果将在此处显示", 
                fontsize=12, ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.axis('off')
        self.canvas.draw()
    
    def calculate(self):
        """执行计算"""
        try:
            # 获取全局参数
            COMMON_PARAMS = {
                "I": self.I_var.get(),
                "V": self.V_var.get(),
                "P_in": self.P_in_var.get(),  # 获取输入功率
                "w": self.w_var.get(),
                "t": self.t_var.get(),
                "f0_GHz": self.f0_GHz_var.get()  # 添加频率到全局参数
            }
            
            # 获取Loss_attu参数
            Loss_attu = self.loss_attu_var.get()
            
            # 获取分段参数
            SEGMENTS = []
            columns = ("len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K", "type")
            
            for row in self.segment_entries:
                # 检查是否有有效输入
                if any(entry.get() for entry in row):
                    seg = {}
                    for i, param in enumerate(columns):
                        # 尝试转换为数值，否则保留为字符串
                        value = row[i].get()
                        try:
                            value = float(value)
                        except ValueError:
                            # 类型列保留字符串
                            if param == "type":
                                value = str(value)
                            else:
                                value = 0.0  # 默认为0
                        seg[param] = value
                    
                    SEGMENTS.append(seg)
            
            if not SEGMENTS:
                raise ValueError("请至少输入一个分段参数")
            
            # 确保输入功率有效
            if COMMON_PARAMS["P_in"] <= 0:
                raise ValueError("输入功率必须大于0")
            
            # 计算结果显示开始
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "开始计算...\n")
            self.result_text.insert(tk.END, f"全局参数: {COMMON_PARAMS}\n")
            self.result_text.insert(tk.END, f"Loss_attu: {Loss_attu}\n")
            self.result_text.insert(tk.END, f"分段参数: {len(SEGMENTS)}段\n")
            self.result_text.see(tk.END)
            self.result_text.update()
            
            # ========================= 主计算逻辑 =========================
            results = []
            C_list = []

            for seg_idx, seg in enumerate(SEGMENTS):
                # 显示当前段计算信息
                seg_info = f"\n段{seg_idx}: " + ", ".join(f"{k}={v}" for k, v in seg.items())
                self.result_text.insert(tk.END, seg_info + "\n")
                self.result_text.see(tk.END)
                self.result_text.update()
                
                # 参数计算
                input_params = self.build_input_params(COMMON_PARAMS, seg)
                calc_result = simple_calculation(*input_params)
                
                # 在文本区域显示计算结果
                self.result_text.insert(tk.END, f"计算参数:\n")
                for k, v in calc_result.items():
                    self.result_text.insert(tk.END, f"  {k}: {v}\n")
                self.result_text.see(tk.END)
                self.result_text.update()
                
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
                    "p_SWS": seg["p_SWS"] * 1e-3,  # 转换为米
                    "m": 50,
                    "Space_cut": 10,
                    "y_end": L + (results[-1]["y"][-1] if seg_idx > 0 and results else 0)
                }

                # 分段处理
                if seg["type"] == "initial":
                    self.handle_initial_segment(params, COMMON_PARAMS, calc_result, results)
                elif seg["type"] == "attenuator":
                    self.handle_attenuator_segment(params, results, seg_idx, Loss_attu, L)
                else:
                    self.handle_normal_segment(params, results, seg_idx)

            # ========================= 结果处理与可视化 =========================
            self.process_and_visualize(results, C_list, COMMON_PARAMS, SEGMENTS, Loss_attu)
            
        except Exception as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, f"\n计算错误: {str(e)}\n")
            self.result_text.see(tk.END)
            self.result_text.config(state=tk.DISABLED)
            messagebox.showerror("计算错误", f"计算过程中出错: {str(e)}")
    
    def build_input_params(self, common_params, seg):
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
            seg["Fn_K"], 
            common_params["f0_GHz"],
            seg["Vpc"]
        ]

    def handle_initial_segment(self, params, common_params, calc_result, results):
        """处理初始段"""
        # 使用全局参数中的输入功率P_in
        P_in = common_params["P_in"]
        P_flux = params["C"] * common_params["I"] * common_params["V"] * 2
        params.update({
            "A0": np.sqrt(P_in / P_flux),  # 使用用户输入的P_in
            "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"]
        })
        self.result_text.insert(tk.END, f"\n初始段参数:\n{params}\n")
        self.result_text.insert(tk.END, f"使用输入功率: {P_in} W\n")  # 添加日志
        results.append(solveTWTNOLINE_INIT(**params))
        self.result_text.update()

    def handle_attenuator_segment(self, params, results, seg_idx, Loss_attu, L):
        """处理衰减段"""
        prev = results[seg_idx-1]
        d_attu = 0.01836 * Loss_attu / (L / (2 * np.pi))
        self.result_text.insert(tk.END, f"\n衰减段损耗系数: {d_attu}\n")
        params.update({
            "result_y_ends": prev["y"][-1],
            "result_A_ends": prev["A_Ends"] * 10**(-Loss_attu/20),
            "result_dA_dy": prev["dA_dy_Ends"] * 0,  # 置零
            "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"],
            "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_Drift(**params))
        self.result_text.update()

    def handle_normal_segment(self, params, results, seg_idx):
        """处理常规段"""
        prev = results[seg_idx-1]
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
        self.result_text.update()

    def process_and_visualize(self, results, C_list, common_params, segments, Loss_attu):
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
        
        # 显示最终结果
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, "\n======== 最终计算结果 ========\n")
        self.result_text.insert(tk.END, f"输入功率: {common_params['P_in']} W\n")  # 显示输入功率
        self.result_text.insert(tk.END, f"非线性理论增益: {10 * np.log10(P_Out[-1]/common_params['P_in']):.4f} dB\n")  # 使用输入功率计算增益
        self.result_text.insert(tk.END, f"输出功率: {P_Out[-1]:.4f} W\n")
        self.result_text.insert(tk.END, f"最大效率: {Eff_max:.4f}%\n")
        self.result_text.insert(tk.END, f"最大功率: {P_max:.4f} W\n")
        self.result_text.insert(tk.END, f"Loss_attu: {Loss_attu}\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        # 可视化
        self.visualize_results(Y_Finall, A_Fianll, theta_Fianll, u_Finall, Lenth, P_Out, common_params['P_in'], results[-1])

    def visualize_results(self, Y, A, theta, u, Lenth, P_Out, P_in, final_seg):
        """可视化绘图并保存每个子图"""
        # 创建文件夹保存结果
        timestamp = datetime.now().strftime("%Y%m")
        folder_path = f".\Results\TWT_Plots_{timestamp}"
        os.makedirs(folder_path, exist_ok=True)
        
        self.fig.clear()
        
        # 1. 振幅演化
        ax1 = self.fig.add_subplot(2, 3, 1)
        ax1.plot(Y, A, 'navy')
        ax1.set_xlabel("Position y", fontsize=10)
        ax1.set_ylabel("Amplitude A(y)", fontsize=10)
        ax1.set_title("Amplitude Growth", fontsize=12)
        ax1.grid(alpha=0.3)
        
        # 单独保存振幅演化图
        fig_amp = plt.figure(figsize=(8, 6))
        ax_amp = fig_amp.add_subplot(111)
        ax_amp.plot(Y, A, 'navy')
        ax_amp.set_xlabel("Position y", fontsize=10)
        ax_amp.set_ylabel("Amplitude A(y)", fontsize=10)
        ax_amp.set_title("Amplitude Growth", fontsize=12)
        ax_amp.grid(alpha=0.3)
        fig_amp.savefig(os.path.join(folder_path, "1_Amplitude_Growth.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_amp)
        
        # 2. 相位演化
        ax2 = self.fig.add_subplot(2, 3, 2)
        ax2.plot(Y, theta, 'maroon')
        ax2.set_xlabel("Position y", fontsize=10)
        ax2.set_ylabel("Phase Shift θ(y)", fontsize=10)
        ax2.set_title("Phase Evolution", fontsize=12)
        ax2.grid(alpha=0.3)
        
        # 单独保存相位演化图
        fig_phase = plt.figure(figsize=(8, 6))
        ax_phase = fig_phase.add_subplot(111)
        ax_phase.plot(Y, theta, 'maroon')
        ax_phase.set_xlabel("Position y", fontsize=10)
        ax_phase.set_ylabel("Phase Shift θ(y)", fontsize=10)
        ax_phase.set_title("Phase Evolution", fontsize=12)
        ax_phase.grid(alpha=0.3)
        fig_phase.savefig(os.path.join(folder_path, "2_Phase_Evolution.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_phase)
        
        # 3. 速度分布
        ax3 = self.fig.add_subplot(2, 3, 3)
        scatter = ax3.scatter(
            final_seg["phi_final"], final_seg["u_final"],
            c=final_seg["phi_final"], cmap='hsv', s=20, edgecolor='k', lw=0.5
        )
        self.fig.colorbar(scatter, ax=ax3, label="Final Phase ϕ(y_end)")
        ax3.set_xlabel("Final Phase ϕ(y_end)", fontsize=10)
        ax3.set_ylabel("Final Velocity u(y_end)", fontsize=10)
        ax3.set_title("Velocity Distribution", fontsize=12)
        ax3.grid(alpha=0.3)
        
        # 单独保存速度分布图
        fig_vel = plt.figure(figsize=(8, 6))
        ax_vel = fig_vel.add_subplot(111)
        vel_scatter = ax_vel.scatter(
            final_seg["phi_final"], final_seg["u_final"],
            c=final_seg["phi_final"], cmap='hsv', s=20, edgecolor='k', lw=0.5
        )
        fig_vel.colorbar(vel_scatter, ax=ax_vel, label="Final Phase ϕ(y_end)")
        ax_vel.set_xlabel("Final Phase ϕ(y_end)", fontsize=10)
        ax_vel.set_ylabel("Final Velocity u(y_end)", fontsize=10)
        ax_vel.set_title("Velocity Distribution", fontsize=12)
        ax_vel.grid(alpha=0.3)
        fig_vel.savefig(os.path.join(folder_path, "3_Velocity_Distribution.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_vel)
        
        # 4. 相位分布
        ax4 = self.fig.add_subplot(2, 3, 4)
        scatter = ax4.scatter(
            final_seg["phi0_grid"], final_seg["phi_final"],
            c=final_seg["phi0_grid"], cmap='hsv', s=20, edgecolor='k', lw=0.5
        )
        self.fig.colorbar(scatter, ax=ax4, label="Initial Phase")
        ax4.set_xlabel("Initial Phase ϕ₀", fontsize=10)
        ax4.set_ylabel("Final Phase ϕ(y_end)", fontsize=10)
        ax4.set_title("Phase Distribution", fontsize=12)
        ax4.grid(alpha=0.3)
        
        # 单独保存相位分布图
        fig_phase_dist = plt.figure(figsize=(8, 6))
        ax_phase_dist = fig_phase_dist.add_subplot(111)
        phase_scatter = ax_phase_dist.scatter(
            final_seg["phi0_grid"], final_seg["phi_final"],
            c=final_seg["phi0_grid"], cmap='hsv', s=20, edgecolor='k', lw=0.5
        )
        fig_phase_dist.colorbar(phase_scatter, ax=ax_phase_dist, label="Initial Phase")
        ax_phase_dist.set_xlabel("Initial Phase ϕ₀", fontsize=10)
        ax_phase_dist.set_ylabel("Final Phase ϕ(y_end)", fontsize=10)
        ax_phase_dist.set_title("Phase Distribution", fontsize=12)
        ax_phase_dist.grid(alpha=0.3)
        fig_phase_dist.savefig(os.path.join(folder_path, "4_Phase_Distribution.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_phase_dist)
        
        # 5. 电子相空间
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax5.plot(Lenth, u, 'navy')
        ax5.set_xlabel("Position Z(Interaction Length)", fontsize=10)
        ax5.set_ylabel("Electron Velocity (u)", fontsize=10)
        ax5.set_title("Electron Phase Space", fontsize=12)
        ax5.grid(alpha=0.3)
        
        # 单独保存电子相空间图
        fig_elec = plt.figure(figsize=(8, 6))
        ax_elec = fig_elec.add_subplot(111)
        ax_elec.plot(Lenth, u, 'navy')
        ax_elec.set_xlabel("Position Z(Interaction Length)", fontsize=10)
        ax_elec.set_ylabel("Electron Velocity (u)", fontsize=10)
        ax_elec.set_title("Electron Phase Space", fontsize=12)
        ax_elec.grid(alpha=0.3)
        fig_elec.savefig(os.path.join(folder_path, "5_Electron_Phase_Space.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_elec)
        
        # 6. 功率演化
        ax6 = self.fig.add_subplot(2, 3, 6)
        ax6.plot(Lenth, P_Out, 'darkgreen')
        ax6.axhline(y=P_in, color='gray', linestyle='--', alpha=0.7, label='Input Power')
        ax6.set_xlabel("Position Z(Interaction Length)", fontsize=10)
        ax6.set_ylabel("Output Power (W)", fontsize=10)
        ax6.set_title("Power Evolution", fontsize=12)
        ax6.legend(loc='best')
        ax6.grid(alpha=0.3)
        
        # 单独保存功率演化图
        fig_power = plt.figure(figsize=(8, 6))
        ax_power = fig_power.add_subplot(111)
        ax_power.plot(Lenth, P_Out, 'darkgreen')
        ax_power.axhline(y=P_in, color='gray', linestyle='--', alpha=0.7, label='Input Power')
        ax_power.set_xlabel("Position Z(Interaction Length)", fontsize=10)
        ax_power.set_ylabel("Output Power (W)", fontsize=10)
        ax_power.set_title("Power Evolution", fontsize=12)
        ax_power.legend(loc='best')
        ax_power.grid(alpha=0.3)
        fig_power.savefig(os.path.join(folder_path, "6_Power_Evolution.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_power)
        
        # 保存完整图形
        self.fig.savefig(os.path.join(folder_path, "Complete_Plot.png"), dpi=300, bbox_inches='tight')
        
        # 绘制完整图形
        self.fig.tight_layout()
        self.canvas.draw()
        
        # 在结果窗口中显示导出信息
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, f"\n所有绘图结果已保存到文件夹: {folder_path}\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)



if __name__ == "__main__":
    root = tk.Tk()
    app = TWT_GUI(root)
    root.mainloop()