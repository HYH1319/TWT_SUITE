import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os

from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VSHEETBEAM_VF import solveTWTNOLINE_OUTPUT, solveTWTNOLINE_INIT, solveTWTNOLINE_Drift

class TWT_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("行波管计算器")
        self.root.geometry("1200x800")
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.input_frame = ttk.LabelFrame(self.main_frame, text="参数设置")
        self.input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.LEFT)
        
        self.result_frame = ttk.LabelFrame(self.main_frame, text="计算结果")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.RIGHT)
        
        self.create_input_controls()
        self.create_result_display()
        
        self.default_segments = [
            {"len": 50, "Vpc": 0.2893, "p_SWS": 0.50, "Kc": 3.88, "Loss_perunit": 0, "Fn_K": 1, "type": "initial"},
        ]
        self.load_defaults()

    def create_input_controls(self):
        """创建参数输入控件"""
        global_frame = ttk.LabelFrame(self.input_frame, text="全局参数")
        global_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(global_frame, text="电流 I (A):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.I_var = tk.DoubleVar(value=0.3)
        ttk.Entry(global_frame, textvariable=self.I_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(global_frame, text="电压 V (V):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.V_var = tk.DoubleVar(value=23000)
        ttk.Entry(global_frame, textvariable=self.V_var, width=10).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(global_frame, text="频率 f0 (GHz):").grid(row=0, column=4, padx=5, pady=2, sticky=tk.W)
        self.f0_var = tk.DoubleVar(value=211)
        ttk.Entry(global_frame, textvariable=self.f0_var, width=10).grid(row=0, column=5, padx=5, pady=2)

        ttk.Label(global_frame, text="宽度 w (mm):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.w_var = tk.DoubleVar(value=0.2)
        ttk.Entry(global_frame, textvariable=self.w_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(global_frame, text="厚度 t (mm):").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.t_var = tk.DoubleVar(value=0.2)
        ttk.Entry(global_frame, textvariable=self.t_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(global_frame, text="输入功率 P_in (W):").grid(row=1, column=4, padx=5, pady=2, sticky=tk.W)
        self.P_in_var = tk.DoubleVar(value=0.10)
        ttk.Entry(global_frame, textvariable=self.P_in_var, width=10).grid(row=1, column=5, padx=5, pady=2)

        ttk.Label(global_frame, text="Loss_attu:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.loss_attu_var = tk.DoubleVar(value=0)
        ttk.Entry(global_frame, textvariable=self.loss_attu_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # ================= 修改点 1：使用 Canvas 实现滚动区域 =================
        segment_header = ttk.LabelFrame(self.input_frame, text="分段参数")
        segment_header.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas_frame = ttk.Frame(segment_header)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.seg_canvas = tk.Canvas(canvas_frame, height=150)
        seg_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.seg_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.seg_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.seg_canvas.configure(scrollregion=self.seg_canvas.bbox("all"))
        )

        self.seg_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.seg_canvas.configure(yscrollcommand=seg_scrollbar.set)

        self.seg_canvas.pack(side="left", fill="both", expand=True)
        seg_scrollbar.pack(side="right", fill="y")
        # =====================================================================

        columns = ("len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K", "type")
        headers = ("周期数(个)", "Vpc", "螺距", "耦合阻抗", "每单元损耗", "填充因子", "类型")
        for col, header in enumerate(headers):
            ttk.Label(self.scrollable_frame, text=header, width=10).grid(row=0, column=col, padx=2, pady=2)

        # ================= 修改点 2：初始化为空列表，不再写死10行循环 =================
        self.segment_entries = []
        self.entry_frames = []
        # =========================================================================

        # ================= 修改点 3：增加动态添加/删除按钮 =================
        button_frame = ttk.Frame(self.input_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(button_frame, text="加载默认值", command=self.load_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="添加分段", command=self.add_segment_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="删除末段", command=self.remove_segment_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="计算", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="重置", command=self.reset).pack(side=tk.LEFT, padx=5)
        # =====================================================================

    # ================= 修改点 4：抽离单行生成的逻辑 =================
    def add_segment_row(self, initial_data=None):
        columns = ("len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K", "type")
        type_options = ["initial", "attenuator", "O"]
        
        entry_frame = ttk.Frame(self.scrollable_frame)
        row_num = len(self.segment_entries) + 1
        entry_frame.grid(row=row_num, column=0, columnspan=len(columns), sticky="ew", padx=2, pady=2)
        self.entry_frames.append(entry_frame)
        
        row_entries = []
        for j, param in enumerate(columns):
            var = tk.StringVar()
            entry = ttk.Entry(entry_frame, textvariable=var, width=10)
            entry.grid(row=0, column=j, padx=2, pady=2)
            row_entries.append(var)
            
        combo = ttk.Combobox(entry_frame, textvariable=row_entries[6], values=type_options, width=8)
        combo.grid(row=0, column=len(columns)-1, padx=2, pady=2)
        
        if initial_data:
            for i, param in enumerate(columns[:-1]):
                row_entries[i].set(initial_data.get(param, ""))
            row_entries[6].set(initial_data.get("type", "initial"))
            
        self.segment_entries.append(row_entries)

    def remove_segment_row(self):
        if self.segment_entries:
            self.segment_entries.pop()
            frame = self.entry_frames.pop()
            frame.destroy()
    # ==============================================================

    def create_result_display(self):
        notebook = ttk.Notebook(self.result_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        text_frame = ttk.Frame(notebook)
        self.result_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)
        notebook.add(text_frame, text="计算结果")

        plot_frame = ttk.Frame(notebook)
        self.fig = plt.figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Result shown Here", fontsize=12, ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.axis('off')
        self.canvas.draw()
        notebook.add(plot_frame, text="可视化结果")

    def load_defaults(self):
        self.I_var.set(0.3)
        self.V_var.set(23000)
        self.f0_var.set(211)
        self.P_in_var.set(0.10)
        self.w_var.set(0.2)
        self.t_var.set(0.2)
        self.loss_attu_var.set(0)
        
        # ================= 修改点 5：清空现有行并加载默认行 =================
        while self.segment_entries:
            self.remove_segment_row()
        self.add_segment_row(initial_data=self.default_segments[0])
        # ==============================================================

    def reset(self):
        self.I_var.set("")
        self.V_var.set("")
        self.f0_var.set("")
        self.P_in_var.set("")
        self.w_var.set("")
        self.t_var.set("")
        self.loss_attu_var.set("")
        
        # ================= 修改点 6：重置时清空所有动态行 =================
        while self.segment_entries:
            self.remove_segment_row()
        # ==============================================================

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "计算结果将在此处显示", fontsize=12, ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.axis('off')
        self.canvas.draw()

    def calculate(self):
        try:
            COMMON_PARAMS = {
                "I": self.I_var.get(), "V": self.V_var.get(), "f0": self.f0_var.get(),
                "P_in": self.P_in_var.get(), "w": self.w_var.get(), "t": self.t_var.get(),
            }
            Loss_attu = self.loss_attu_var.get()
            
            SEGMENTS = []
            columns = ("len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K", "type")
            for row in self.segment_entries:
                if any(entry.get() for entry in row):
                    seg = {}
                    for i, param in enumerate(columns):
                        value = row[i].get()
                        try:
                            value = float(value)
                        except ValueError:
                            value = str(value) if param == "type" else 0.0
                        seg[param] = value
                    SEGMENTS.append(seg)
                    
            if not SEGMENTS:
                raise ValueError("请至少输入一个分段参数")
            if COMMON_PARAMS["P_in"] <= 0:
                raise ValueError("输入功率必须大于0")

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "开始计算...\n")
            self.result_text.see(tk.END)
            self.result_text.update()

            results = []
            C_list = []
            for seg_idx, seg in enumerate(SEGMENTS):
                self.result_text.insert(tk.END, f"\n段{seg_idx}: " + ", ".join(f"{k}={v}" for k, v in seg.items()) + "\n")
                self.result_text.see(tk.END)
                self.result_text.update()

                input_params = self.build_input_params(COMMON_PARAMS, seg)
                calc_result = simple_calculation(*input_params)
                
                self.result_text.insert(tk.END, f"计算参数:\n")
                for k, v in calc_result.items():
                    self.result_text.insert(tk.END, f" {k}: {v}\n")
                self.result_text.see(tk.END)
                self.result_text.update()

                C = calc_result["小信号增益因子C"]
                L = 2 * np.pi * calc_result["互作用长度N"] * C
                C_list.append(C)

                Rn_sqr_values=calc_result["Rowe特征值R"]**2 if COMMON_PARAMS["w"] == COMMON_PARAMS["t"] else calc_result["等离子体频率降低因子Fn"]**2
                wp_omega = calc_result["等离子体频率Wp"] / (2 * np.pi * COMMON_PARAMS["f0"] * 1e9)  # 使用全局频率
                Space_CUT=20
                beam_params=[Rn_sqr_values,wp_omega,Space_CUT]

                params = {
                    "C": C, 
                    "b": calc_result["非同步参量b"],
                    "d": calc_result["损耗因子d"],
                    "beam_params": beam_params,
                    "m": 64,
                    "y_end": L + (results[-1]["y"][-1] if seg_idx > 0 and results else 0)
                }

                if seg["type"] == "initial":
                    self.handle_initial_segment(params, COMMON_PARAMS, calc_result, results)
                elif seg["type"] == "attenuator":
                    self.handle_attenuator_segment(params, results, seg_idx, Loss_attu, L)
                else:
                    self.handle_normal_segment(params, results, seg_idx)

            self.process_and_visualize(results, C_list, COMMON_PARAMS, SEGMENTS, Loss_attu)
        except Exception as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, f"\n计算错误: {str(e)}\n")
            self.result_text.see(tk.END)
            self.result_text.config(state=tk.DISABLED)
            messagebox.showerror("计算错误", f"计算过程中出错: {str(e)}")

    def build_input_params(self, common_params, seg):
        return [common_params["I"], common_params["V"], seg["Kc"], seg["Loss_perunit"], seg["p_SWS"], seg["len"], common_params["w"], common_params["t"], seg["Fn_K"], common_params["f0"], seg["Vpc"]]

    def handle_initial_segment(self, params, common_params, calc_result, results):
        P_in = common_params["P_in"]
        P_flux = params["C"] * common_params["I"] * common_params["V"] * 2
        params.update({
            "A0": np.sqrt(P_in / P_flux),
            "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"]
        })
        self.result_text.insert(tk.END, f"\n初始段参数:\n{params}\n使用输入功率: {P_in} W\n")
        results.append(solveTWTNOLINE_INIT(**params))
        self.result_text.update()

    def handle_attenuator_segment(self, params, results, seg_idx, Loss_attu, L):
        prev = results[seg_idx-1]
        d_attu = 0.01836 * Loss_attu / (L / (2 * np.pi))
        self.result_text.insert(tk.END, f"\n衰减段损耗系数: {d_attu}\n")
        params.update({
            "result_y_ends": prev["y"][-1], "result_A_ends": prev["A_Ends"] * 10**(-Loss_attu/20),
            "result_dA_dy": prev["dA_dy_Ends"] * 0, "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"], "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_Drift(**params))
        self.result_text.update()

    def handle_normal_segment(self, params, results, seg_idx):
        prev = results[seg_idx-1]
        params.update({
            "result_y_ends": prev["y"][-1], "result_A_ends": prev["A_Ends"],
            "result_dA_dy": prev["dA_dy_Ends"], "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"], "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_OUTPUT(**params))
        self.result_text.update()

    def process_and_visualize(self, results, C_list, common_params, segments, Loss_attu):
        Y_Finall = np.concatenate([r["y"] for r in results])
        A_Fianll = np.concatenate([r["A"] for r in results])
        theta_Fianll = np.concatenate([r["theta"] for r in results])
        u_Finall = np.concatenate([r["u_now"] for r in results])
        P_Out = 2 * common_params["I"] * common_params["V"] * np.concatenate([C_list[i] * (results[i]["A"]**2) for i in range(len(segments))])
        
        P_max = P_Out.max()
        Eff_max = P_max / (common_params["I"] * common_params["V"]) * 100
        Lenth = Y_Finall / (2 * np.pi * np.mean(C_list))
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, "\n======== 最终计算结果 ========\n")
        self.result_text.insert(tk.END, f"工作频率: {common_params['f0']} GHz\n输入功率: {common_params['P_in']} W\n")
        self.result_text.insert(tk.END, f"非线性理论增益: {10 * np.log10(P_Out[-1]/common_params['P_in']):.4f} dB\n")
        self.result_text.insert(tk.END, f"输出功率: {P_Out[-1]:.4f} W\n最大效率: {Eff_max:.4f}%\n最大功率: {P_max:.4f} W\nLoss_attu: {Loss_attu}\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        self.visualize_results(Y_Finall, A_Fianll, theta_Fianll, u_Finall, Lenth, P_Out, common_params['P_in'], results[-1])

    def visualize_results(self, Y, A, theta, u, Lenth, P_Out, P_in, final_seg):
        timestamp = datetime.now().strftime("%Y%m")
        
        # ================= 修改点 7：修复路径转义Bug =================
        folder_path = os.path.join(".", "Results", f"TWT_Plots_{timestamp}")
        # ==============================================================
        os.makedirs(folder_path, exist_ok=True)
        
        self.fig.clear()
        
        # 1. 振幅演化
        ax1 = self.fig.add_subplot(2, 3, 1)
        ax1.plot(Y, A, 'navy'); ax1.set_xlabel("Position y"); ax1.set_ylabel("Amplitude A(y)"); ax1.set_title("Amplitude Growth"); ax1.grid(alpha=0.3)
        fig_amp = plt.figure(figsize=(8, 6)); ax_amp = fig_amp.add_subplot(111)
        ax_amp.plot(Y, A, 'navy'); ax_amp.set_xlabel("Position y"); ax_amp.set_ylabel("Amplitude A(y)"); ax_amp.set_title("Amplitude Growth"); ax_amp.grid(alpha=0.3)
        fig_amp.savefig(os.path.join(folder_path, "1_Amplitude_Growth.png"), dpi=300, bbox_inches='tight'); plt.close(fig_amp)

        # 2. 相位演化
        ax2 = self.fig.add_subplot(2, 3, 2)
        ax2.plot(Y, theta, 'maroon'); ax2.set_xlabel("Position y"); ax2.set_ylabel("Phase Shift θ(y)"); ax2.set_title("Phase Evolution"); ax2.grid(alpha=0.3)
        fig_phase = plt.figure(figsize=(8, 6)); ax_phase = fig_phase.add_subplot(111)
        ax_phase.plot(Y, theta, 'maroon'); ax_phase.set_xlabel("Position y"); ax_phase.set_ylabel("Phase Shift θ(y)"); ax_phase.set_title("Phase Evolution"); ax_phase.grid(alpha=0.3)
        fig_phase.savefig(os.path.join(folder_path, "2_Phase_Evolution.png"), dpi=300, bbox_inches='tight'); plt.close(fig_phase)

        # 3. 速度分布
        ax3 = self.fig.add_subplot(2, 3, 3)
        scatter = ax3.scatter(final_seg["phi_final"], final_seg["u_final"], c=final_seg["phi_final"], cmap='hsv', s=20, edgecolor='k', lw=0.5)
        self.fig.colorbar(scatter, ax=ax3, label="Final Phase"); ax3.set_xlabel("Final Phase"); ax3.set_ylabel("Final Velocity"); ax3.set_title("Velocity Distribution"); ax3.grid(alpha=0.3)
        fig_vel = plt.figure(figsize=(8, 6)); ax_vel = fig_vel.add_subplot(111)
        vel_scatter = ax_vel.scatter(final_seg["phi_final"], final_seg["u_final"], c=final_seg["phi_final"], cmap='hsv', s=20, edgecolor='k', lw=0.5)
        fig_vel.colorbar(vel_scatter, ax=ax_vel, label="Final Phase"); ax_vel.set_xlabel("Final Phase"); ax_vel.set_ylabel("Final Velocity"); ax_vel.set_title("Velocity Distribution"); ax_vel.grid(alpha=0.3)
        fig_vel.savefig(os.path.join(folder_path, "3_Velocity_Distribution.png"), dpi=300, bbox_inches='tight'); plt.close(fig_vel)

        # 4. 相位分布
        ax4 = self.fig.add_subplot(2, 3, 4)
        scatter = ax4.scatter(final_seg["phi0_grid"], final_seg["phi_final"], c=final_seg["phi0_grid"], cmap='hsv', s=20, edgecolor='k', lw=0.5)
        self.fig.colorbar(scatter, ax=ax4, label="Initial Phase"); ax4.set_xlabel("Initial Phase"); ax4.set_ylabel("Final Phase"); ax4.set_title("Phase Distribution"); ax4.grid(alpha=0.3)
        fig_phase_dist = plt.figure(figsize=(8, 6)); ax_phase_dist = fig_phase_dist.add_subplot(111)
        phase_scatter = ax_phase_dist.scatter(final_seg["phi0_grid"], final_seg["phi_final"], c=final_seg["phi0_grid"], cmap='hsv', s=20, edgecolor='k', lw=0.5)
        fig_phase_dist.colorbar(phase_scatter, ax=ax_phase_dist, label="Initial Phase"); ax_phase_dist.set_xlabel("Initial Phase"); ax_phase_dist.set_ylabel("Final Phase"); ax_phase_dist.set_title("Phase Distribution"); ax_phase_dist.grid(alpha=0.3)
        fig_phase_dist.savefig(os.path.join(folder_path, "4_Phase_Distribution.png"), dpi=300, bbox_inches='tight'); plt.close(fig_phase_dist)

        # 5. 电子相空间
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax5.plot(Lenth, u, 'navy'); ax5.set_xlabel("Position Z"); ax5.set_ylabel("Electron Velocity"); ax5.set_title("Electron Phase Space"); ax5.grid(alpha=0.3)
        fig_elec = plt.figure(figsize=(8, 6)); ax_elec = fig_elec.add_subplot(111)
        ax_elec.plot(Lenth, u, 'navy'); ax_elec.set_xlabel("Position Z"); ax_elec.set_ylabel("Electron Velocity"); ax_elec.set_title("Electron Phase Space"); ax_elec.grid(alpha=0.3)
        fig_elec.savefig(os.path.join(folder_path, "5_Electron_Phase_Space.png"), dpi=300, bbox_inches='tight'); plt.close(fig_elec)

        # 6. 功率演化
        ax6 = self.fig.add_subplot(2, 3, 6)
        ax6.plot(Lenth, P_Out, 'darkgreen'); ax6.axhline(y=P_in, color='gray', linestyle='--', alpha=0.7, label='Input Power')
        ax6.set_xlabel("Position Z"); ax6.set_ylabel("Output Power (W)"); ax6.set_title("Power Evolution"); ax6.legend(loc='best'); ax6.grid(alpha=0.3)
        fig_power = plt.figure(figsize=(8, 6)); ax_power = fig_power.add_subplot(111)
        ax_power.plot(Lenth, P_Out, 'darkgreen'); ax_power.axhline(y=P_in, color='gray', linestyle='--', alpha=0.7, label='Input Power')
        ax_power.set_xlabel("Position Z"); ax_power.set_ylabel("Output Power (W)"); ax_power.set_title("Power Evolution"); ax_power.legend(loc='best'); ax_power.grid(alpha=0.3)
        fig_power.savefig(os.path.join(folder_path, "6_Power_Evolution.png"), dpi=300, bbox_inches='tight'); plt.close(fig_power)

        self.fig.savefig(os.path.join(folder_path, "Complete_Plot.png"), dpi=300, bbox_inches='tight')
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, f"\n所有绘图结果已保存到文件夹: {folder_path}\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = TWT_GUI(root)
    root.mainloop()
