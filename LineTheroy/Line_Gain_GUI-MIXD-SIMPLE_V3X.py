import tkinter as tk
from tkinter import ttk, messagebox
import sys

from TWT_CORE_MORE_COMPLEX import detailed_calculation

class TripleCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("三参量计算器")
        self.gmax_values = {1: None, 2: None, 3: None}
        self.A_values={1: None}
        self.Ab_values = {1: None, 2: None}
        self.loss_values = {1: None, 2: None, 3: None}
        
        # 参数名称列表（与输入顺序一致）
        self.param_labels = [
            "电流 I(A)",
            "电压 (V)",
            "耦合阻抗 Kc*",
            "每单元损耗L*",
            "截断长度 Lenth_Att",
            "周期长度 p_SWS",
            "周期个数 N_Unit",
            "束流宽度 w",
            "束流厚度 t",
            "Fn_K*",
            "频率 freq*",
            "相速度 Vpc*",
        ]
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建三组输入面板
        self.create_input_group(main_frame, "参数组1", 0)
        self.create_input_group(main_frame, "参数组2", 1)
        self.create_input_group(main_frame, "参数组3", 2)
        
        # 添加全局计算按钮和结果显示
        self.create_global_controls(main_frame)
        
    def create_input_group(self, parent, title, column):
        """创建参数输入面板"""
        group_frame = ttk.LabelFrame(parent, text=title, padding="5")
        group_frame.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")
        
        # 参数默认值
        parameters = [
            ("0.3"),    # 电流
            ("23000"),  # 电压
            ("3.8812150306728705"),      # 耦合阻抗
            ("0"),      # 损耗
            ("0"),      # 截断长度
            ("0.50"),   # 周期长度
            ("50"),     # 周期个数（必须整数）
            ("0.20"),   # 束流宽度
            ("0.20"),   # 束流厚度
            ("1"),      # Fn_K
            ("211"),    # 频率
            ("0.2893778968136611"),  # 相速度
        ]
        
        # 创建输入组件
        entries = []
        for idx, default in enumerate(parameters):
            frame = ttk.Frame(group_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=self.param_labels[idx], width=15)
            label.pack(side=tk.LEFT)
            
            entry = ttk.Entry(frame)
            entry.insert(0, default)
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            entries.append(entry)
        
        # 计算按钮
        group_num = column + 1
        ttk.Button(group_frame, 
                 text="执行计算", 
                 command=lambda: self.run_calculation(entries, group_num)
                ).pack(pady=5)
        
        # 结果展示区
        result_text = tk.Text(group_frame, height=12, width=35)
        result_text.pack(fill=tk.BOTH, expand=True)
        setattr(self, f"result_text_{group_num}", result_text)
        
    def create_global_controls(self, parent):
        """创建全局控制组件"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky="ew")
        
        ttk.Button(control_frame,
                 text="计算 Required Return_Loss",
                 command=self.calculate_required_return_loss
                ).pack(side=tk.LEFT, padx=5)
        
        self.global_result = ttk.Label(control_frame, text="等待计算...")
        self.global_result.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame,
                 text="退出",
                 command=self.root.quit
                ).pack(side=tk.RIGHT, padx=5)

    def run_calculation(self, entries, group_num):
        """执行核心计算"""
        try:
            inputs = []
            for i, entry in enumerate(entries):
                param_name = self.param_labels[i]
                raw_value = entry.get().strip()  # 去除前后空格
                
                try:
                    if i == 6:  # 周期个数需转换为整数
                        value = int(raw_value)
                    else:
                        value = float(raw_value)
                except ValueError:
                    error_msg = (
                        f"参数 [{param_name}] 输入无效: '{raw_value}'\n"
                        f"要求: {'整数' if i == 6 else '数字'}"
                    )
                    raise ValueError(error_msg)
                
                inputs.append(value)
            
            result = detailed_calculation(*inputs)
            
            # 保存关键参数
            self.gmax_values[group_num] = result['线性最大增益Gmax']
            self.A_values[group_num]=result['初始化调制增益降低量A']
            self.loss_values[group_num] = inputs[3]*inputs[6]
            
            if group_num in (1, 2):
                self.Ab_values[group_num] = result['衰减降低增益量Ab']
            
            # 显示结果
            result_text = getattr(self, f"result_text_{group_num}")
            result_text.delete(1.0, tk.END)
            for key, value in result.items():
                result_text.insert(tk.END, f"{key}: {value:.6f}\n")
                
        except Exception as e:
            messagebox.showerror("输入错误" if isinstance(e, ValueError) else "计算错误", 
                              str(e))

    def calculate_required_return_loss(self):
        """计算最终结果"""
        try:
            missing_groups = []
            for k in self.gmax_values:
                if self.gmax_values[k] is None:
                    missing_groups.append(str(k))
            if missing_groups:
                raise ValueError(f"请先完成参数组 {', '.join(missing_groups)} 的计算")
            
            Final_G = (self.gmax_values[1] + self.Ab_values[1] +
                      self.gmax_values[2] + self.Ab_values[2] +
                      self.gmax_values[3]+self.A_values[1])
            
            total_loss = sum(self.loss_values.values())
            Required_Return_LOSS = (Final_G - total_loss) / 2
            
            result_str = (
                f"最终增益 Final_G: {Final_G:.6f}\n"
                f"总损耗: {total_loss:.6f}\n"
                f"Required Return_Loss: {Required_Return_LOSS:.6f}"
            )
            self.global_result.config(text=result_str)
            
            print("\n" + "="*40)
            print(result_str)
            print("="*40)
            
        except Exception as e:
            messagebox.showerror("计算错误", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = TripleCalculatorApp(root)
    root.mainloop()