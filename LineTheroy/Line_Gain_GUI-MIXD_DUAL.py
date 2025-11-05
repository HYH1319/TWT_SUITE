import tkinter as tk
from tkinter import ttk, messagebox
import sys

from TWT_CORE_COMPLEX import detailed_calculation

class CalculationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("双参数组计算器")
        self.gmax1 = None
        self.gmax2 = None
        self.loss2 = None
        self.A_val = None
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建两组输入面板
        self.create_input_group(main_frame, "参数组1", 0)
        self.create_input_group(main_frame, "参数组2", 1)
        
        # 添加全局计算按钮和结果显示
        self.create_global_controls(main_frame)
        
    def create_input_group(self, parent, title, column):
        """创建参数输入面板"""
        group_frame = ttk.LabelFrame(parent, text=title, padding="5")
        group_frame.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")
        
        # 参数列表及默认值
        parameters = [
            ("电流 I(A)", "0.3"),
            ("电压 (V)", "23000"),
            ("耦合阻抗 Kc*", "3.6"),
            ("每单元损耗Loss_perunit*", "0"),
            ("周期长度 p_SWS", "0.50"),
            ("周期个数 N_Unit", "50"),
            ("束流宽度 w", "0.20"),
            ("束流厚度 t", "0.20"),
            ("Fn_K*", "1"),
            ("频率 freq*", "211"),
            ("相速度 Vpc*", "0.288"),

        ]
        
        # 创建输入组件
        entries = []
        for text, default in parameters:
            frame = ttk.Frame(group_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=text, width=15).pack(side=tk.LEFT)
            entry = ttk.Entry(frame)
            entry.insert(0, default)
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            entries.append(entry)
        
        # 计算按钮
        ttk.Button(group_frame, 
                 text="执行计算", 
                 command=lambda: self.run_calculation(entries, column)
                ).pack(pady=5)
        
        # 结果展示区
        result_text = tk.Text(group_frame, height=12, width=40)
        result_text.pack(fill=tk.BOTH, expand=True)
        setattr(self, f"result_text_{column}", result_text)
        
    def create_global_controls(self, parent):
        """创建全局控制组件"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        
        # 全局计算按钮
        ttk.Button(control_frame,
                 text="计算 Required Return_Loss",
                 command=self.calculate_required_return_loss
                ).pack(side=tk.LEFT, padx=5)
        
        # 结果显示标签
        self.global_result = ttk.Label(control_frame, text="等待计算...")
        self.global_result.pack(side=tk.LEFT, padx=10)
        
        # 退出按钮
        ttk.Button(control_frame,
                 text="退出",
                 command=self.root.quit
                ).pack(side=tk.RIGHT, padx=5)

    def run_calculation(self, entries, group_num):
        """执行核心计算"""
        try:
            # 获取输入值
            inputs = []
            for i, entry in enumerate(entries):
                value = entry.get()
                if i == 5:  # 周期个数需转换为整数
                    inputs.append(int(value))
                else:
                    inputs.append(float(value))
            
            # 执行计算
            result = detailed_calculation(*inputs)
            self.A_val=result['初始化调制增益降低量A']
            # 保存关键参数
            if group_num == 0:
                self.gmax1 = result['线性最大增益Gmax']
                self.Ab=result['衰减降低增益量Ab']
                self.loss = inputs[3]*inputs[5]
            elif group_num == 1:
                self.gmax2 = result['线性最大增益Gmax']
                self.loss2 = inputs[3]*inputs[5]  # 保存参数组2的Loss值
            
            # 显示结果
            result_text = getattr(self, f"result_text_{group_num}")
            result_text.delete(1.0, tk.END)
            for key, value in result.items():
                result_text.insert(tk.END, f"{key}: {value:.6f}\n")
                
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
        except Exception as e:
            messagebox.showerror("计算错误", str(e))

    def calculate_required_return_loss(self):
        """计算最终结果"""
        try:
            # 检查前置计算
            if None in (self.gmax1, self.gmax2, self.loss2):
                raise ValueError("请先完成两个参数组的计算")
            
            # 执行公式计算
            Required_Return_LOSS = (self.gmax1 + self.Ab + self.gmax2 - self.loss - self.loss2+self.A_val)/2
            Fiannal_G=self.gmax1 + self.Ab + self.gmax2+self.A_val
            
            # 更新界面显示
            result_str =(
                f"Required Return_Loss: {Required_Return_LOSS:.6f}\n"
                f"Total Gain of TWT: {Fiannal_G:.6f}\n"
                         ) 
            self.global_result.config(text=result_str)
            
            # 控制台打印
            print("\n" + "="*40)
            print(result_str)
            print("="*40)
            
        except ValueError as e:
            messagebox.showerror("错误", str(e))
        except Exception as e:
            messagebox.showerror("计算错误", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = CalculationApp(root)
    root.mainloop()