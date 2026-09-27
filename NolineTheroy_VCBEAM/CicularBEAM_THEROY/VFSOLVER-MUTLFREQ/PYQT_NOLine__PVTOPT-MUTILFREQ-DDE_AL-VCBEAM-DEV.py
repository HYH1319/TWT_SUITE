import sys
import os
import json
import logging
import time
import traceback
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from datetime import datetime
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox,
    QProgressBar, QFileDialog, QGridLayout, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER_MIX_WITH_PVT import calculate_SEGMENT_TWT_NOLINE

DE_STRATEGIES = [
    ("best/1/bin", "best1bin"),
    ("rand/1/bin", "rand1bin"),
    ("current-to-best/1/bin", "currenttobest1bin")
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_WORKERS = max(1, cpu_count())
CONFIG_FILE = "./configUDF/OPT/twt_DDE_config_MUTIL.json"

def adjust_params(p_SWS, idx, Vpc, Kc, coeffs):
    """参数调整函数"""
    current_p = p_SWS[idx]
    first_p = p_SWS[0]
    delta = (current_p - first_p) / first_p
    return {
        "Vpc": Vpc + coeffs['vpc'] * delta * Vpc,
        "Kc": Kc + coeffs['kc'] * delta * Kc
    }

def evaluate_single_wrapper(args):
    """包装函数用于多进程，避免重复创建evaluator"""
    p_SWS, fixed_params, freq_params, coeffs = args
    total_power = 0.0
    
    try:
        for freq_data in freq_params:
            para_func = partial(adjust_params, coeffs=coeffs)
            
            result = calculate_SEGMENT_TWT_NOLINE(
                I=fixed_params["i"],
                V=fixed_params["v"],
                Kc=freq_data['Kc'],
                Loss_perunit=fixed_params["loss_perunit"],
                SectionedSEGMENT_IDX=fixed_params["section_seg_idx"],
                p_SWS=p_SWS,
                N_unit=fixed_params["n_unit"],
                w=fixed_params["w"],
                t=fixed_params["t"],
                Fn_K=fixed_params["Fn_K"],
                f0_GHz=freq_data['freq'],
                Vpc=freq_data['Vpc'],
                para_func=para_func,
                P_in=fixed_params["p_in"],
                Loss_attu=fixed_params["loss_attu"]
            )
            total_power += result["输出功率P_out"]
        
        return -total_power
        
    except Exception as e:
        logger.error(f"计算错误: {str(e)}")
        return float("inf")

class MultiFreqFitnessEvaluator:
    """优化后的多频点评估器"""
    def __init__(self, fixed_params):
        self.fixed_params = fixed_params
        self.coeffs = {
            'vpc': fixed_params['vpc_coeff'],
            'kc': fixed_params['kc_coeff']
        }
        self.freq_params = []
        for i in range(1, 4):
            self.freq_params.append({
                'Kc': fixed_params[f'freq{i}_kc'],
                'Vpc': fixed_params[f'freq{i}_vpc'],
                'freq': fixed_params[f'freq{i}_freq']
            })
    
    def calculate_initial_power(self, p_SWS):
        """计算初始总功率"""
        args = (p_SWS, self.fixed_params, self.freq_params, self.coeffs)
        return -evaluate_single_wrapper(args)
    
    def __call__(self, p_SWS):
        """适应度评估函数"""
        args = (p_SWS, self.fixed_params, self.freq_params, self.coeffs)
        return evaluate_single_wrapper(args)

class DistributedDETask(QThread):
    """优化后的分布式差分进化任务"""
    finished = pyqtSignal(object)
    progress_updated = pyqtSignal(int, float)
    log_message = pyqtSignal(str)
    iteration_complete = pyqtSignal(int, float)
    initial_power_calculated = pyqtSignal(float)
    
    def __init__(self, fixed_params, freq_params, coeffs, bounds, pop_size, max_iter, 
                 workers, strategy='best1bin', F=0.5, CR=0.7, initial_p=None):
        super().__init__()
        self.fixed_params = fixed_params
        self.freq_params = freq_params
        self.coeffs = coeffs
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.workers = workers
        self.strategy = strategy
        self.F = F
        self.CR = CR
        self.initial_p = initial_p
        self.is_running = False
        self.pool = None
        self.best_fitness = float('inf')
        self.best_params = None
        self.current_iter = 0
        
        self.bounds_low = np.array([b[0] for b in bounds])
        self.bounds_high = np.array([b[1] for b in bounds])
        self.n_dim = len(bounds)
    
    def __del__(self):
        self.stop()
    
    def run(self):
        try:
            self.is_running = True
            
            self.log_message.emit(f"创建进程池 ({self.workers} 个进程)...")
            self.pool = Pool(processes=self.workers)
            
            pop = np.random.uniform(
                low=self.bounds_low,
                high=self.bounds_high,
                size=(self.pop_size, self.n_dim)
            )
            
            if self.initial_p is not None and len(self.initial_p) == self.n_dim:
                pop[0] = self.initial_p
                self.log_message.emit(f"使用用户提供的初始参数")
                
                args = (self.initial_p, self.fixed_params, self.freq_params, self.coeffs)
                initial_power = -evaluate_single_wrapper(args)
                self.initial_power_calculated.emit(initial_power)
            
            self.log_message.emit(f"评估初始种群 ({self.pop_size}个个体)...")
            args_list = [(p, self.fixed_params, self.freq_params, self.coeffs) for p in pop]
            fitness = np.array(self.pool.map(evaluate_single_wrapper, args_list))
            
            min_idx = np.argmin(fitness)
            self.best_fitness = fitness[min_idx]
            self.best_params = pop[min_idx].copy()
            best_power = -self.best_fitness
            self.log_message.emit(f"初始最佳功率: {best_power:.2f} W")
            
            for iter_num in range(1, self.max_iter + 1):
                if not self.is_running:
                    break
                
                self.current_iter = iter_num
                
                mutant = self._mutate_vectorized(pop, fitness)
                trial = self._crossover_vectorized(pop, mutant)
                
                args_list = [(t, self.fixed_params, self.freq_params, self.coeffs) for t in trial]
                trial_fitness = np.array(self.pool.map(evaluate_single_wrapper, args_list))
                
                improvement_mask = trial_fitness < fitness
                if np.any(improvement_mask):
                    pop[improvement_mask] = trial[improvement_mask]
                    fitness[improvement_mask] = trial_fitness[improvement_mask]
                    
                    min_idx = np.argmin(fitness)
                    if fitness[min_idx] < self.best_fitness:
                        self.best_fitness = fitness[min_idx]
                        self.best_params = pop[min_idx].copy()
                
                progress = int(iter_num / self.max_iter * 100)
                power_val = -self.best_fitness
                
                if iter_num % 1 == 0 or iter_num == self.max_iter:
                    self.iteration_complete.emit(iter_num, power_val)
                    self.progress_updated.emit(progress, power_val)
                    self.log_message.emit(
                        f"迭代 {iter_num}/{self.max_iter}, "
                        f"最佳总功率: {power_val:.2f} W"
                    )
            
            if self.is_running:
                self.finished.emit({
                    'optimized_p_SWS': self.best_params,
                    'best_power': -self.best_fitness
                })
        
        except Exception as e:
            self.log_message.emit(f"分布式DE出错: {str(e)}\n{traceback.format_exc()}")
            self.finished.emit(None)
        finally:
            self.stop()
    
    def _mutate_vectorized(self, pop, fitness):
        """
        正确且稳健的DE变异操作。
        
        关键改进：
        1. 保证为每个个体选择的索引是唯一的。
        2. 包含边界检查，确保变异向量在可行域内。
        """
        n_pop, n_dim = pop.shape
        mutants = np.empty_like(pop)
        
        # 找到当前种群的最佳个体（如果策略需要）
        if 'best' in self.strategy:
            best_idx = np.argmin(fitness)
            best_vec = pop[best_idx]

        # 对种群中的每个个体进行变异
        for i in range(n_pop):
            # --- 关键步骤：为个体 i 生成唯一的索引 ---
            # 创建候选索引池（排除自身 i）
            candidates = list(range(n_pop))
            candidates.remove(i)
            
            if self.strategy == 'rand1bin':
                # rand/1/bin: v_i = x_r1 + F * (x_r2 - x_r3)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutants[i] = pop[r1] + self.F * (pop[r2] - pop[r3])
            
            elif self.strategy == 'best1bin':
                # best/1/bin: v_i = x_best + F * (x_r1 - x_r2)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutants[i] = best_vec + self.F * (pop[r1] - pop[r2])
            
            elif self.strategy == 'currenttobest1bin':
                # current-to-best/1/bin: v_i = x_i + F * (x_best - x_i + x_r1 - x_r2)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutants[i] = pop[i] + self.F * (best_vec - pop[i] + pop[r1] - pop[r2])
            
            else:
                # 默认使用 rand/1/bin
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutants[i] = pop[r1] + self.F * (pop[r2] - pop[r3])

        # --- 关键步骤：边界检查 ---
        # 将所有变异后的向量裁剪到定义的边界内
        mutants = np.clip(mutants, self.bounds_low, self.bounds_high)
        
        return mutants
    
    def _crossover_vectorized(self, pop, mutant):
        """向量化的DE交叉操作"""
        n_pop = pop.shape[0]
        
        cross_mask = np.random.rand(n_pop, self.n_dim) < self.CR
        
        rand_dims = np.random.randint(0, self.n_dim, n_pop)
        cross_mask[np.arange(n_pop), rand_dims] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        return trial
    
    def stop(self):
        """停止任务并清理资源"""
        self.is_running = False
        if self.pool:
            try:
                self.log_message.emit("关闭进程池...")
                self.pool.close()
                self.pool.terminate()
                self.pool.join()
                self.pool = None
            except:
                pass

class BaseEditor(QWidget):
    """基础编辑器组件"""
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.widgets = {}
        layout = QFormLayout()
        
        for field in fields:
            if len(field) == 2 and field[1] == "label":
                layout.addRow(QLabel(field[0]))
            else:
                label_text, widget_type, *args = field
                label = QLabel(label_text)
                
                if widget_type == "double":
                    widget = QDoubleSpinBox()
                    widget.setRange(args[0], args[1])
                    widget.setValue(args[2])
                    if len(args) > 3:
                        widget.setSingleStep(args[3])
                    if len(args) > 4:
                        widget.setDecimals(args[4])
                elif widget_type == "int":
                    widget = QSpinBox()
                    widget.setRange(args[0], args[1])
                    widget.setValue(args[2])
                elif widget_type == "text":
                    widget = QLineEdit()
                    if args:
                        widget.setText(args[0])
                elif widget_type == "combo":
                    widget = QComboBox()
                    for item in args:
                        if isinstance(item, tuple) and len(item) == 2:
                            widget.addItem(item[0], item[1])
                        else:
                            widget.addItem(str(item))
                
                layout.addRow(label, widget)
                self.widgets[label_text] = widget
        
        self.setLayout(layout)
    
    def get_value(self, label_text):
        if label_text in self.widgets:
            widget = self.widgets[label_text]
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                return widget.value()
            elif isinstance(widget, QLineEdit):
                return widget.text()
            elif isinstance(widget, QComboBox):
                return widget.currentData() if widget.currentData() else widget.currentText()
        return None

    def set_value(self, label_text, value):
        if label_text in self.widgets:
            widget = self.widgets[label_text]
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)

class FixedParamsEditor(BaseEditor):
    """固定参数编辑器"""
    def __init__(self, parent=None):
        fields = [
            ("电子枪参数", "label"),
            ("电流 I (A):", "double", 0, 5, 0.3, 0.001, 4),
            ("电压 V (V):", "double", 0, 30000, 23000, 100, 0),
            
            ("慢波结构参数", "label"),
            ("每单元损耗参数:", "double", 0, 1e3, 0.0, 0.001, 4),
            ("衰减段索引 (逗号分隔):", "text", "1"),
            ("各段周期数 (逗号分隔):", "text", "20,5,10,10,10,10"),
            
            ("束流参数", "label"),
            ("束流宽度 w (mm):", "double", 0, 10, 0.2, 0.001, 4),
            ("束流厚度 t (mm):", "double", 0, 10, 0.2, 0.001, 4),
            
            ("填充参数", "label"),
            ("填充率倒数 Fn_K:", "double", 1, 10, 1.0, 0.001, 4),
            
            ("输入/衰减参数", "label"),
            ("输入功率 (W):", "double", 0, 100, 0.1, 0.001, 4),
            ("衰减量 (dB):", "double", 0, 100, 20, 1, 1),
            
            ("参数调整系数", "label"),
            ("Vpc 调整系数:", "double", 0, 10, 0.82, 0.001, 4),
            ("Kc 调整系数:", "double", 0, 10, 1.6, 0.001, 4),
            
            ("频点1参数", "label"),
            ("频点1 Kc值 (Ω):", "double", 0, 10, 2.67, 0.001, 4),
            ("频点1 Vpc值 (c):", "double", 0, 1, 0.2867, 0.001, 4),
            ("频点1 频率 (GHz):", "double", 0, 3000, 220.0, 1, 1),
            
            ("频点2参数", "label"),
            ("频点2 Kc值 (Ω):", "double", 0, 10, 2.67, 0.001, 4),
            ("频点2 Vpc值 (c):", "double", 0, 1, 0.2867, 0.001, 4),
            ("频点2 频率 (GHz):", "double", 0, 3000, 225.0, 1, 1),
            
            ("频点3参数", "label"),
            ("频点3 Kc值 (Ω):", "double", 0, 10, 2.67, 0.001, 4),
            ("频点3 Vpc值 (c):", "double", 0, 1, 0.2867, 0.001, 4),
            ("频点3 频率 (GHz):", "double", 0, 3000, 230.0, 1, 1),
        ]
        super().__init__(fields, parent)
    
    def get_params(self):
        try:
            def parse_list(value, dtype=float):
                return [dtype(v.strip()) for v in value.split(',') if v.strip()]
            
            section_seg_idx = parse_list(
                self.get_value("衰减段索引 (逗号分隔):"), int
            )
            n_unit = parse_list(
                self.get_value("各段周期数 (逗号分隔):"), int
            )
            
            n_dim = len(n_unit)
            
            params = {
                "i": self.get_value("电流 I (A):"),
                "v": self.get_value("电压 V (V):"),
                "loss_perunit": self.get_value("每单元损耗参数:"),
                "section_seg_idx": section_seg_idx,
                "n_unit": n_unit,
                "w": self.get_value("束流宽度 w (mm):"),
                "t": self.get_value("束流厚度 t (mm):"),
                "Fn_K": self.get_value("填充率倒数 Fn_K:"),
                "p_in": self.get_value("输入功率 (W):"),
                "loss_attu": self.get_value("衰减量 (dB):"),
                "vpc_coeff": self.get_value("Vpc 调整系数:"),
                "kc_coeff": self.get_value("Kc 调整系数:"),
                "n_dim": n_dim
            }
            
            for i in range(1, 4):
                params[f'freq{i}_kc'] = self.get_value(f"频点{i} Kc值 (Ω):")
                params[f'freq{i}_vpc'] = self.get_value(f"频点{i} Vpc值 (c):")
                params[f'freq{i}_freq'] = self.get_value(f"频点{i} 频率 (GHz):")
            
            return params
            
        except Exception as e:
            QMessageBox.warning(self, "输入错误", f"参数格式无效: {str(e)}")
            return None

class DDEConfigEditor(BaseEditor):
    """DDE配置编辑器"""
    def __init__(self, parent=None):
        fields = [
            ("优化策略:", "combo", *DE_STRATEGIES),
            ("种群大小:", "int", 10, 1000, 100),
            ("最大迭代次数:", "int", 10, 5000, 200),
            ("变异因子 (F):", "double", 0.1, 2.0, 0.5, 0.01, 3),
            ("交叉概率 (CR):", "double", 0.0, 1.0, 0.7, 0.01, 2),
            ("参数下限 (逗号分隔):", "text", "0.499,0.490,0.49,0.49,0.49,0.49"),
            ("参数上限 (逗号分隔):", "text", "0.501,0.510,0.51,0.51,0.51,0.51"),
            ("并行进程数:", "int", 1, MAX_WORKERS, MAX_WORKERS),
        ]
        super().__init__(fields, parent)
    
    def get_config(self, n_dim):
        try:
            lb_text = self.get_value("参数下限 (逗号分隔):")
            ub_text = self.get_value("参数上限 (逗号分隔):")
            
            lb = [float(v.strip()) for v in lb_text.split(",") if v.strip()]
            ub = [float(v.strip()) for v in ub_text.split(",") if v.strip()]
            
            if len(lb) == 1 and n_dim > 1:
                lb = lb * n_dim
            if len(ub) == 1 and n_dim > 1:
                ub = ub * n_dim
            
            if len(lb) != n_dim or len(ub) != n_dim:
                raise ValueError(
                    f"边界维度({len(lb)}, {len(ub)})必须等于参数维度({n_dim})"
                )
            
            bounds = [(low, high) for low, high in zip(lb, ub)]
            
            return {
                "strategy": self.get_value("优化策略:"),
                "pop_size": self.get_value("种群大小:"),
                "max_iter": self.get_value("最大迭代次数:"),
                "F": self.get_value("变异因子 (F):"),
                "CR": self.get_value("交叉概率 (CR):"),
                "bounds": bounds,
                "workers": min(
                    self.get_value("并行进程数:"),
                    MAX_WORKERS
                )
            }
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None

class InitialParamsEditor(BaseEditor):
    """初始参数编辑器"""
    def __init__(self, parent=None):
        fields = [
            ("初始p_SWS值 (逗号分隔):", "text", "0.50,0.50,0.50,0.50,0.50,0.50"),
        ]
        super().__init__(fields, parent)
    
    def get_initial_p(self, n_dim):
        try:
            initial_p_text = self.get_value("初始p_SWS值 (逗号分隔):")
            initial_p = [float(v.strip()) for v in initial_p_text.split(",") if v.strip()]
            
            if len(initial_p) == 1 and n_dim > 1:
                initial_p = initial_p * n_dim
            
            if len(initial_p) != n_dim:
                raise ValueError(
                    f"初始p_SWS维度({len(initial_p)})必须等于参数维度({n_dim})"
                )
            
            return initial_p
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None

class TWTMainWindow(QMainWindow):
    """行波管优化工具主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行波管多频点分布式差分进化优化工具 [性能优化版]")
        self.setGeometry(100, 100, 1200, 800)
        self.optimization_thread = None
        self.optimization_result = None
        self.init_ui()
        self.load_settings()
        
        self.status_bar = self.statusBar()
        self.cpu_info = f"系统CPU核心数: {cpu_count()} | 可用工作进程数: {MAX_WORKERS}"
        self.status_bar.showMessage(f"就绪 [性能优化版] | {self.cpu_info}")
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        tabs = QTabWidget()
        tabs.setFont(QFont("Arial", 10))
        left_layout.addWidget(tabs)
        
        self.fixed_params_editor = FixedParamsEditor()
        self.dde_editor = DDEConfigEditor()
        self.initial_editor = InitialParamsEditor()
        
        tabs.addTab(self.fixed_params_editor, "固定参数")
        tabs.addTab(self.dde_editor, "DDE配置")
        tabs.addTab(self.initial_editor, "初始参数")
        
        control_layout = QHBoxLayout()
        
        self.run_button = QPushButton("开始优化")
        self.run_button.clicked.connect(self.start_optimization)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #a5d6a7;
            }
        """)
        
        self.stop_button = QPushButton("停止优化")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #ffcdd2;
            }
        """)
        
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #bbdefb;
            }
        """)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_ui)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.reset_button)
        
        left_layout.addLayout(control_layout)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        status_group = QGroupBox("优化状态")
        status_layout = QGridLayout(status_group)
        
        self.status_labels = {
            "initial_power": ("初始总功率", "#e0e0e0"),
            "current_power": ("当前总功率", "#e0f7fa"),
            "best_power": ("最佳总功率", "#c8e6c9")
        }
        
        self.status_values = {}
        row = 0
        for key, (label, color) in self.status_labels.items():
            status_layout.addWidget(QLabel(f"{label}:"), row, 0)
            
            value_label = QLabel("—")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet(f"""
                QLabel {{
                    border: 1px solid #b0b0b0;
                    border-radius: 5px;
                    padding: 5px;
                    min-height: 30px;
                    font-weight: bold;
                    background-color: {color};
                    font-size: 11pt;
                }}
            """)
            status_layout.addWidget(value_label, row, 1)
            
            status_layout.addWidget(QLabel("单位 (W):"), row, 2)
            
            unit_label = QLabel("—")
            unit_label.setAlignment(Qt.AlignCenter)
            unit_label.setStyleSheet(f"""
                QLabel {{
                    border: 1px solid #b0b0b0;
                    border-radius: 5px;
                    padding: 5px;
                    min-height: 30px;
                    font-weight: bold;
                    background-color: {color};
                    font-size: 11pt;
                }}
            """)
            status_layout.addWidget(unit_label, row, 3)
            
            self.status_values[key] = (value_label, unit_label)
            row += 1
        
        info_layout = QHBoxLayout()
        
        self.elapsed_label = QLabel("已用时间: —")
        self.iter_label = QLabel("当前迭代: —")
        self.pop_label = QLabel("种群大小: —")
        self.workers_label = QLabel("工作进程: —")
        
        for label in [self.elapsed_label, self.iter_label, self.pop_label, self.workers_label]:
            label.setStyleSheet("font-weight: bold;")
            info_layout.addWidget(label)
        
        status_layout.addLayout(info_layout, row, 0, 1, 4)
        right_layout.addWidget(status_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                text-align: center;
                border: 1px solid #b0b0b0;
                border-radius: 5px;
                height: 25px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        right_layout.addWidget(self.progress_bar)
        
        log_group = QGroupBox("优化日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
            background-color: #f8f8f8;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
        """)
        
        log_layout.addWidget(self.log_area)
        right_layout.addWidget(log_group, 1)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background: white;
                border-radius: 6px;
                margin-top: 5px;
            }
            QTabBar::tab {
                background: #e0e0e0;
                border: 1px solid #c0c0c0;
                padding: 8px 15px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
            }
            QGroupBox {
                border: 1px solid #b0b0b0;
                border-radius: 6px;
                margin-top: 15px;
                font-weight: bold;
                color: #303030;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #f0f0f0;
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                margin-top: -12px;
            }
        """)
    
    def save_settings(self):
        try:
            settings = {
                "fixed_params": self._collect_editor_settings(self.fixed_params_editor),
                "dde_config": self._collect_editor_settings(self.dde_editor),
                "initial_params": self._collect_editor_settings(self.initial_editor)
            }
            
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
                
            self.log_message(f"配置已保存到 {CONFIG_FILE}")
            return True
        except Exception as e:
            self.log_message(f"保存配置失败: {str(e)}")
            return False
    
    def load_settings(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                self._apply_editor_settings(self.fixed_params_editor, settings.get("fixed_params", {}))
                self._apply_editor_settings(self.dde_editor, settings.get("dde_config", {}))
                self._apply_editor_settings(self.initial_editor, settings.get("initial_params", {}))
                
                self.log_message(f"已加载上次保存的配置")
                return True
        except Exception as e:
            self.log_message(f"加载配置失败: {str(e)}")
        return False
    
    def _collect_editor_settings(self, editor):
        settings = {}
        for label in editor.widgets.keys():
            settings[label] = editor.get_value(label)
        return settings
    
    def _apply_editor_settings(self, editor, settings):
        for label, value in settings.items():
            try:
                if value is not None:
                    editor.set_value(label, value)
            except Exception as e:
                self.log_message(f"设置 {label} 失败: {str(e)}")
    
    def closeEvent(self, event):
        if self.save_settings():
            self.log_message("自动保存配置成功")
        
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.optimization_thread.wait()
            
        event.accept()
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
        self.log_area.ensureCursorVisible()
    
    def start_optimization(self):
        fixed_params = self.fixed_params_editor.get_params()
        if fixed_params is None:
            return
        
        n_dim = fixed_params["n_dim"]
        
        dde_config = self.dde_editor.get_config(n_dim)
        if dde_config is None:
            return
        
        initial_p = self.initial_editor.get_initial_p(n_dim)
        if initial_p is None:
            return
        
        freq_params = []
        for i in range(1, 4):
            freq_params.append({
                'Kc': fixed_params[f'freq{i}_kc'],
                'Vpc': fixed_params[f'freq{i}_vpc'],
                'freq': fixed_params[f'freq{i}_freq']
            })
        
        coeffs = {
            'vpc': fixed_params['vpc_coeff'],
            'kc': fixed_params['kc_coeff']
        }
        
        try:
            args = (np.array(initial_p), fixed_params, freq_params, coeffs)
            initial_power = -evaluate_single_wrapper(args)
            
            self.handle_initial_power(initial_power)
            self.log_message(f"初始总功率: {initial_power:.2f} W")
        except Exception as e:
            self.log_message(f"初始功率计算失败: {str(e)}")
            QMessageBox.warning(self, "计算错误", f"初始功率计算失败:\n{str(e)}")
            return
        
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.log_area.clear()
        self.log_area.append("===== 多频点分布式差分进化优化 [性能优化版] =====")
        self.log_area.append(f"参数维度: {n_dim}")
        self.log_area.append(f"初始p_SWS: {initial_p}")
        self.log_area.append(f"初始总功率: {initial_power:.2f} W")
        self.log_area.append(f"DDE配置: 策略={dde_config['strategy']}, "
                            f"种群={dde_config['pop_size']}, "
                            f"迭代={dde_config['max_iter']}, "
                            f"进程={dde_config['workers']}")
        
        for i in range(1, 4):
            self.log_message(f"频点{i}: Kc={fixed_params[f'freq{i}_kc']:.4f}Ω, "
                           f"Vpc={fixed_params[f'freq{i}_vpc']:.4f}c, "
                           f"频率={fixed_params[f'freq{i}_freq']:.1f}GHz")
        
        self.progress_bar.setValue(0)
        
        for key in ["current_power", "best_power"]:
            value_label, unit_label = self.status_values[key]
            value_label.setText("—")
            unit_label.setText("—")
        
        self.elapsed_label.setText("已用时间: —")
        self.iter_label.setText("当前迭代: —")
        self.pop_label.setText(f"种群大小: {dde_config['pop_size']}")
        self.workers_label.setText(f"工作进程: {dde_config['workers']}")
        
        self.start_time = time.time()
        
        self.optimization_thread = DistributedDETask(
            fixed_params=fixed_params,
            freq_params=freq_params,
            coeffs=coeffs,
            bounds=dde_config["bounds"],
            pop_size=dde_config["pop_size"],
            max_iter=dde_config["max_iter"],
            workers=dde_config["workers"],
            strategy=dde_config["strategy"],
            F=dde_config["F"],
            CR=dde_config["CR"],
            initial_p=initial_p
        )
        
        self.optimization_thread.log_message.connect(self.log_message)
        self.optimization_thread.progress_updated.connect(self.handle_progress_update)
        self.optimization_thread.iteration_complete.connect(self.handle_iteration_complete)
        self.optimization_thread.initial_power_calculated.connect(self.handle_initial_power)
        self.optimization_thread.finished.connect(self.optimization_finished)
        
        self.optimization_thread.start()
    
    def handle_initial_power(self, power):
        value_label, unit_label = self.status_values["initial_power"]
        value_label.setText(f"{power:.2f}")
        unit_label.setText("W")
    
    def handle_iteration_complete(self, iteration, power):
        self.iter_label.setText(f"当前迭代: {iteration}")
        value_label, unit_label = self.status_values["current_power"]
        value_label.setText(f"{power:.2f}")
        unit_label.setText("W")
    
    def handle_progress_update(self, progress, best_power):
        self.progress_bar.setValue(progress)
        
        value_label, unit_label = self.status_values["best_power"]
        value_label.setText(f"{best_power:.2f}")
        unit_label.setText("W")
        
        if hasattr(self, 'start_time'):
            elapsed = time.time() - self.start_time
            mins, secs = divmod(elapsed, 60)
            self.elapsed_label.setText(f"已用时间: {int(mins):02d}:{int(secs):02d}")
    
    def stop_optimization(self):
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.log_message("优化已停止")
            self.stop_button.setEnabled(False)
            self.run_button.setEnabled(True)
    
    def optimization_finished(self, result):
        self.optimization_result = result
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        
        if result is None:
            self.log_message("===== 优化失败! =====")
            self.status_bar.showMessage("优化失败! " + self.cpu_info)
            return
        
        self.log_message("\n===== 优化完成! =====")
        self.log_message(f"优化后总功率: {result['best_power']:.2f} W")
        self.log_message(f"优化p_SWS: {result['optimized_p_SWS'].tolist()}")
        
        fixed_params = self.fixed_params_editor.get_params()
        freq_params = []
        for i in range(1, 4):
            freq_params.append({
                'Kc': fixed_params[f'freq{i}_kc'],
                'Vpc': fixed_params[f'freq{i}_vpc'],
                'freq': fixed_params[f'freq{i}_freq']
            })
        
        coeffs = {
            'vpc': fixed_params['vpc_coeff'],
            'kc': fixed_params['kc_coeff']
        }
        
        for i, freq_data in enumerate(freq_params):
            try:
                para_func = partial(adjust_params, coeffs=coeffs)
                
                freq_result = calculate_SEGMENT_TWT_NOLINE(
                    I=fixed_params["i"],
                    V=fixed_params["v"],
                    Kc=freq_data['Kc'],
                    Loss_perunit=fixed_params["loss_perunit"],
                    SectionedSEGMENT_IDX=fixed_params["section_seg_idx"],
                    p_SWS=result['optimized_p_SWS'],
                    N_unit=fixed_params["n_unit"],
                    w=fixed_params["w"],
                    t=fixed_params["t"],
                    Fn_K=fixed_params["Fn_K"],
                    f0_GHz=freq_data['freq'],
                    Vpc=freq_data['Vpc'],
                    para_func=para_func,
                    P_in=fixed_params["p_in"],
                    Loss_attu=fixed_params["loss_attu"]
                )
                self.log_message(f"频点{i+1}功率: {freq_result['输出功率P_out']:.2f} W")
            except Exception as e:
                self.log_message(f"频点{i+1}功率计算失败: {str(e)}")
        
        self.log_message("优化结果已准备就绪，可以保存")
        
        value_label, unit_label = self.status_values["best_power"]
        value_label.setText(f"{result['best_power']:.2f}")
        unit_label.setText("W")
        
        self.save_button.setEnabled(True)
        self.status_bar.showMessage(f"优化完成! 最佳总功率: {result['best_power']:.2f} W | {self.cpu_info}")
    
    def save_results(self):
        if self.optimization_result is None:
            QMessageBox.warning(self, "保存失败", "没有可用的优化结果")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存优化结果", 
            f"TWT_多频点DDE_优化结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
            "文本文件 (*.txt);;所有文件 (*)"
        )
        
        if not path:
            return
        
        if not path.endswith('.txt'):
            path += '.txt'
        
        try:
            fixed_params = self.fixed_params_editor.get_params()
            freq_params = []
            for i in range(1, 4):
                freq_params.append({
                    'Kc': fixed_params[f'freq{i}_kc'],
                    'Vpc': fixed_params[f'freq{i}_vpc'],
                    'freq': fixed_params[f'freq{i}_freq']
                })
            
            coeffs = {
                'vpc': fixed_params['vpc_coeff'],
                'kc': fixed_params['kc_coeff']
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write("===== 行波管多频点分布式差分进化优化结果 [性能优化版] =====\n\n")
                f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"最佳总功率: {self.optimization_result['best_power']:.4f} W\n\n")
                
                for i, freq_data in enumerate(freq_params):
                    f.write(f"频点{i+1}参数:\n")
                    f.write(f"  Kc: {freq_data['Kc']:.6f} Ω\n")
                    f.write(f"  Vpc: {freq_data['Vpc']:.6f} c\n")
                    f.write(f"  频率: {freq_data['freq']:.6f} GHz\n")
                
                f.write("\n优化后的参数值 (mm):\n")
                for i, p in enumerate(self.optimization_result['optimized_p_SWS']):
                    f.write(f"段 {i+1}: {p:.6f}\n")
                
                f.write("\n各频点功率:\n")
                total_power = 0
                for i, freq_data in enumerate(freq_params):
                    try:
                        para_func = partial(adjust_params, coeffs=coeffs)
                        
                        freq_result = calculate_SEGMENT_TWT_NOLINE(
                            I=fixed_params["i"],
                            V=fixed_params["v"],
                            Kc=freq_data['Kc'],
                            Loss_perunit=fixed_params["loss_perunit"],
                            SectionedSEGMENT_IDX=fixed_params["section_seg_idx"],
                            p_SWS=self.optimization_result['optimized_p_SWS'],
                            N_unit=fixed_params["n_unit"],
                            w=fixed_params["w"],
                            t=fixed_params["t"],
                            Fn_K=fixed_params["Fn_K"],
                            f0_GHz=freq_data['freq'],
                            Vpc=freq_data['Vpc'],
                            para_func=para_func,
                            P_in=fixed_params["p_in"],
                            Loss_attu=fixed_params["loss_attu"]
                        )
                        power = freq_result["输出功率P_out"]
                        total_power += power
                        f.write(f"频点{i+1}功率: {power:.4f} W\n")
                    except Exception as e:
                        f.write(f"频点{i+1}功率计算失败: {str(e)}\n")
                
                f.write(f"\n总功率验证: {total_power:.4f} W\n")
                
                p_list = [f"{p:.6f}" for p in self.optimization_result['optimized_p_SWS']]
                f.write("\n复制到代码中的格式:\n")
                f.write(f'optimized_p_SWS = np.array([{", ".join(p_list)}])\n')
            
            self.log_message(f"结果已保存至: {path}")
            QMessageBox.information(self, "保存成功", f"优化结果已保存到:\n{path}")
        except Exception as e:
            self.log_message(f"保存失败: {str(e)}")
            QMessageBox.critical(self, "保存失败", f"保存结果时出错:\n{str(e)}")
    
    def reset_ui(self):
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.stop_optimization()
        
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        for key in self.status_values:
            value_label, unit_label = self.status_values[key]
            value_label.setText("—")
            unit_label.setText("—")
        
        self.progress_bar.setValue(0)
        self.log_area.clear()
        self.status_bar.showMessage(f"已重置 | {self.cpu_info}")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        mp.freeze_support()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TWTMainWindow()
    window.show()
    sys.exit(app.exec_())