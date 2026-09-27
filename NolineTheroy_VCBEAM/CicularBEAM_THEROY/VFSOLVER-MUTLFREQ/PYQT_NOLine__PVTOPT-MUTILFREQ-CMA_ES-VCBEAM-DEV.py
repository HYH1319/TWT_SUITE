import sys
import os
import json
import logging
import time
import traceback
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, shared_memory
from datetime import datetime
from functools import partial
import pickle

import cma
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox,
    QProgressBar, QFileDialog, QGridLayout, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER_MIX_WITH_PVT import calculate_SEGMENT_TWT_NOLINE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_WORKERS = max(1, cpu_count())
CONFIG_FILE = "./configUDF/OPT/twt_CMAES_config_MUTIL.json"

def adjust_params(p_SWS, idx, Vpc, Kc, coeffs):
    """参数调整函数"""
    current_p = p_SWS[idx]
    first_p = p_SWS[0]
    delta = (current_p - first_p) / first_p
    return {
        "Vpc": Vpc + coeffs['vpc'] * delta * Vpc,
        "Kc": Kc + coeffs['kc'] * delta * Kc
    }

class OptimizedEvaluator:
    """优化的评估器，支持预计算和缓存"""
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
        
        # 预计算不变的参数
        self.base_params = {
            'I': fixed_params["i"],
            'V': fixed_params["v"],
            'Loss_perunit': fixed_params["loss_perunit"],
            'SectionedSEGMENT_IDX': fixed_params["section_seg_idx"],
            'N_unit': fixed_params["n_unit"],
            'w': fixed_params["w"],
            't': fixed_params["t"],
            'Fn_K': fixed_params["Fn_K"],
            'P_in': fixed_params["p_in"],
            'Loss_attu': fixed_params["loss_attu"]
        }
    
    def evaluate(self, p_SWS):
        """评估单个参数向量"""
        total_power = 0.0
        
        try:
            for freq_data in self.freq_params:
                para_func = partial(adjust_params, coeffs=self.coeffs)
                
                result = calculate_SEGMENT_TWT_NOLINE(
                    I=self.base_params['I'],
                    V=self.base_params['V'],
                    Kc=freq_data['Kc'],
                    Loss_perunit=self.base_params['Loss_perunit'],
                    SectionedSEGMENT_IDX=self.base_params['SectionedSEGMENT_IDX'],
                    p_SWS=p_SWS,
                    N_unit=self.base_params['N_unit'],
                    w=self.base_params['w'],
                    t=self.base_params['t'],
                    Fn_K=self.base_params['Fn_K'],
                    f0_GHz=freq_data['freq'],
                    Vpc=freq_data['Vpc'],
                    para_func=para_func,
                    P_in=self.base_params['P_in'],
                    Loss_attu=self.base_params['Loss_attu']
                )
                total_power += result["输出功率P_out"]
            
            return -total_power  # 返回负值因为是最小化问题
            
        except Exception as e:
            logger.error(f"评估错误: {str(e)}")
            return float("inf")

def parallel_evaluate_worker(args):
    """并行评估工作进程"""
    candidates, fixed_params_data = args
    
    # 反序列化评估器
    evaluator = pickle.loads(fixed_params_data)
    
    results = []
    for candidate in candidates:
        results.append(evaluator.evaluate(candidate))
    
    return results

class DistributedCMAESTask(QThread):
    """真正高度并行化的CMA-ES优化任务"""
    finished = pyqtSignal(object)
    progress_updated = pyqtSignal(int, float)
    log_message = pyqtSignal(str)
    iteration_complete = pyqtSignal(int, float)
    initial_power_calculated = pyqtSignal(float)
    
    def __init__(self, fixed_params, bounds, pop_size, max_iter, 
                 workers, sigma0=None, initial_p=None):
        super().__init__()
        self.fixed_params = fixed_params
        self.original_bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.workers = workers
        self.sigma0 = sigma0
        self.initial_p = initial_p
        self.is_running = False
        self.pool = None
        self.best_fitness = float('inf')
        self.best_params = None
        self.current_iter = 0
        
        # 处理固定参数
        self.variable_indices = []
        self.fixed_values = {}
        self.bounds = []
        
        for i, (low, high) in enumerate(bounds):
            if abs(low - high) < 1e-10:  # 固定参数
                self.fixed_values[i] = low
            else:
                self.variable_indices.append(i)
                self.bounds.append((low, high))
        
        self.n_dim = len(self.bounds)
        
        if self.n_dim == 0:
            raise ValueError("所有参数都是固定的，无需优化")
        
        # 处理初始参数
        if self.initial_p is not None:
            self.initial_p = np.array([self.initial_p[i] for i in self.variable_indices])
        
        # 自动计算初始标准差
        if self.sigma0 is None:
            range_bounds = np.array([high - low for low, high in self.bounds])
            valid_ranges = range_bounds[range_bounds > 1e-10]
            if len(valid_ranges) > 0:
                self.sigma0 = np.mean(valid_ranges) * 0.2
            else:
                self.sigma0 = 0.01
        
        if self.sigma0 <= 0:
            self.sigma0 = 0.01
        
        # 创建优化的评估器
        self.evaluator = OptimizedEvaluator(fixed_params)
        # 序列化评估器以便传递给工作进程
        self.evaluator_data = pickle.dumps(self.evaluator)
    
    def __del__(self):
        self.stop()
    
    def _reconstruct_full_params(self, variable_params):
        """从可变参数重构完整的参数向量"""
        full_params = np.zeros(len(self.original_bounds))
        
        # 填充固定值
        for i, value in self.fixed_values.items():
            full_params[i] = value
        
        # 填充可变值
        for i, var_idx in enumerate(self.variable_indices):
            full_params[var_idx] = variable_params[i]
        
        return full_params
    
    def _prepare_parallel_batches(self, candidates):
        """准备并行批次"""
        batch_size = max(1, len(candidates) // self.workers)
        batches = []
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            # 重构完整参数
            full_batch = [self._reconstruct_full_params(c) for c in batch]
            batches.append((full_batch, self.evaluator_data))
        
        return batches
    
    def run(self):
        try:
            self.is_running = True
            
            self.log_message.emit(f"创建进程池 ({self.workers} 个进程)...")
            self.pool = Pool(processes=self.workers)
            
            # 初始化CMA-ES
            options = {
                'popsize': self.pop_size,
                'bounds': [np.array([b[0] for b in self.bounds]), 
                          np.array([b[1] for b in self.bounds])],
                'verbose': -1,
                'tolfun': 1e-6,
                'tolx': 1e-6,
                'maxiter': self.max_iter,
                'CMA_elitist': True,  # 启用精英策略
                'CMA_on': True,      # 启用协方差矩阵自适应
            }
            
            # 初始均值
            if self.initial_p is not None:
                initial_mean = self.initial_p
            else:
                initial_mean = np.mean(self.bounds, axis=1)
            
            self.es = cma.CMAEvolutionStrategy(
                initial_mean,
                self.sigma0,
                options
            )
            
            # 计算初始功率
            if self.initial_p is not None:
                full_initial_p = self._reconstruct_full_params(self.initial_p)
                initial_power = -self.evaluator.evaluate(full_initial_p)
                self.initial_power_calculated.emit(initial_power)
                self.log_message.emit(f"使用用户提供的初始参数，初始功率: {initial_power:.2f} W")
            
            self.log_message.emit(f"开始CMA-ES优化，种群大小: {self.pop_size}, 最大迭代: {self.max_iter}")
            self.log_message.emit(f"可变参数维度: {self.n_dim}, 固定参数数量: {len(self.fixed_values)}")
            
            # 主优化循环
            while not self.es.stop() and self.current_iter < self.max_iter and self.is_running:
                self.current_iter += 1
                
                # 生成候选解
                candidates = self.es.ask()
                
                # 边界裁剪
                candidates = np.clip(candidates, 
                                    np.array([b[0] for b in self.bounds]), 
                                    np.array([b[1] for b in self.bounds]))
                
                # 并行评估
                batches = self._prepare_parallel_batches(candidates)
                batch_results = self.pool.map(parallel_evaluate_worker, batches)
                
                # 展平结果
                fitness = np.array([item for sublist in batch_results for item in sublist])
                
                # 更新CMA-ES
                self.es.tell(candidates, fitness)
                
                # 更新最佳解
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_fitness:
                    self.best_fitness = fitness[best_idx]
                    self.best_params = self._reconstruct_full_params(candidates[best_idx]).copy()
                
                # 计算进度
                progress = int(self.current_iter / self.max_iter * 100)
                power_val = -self.best_fitness
                
                # 发出信号
                self.iteration_complete.emit(self.current_iter, power_val)
                self.progress_updated.emit(progress, power_val)
                
                # 详细的日志信息
                mean_fitness = np.mean(fitness)
                std_fitness = np.std(fitness)
                self.log_message.emit(
                    f"迭代 {self.current_iter}/{self.max_iter}, "
                    f"最佳功率: {power_val:.2f} W, "
                    f"平均功率: {-mean_fitness:.2f} W, "
                    f"标准差: {std_fitness:.2f}"
                )
                
                # 每10次迭代输出一次CMA-ES状态
                if self.current_iter % 10 == 0:
                    self.log_message.emit(
                        f"CMA-ES状态: σ={self.es.result[2]:.6f}, "
                        f"条件数={np.linalg.cond(self.es.C):.2f}"
                    )
            
            if self.is_running:
                self.finished.emit({
                    'optimized_p_SWS': self.best_params,
                    'best_power': -self.best_fitness,
                    'iterations': self.current_iter,
                    'final_sigma': self.es.result[2] if hasattr(self.es, 'result') else None
                })
        
        except Exception as e:
            self.log_message.emit(f"CMA-ES优化出错: {str(e)}\n{traceback.format_exc()}")
            self.finished.emit(None)
        finally:
            self.stop()
    
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
            ("电流 I (A):", "double", 0, 1e2, 0.3, 0.001, 4),
            ("电压 V (V):", "double", 0, 100000, 23000, 100, 0),
            
            ("慢波结构参数", "label"),
            ("每单元损耗参数:", "double", 0, 1e2, 0.0, 0.001, 4),
            ("衰减段索引 (逗号分隔):", "text", "1"),
            ("各段周期数 (逗号分隔):", "text", "20,5,10,10,10,10"),
            
            ("束流参数", "label"),
            ("束流宽度 w (mm):", "double", 0, 1e2, 0.2, 0.001, 4),
            ("束流厚度 t (mm):", "double", 0, 1e2, 0.2, 0.001, 4),
            
            ("填充参数", "label"),
            ("填充率倒数 Fn_K:", "double", 1, 1e2, 1.0, 0.001, 4),
            
            ("输入/衰减参数", "label"),
            ("输入功率 (W):", "double", 0, 1e3, 0.1, 0.001, 4),
            ("衰减量 (dB):", "double", 0, 1e3, 20, 1, 1),
            
            ("参数调整系数", "label"),
            ("Vpc 调整系数:", "double", 0, 10, 0.82, 0.001, 4),
            ("Kc 调整系数:", "double", 0, 10, 1.6, 0.001, 4),
            
            ("频点1参数", "label"),
            ("频点1 Kc值 (Ω):", "double", 0, 1e3, 2.67, 0.001, 4),
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

class CMAESConfigEditor(BaseEditor):
    """CMA-ES配置编辑器"""
    def __init__(self, parent=None):
        fields = [
            ("种群大小:", "int", 10, 1000, 100),
            ("最大迭代次数:", "int", 10, 5000, 200),
            ("初始标准差 (σ0):", "double", 0.001, 1.0, 0.01, 0.001, 4),
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
            
            # 检查是否所有参数都是固定的
            all_fixed = all(abs(low - high) < 1e-10 for low, high in bounds)
            if all_fixed:
                raise ValueError("所有参数的上下界都相等，无需优化")
            
            return {
                "pop_size": self.get_value("种群大小:"),
                "max_iter": self.get_value("最大迭代次数:"),
                "sigma0": self.get_value("初始标准差 (σ0):"),
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
        self.setWindowTitle("行波管多频点CMA-ES优化工具 [真正高度并行版]")
        self.setGeometry(100, 100, 1200, 800)
        self.optimization_thread = None
        self.optimization_result = None
        self.init_ui()
        self.load_settings()
        
        self.status_bar = self.statusBar()
        self.cpu_info = f"系统CPU核心数: {cpu_count()} | 可用工作进程数: {MAX_WORKERS}"
        self.status_bar.showMessage(f"就绪 [真正高度并行版] | {self.cpu_info}")
    
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
        self.cmaes_editor = CMAESConfigEditor()
        self.initial_editor = InitialParamsEditor()
        
        tabs.addTab(self.fixed_params_editor, "固定参数")
        tabs.addTab(self.cmaes_editor, "CMA-ES配置")
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
                "cmaes_config": self._collect_editor_settings(self.cmaes_editor),
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
                self._apply_editor_settings(self.cmaes_editor, settings.get("cmaes_config", {}))
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
        
        cmaes_config = self.cmaes_editor.get_config(n_dim)
        if cmaes_config is None:
            return
        
        initial_p = self.initial_editor.get_initial_p(n_dim)
        if initial_p is None:
            return
        
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.log_area.clear()
        self.log_area.append("===== 多频点CMA-ES优化 [真正高度并行版] =====")
        self.log_area.append(f"参数维度: {n_dim}")
        self.log_area.append(f"初始p_SWS: {initial_p}")
        self.log_area.append(f"CMA-ES配置: 种群={cmaes_config['pop_size']}, "
                            f"迭代={cmaes_config['max_iter']}, "
                            f"进程={cmaes_config['workers']}, "
                            f"σ0={cmaes_config['sigma0']}")
        
        # 显示固定参数信息
        fixed_count = sum(1 for low, high in cmaes_config['bounds'] if abs(low - high) < 1e-10)
        variable_count = len(cmaes_config['bounds']) - fixed_count
        if fixed_count > 0:
            self.log_message(f"注意: {fixed_count}个参数固定，{variable_count}个参数可变")
        
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
        self.pop_label.setText(f"种群大小: {cmaes_config['pop_size']}")
        self.workers_label.setText(f"工作进程: {cmaes_config['workers']}")
        
        self.start_time = time.time()
        
        try:
            self.optimization_thread = DistributedCMAESTask(
                fixed_params=fixed_params,
                bounds=cmaes_config["bounds"],
                pop_size=cmaes_config["pop_size"],
                max_iter=cmaes_config["max_iter"],
                workers=cmaes_config["workers"],
                sigma0=cmaes_config["sigma0"],
                initial_p=initial_p
            )
            
            self.optimization_thread.log_message.connect(self.log_message)
            self.optimization_thread.progress_updated.connect(self.handle_progress_update)
            self.optimization_thread.iteration_complete.connect(self.handle_iteration_complete)
            self.optimization_thread.initial_power_calculated.connect(self.handle_initial_power)
            self.optimization_thread.finished.connect(self.optimization_finished)
            
            self.optimization_thread.start()
        except Exception as e:
            self.log_message(f"启动优化失败: {str(e)}")
            QMessageBox.critical(self, "启动失败", f"无法启动优化:\n{str(e)}")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
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
        self.log_message(f"总迭代次数: {result['iterations']}")
        if result['final_sigma'] is not None:
            self.log_message(f"最终σ值: {result['final_sigma']:.6f}")
        
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
            f"TWT_多频点CMAES_优化结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
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
                f.write("===== 行波管多频点CMA-ES优化结果 [真正高度并行版] =====\n\n")
                f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"最佳总功率: {self.optimization_result['best_power']:.4f} W\n")
                f.write(f"总迭代次数: {self.optimization_result['iterations']}\n")
                if self.optimization_result['final_sigma'] is not None:
                    f.write(f"最终σ值: {self.optimization_result['final_sigma']:.6f}\n")
                f.write("\n")
                
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
