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
from sko.PSO import PSO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox,
    QProgressBar, QFileDialog, QGridLayout, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer

from Noline_GAIN_MAINCALL_VSHEETBEAMCORE_MIX_WITH_PVT import calculate_SEGMENT_TWT_NOLINE

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 最大并行进程数（保留1个核心给系统）
MAX_WORKERS = max(1, cpu_count())

# 配置文件名
CONFIG_FILE = "./configUDF/twt_PSO_config.json"

def adjust_params(p_SWS, idx, Vpc, Kc, coeffs):
    """参数调整函数 - 定义为模块级函数用于并行"""
    current_p = p_SWS[idx]
    first_p = p_SWS[0]
    delta = (current_p - first_p) / first_p
    return {
        "Vpc": Vpc + coeffs['vpc'] * delta * Vpc,
        "Kc": Kc + coeffs['kc'] * delta * Kc
    }

class FitnessEvaluator:
    def __init__(self, fixed_params):
        self.fixed_params = fixed_params
        
        # 构建coefficient字典
        self.coeffs = {
            'vpc': fixed_params['vpc_coeff'],
            'kc': fixed_params['kc_coeff']
        }
    
    def __call__(self, p_SWS):
        """适应度评估函数 - 可并行执行"""
        try:
            result = calculate_SEGMENT_TWT_NOLINE(
                I=self.fixed_params["i"],
                V=self.fixed_params["v"],
                Kc=self.fixed_params["kc"],
                Loss_perunit=self.fixed_params["loss_perunit"],
                SectionedSEGMENT_IDX=self.fixed_params["section_seg_idx"],
                p_SWS=p_SWS,
                N_unit=self.fixed_params["n_unit"],
                w=self.fixed_params["w"],
                t=self.fixed_params["t"],
                Fn_K=self.fixed_params["Fn_K"],
                f0_GHz=self.fixed_params["f0_GHz"],
                Vpc=self.fixed_params["vpc"],
                para_func=partial(adjust_params, coeffs=self.coeffs),
                P_in=self.fixed_params["p_in"],
                Loss_attu=self.fixed_params["loss_attu"]
            )
            return -result["输出功率P_out"]  # 负值以实现最大化
        except Exception as e:
            logger.error(f"参数错误: {p_SWS}, 错误: {str(e)}\n{traceback.format_exc()}")
            return float("inf")

class ParallelPSOTask(QThread):
    """并行PSO优化任务"""
    finished = pyqtSignal(object)
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    
    def __init__(self, evaluator, initial_p, lb, ub, pop_size, max_iter, cpus):
        super().__init__()
        self.evaluator = evaluator
        self.initial_p = initial_p
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.cpus = cpus
        self.is_running = False
        self.mutex = QMutex()
    
    def run(self):
        try:
            self.is_running = True
            n_dim = len(self.initial_p)
            
            # 初始化PSO优化器
            pso = PSO(
                func=self.evaluator,
                n_dim=n_dim,
                pop=self.pop_size,
                max_iter=1,  # 每次只运行1次迭代
                lb=self.lb,
                ub=self.ub,
                w=0.8, c1=0.5, c2=0.5
            )
            
            # 创建一个与种群大小相同的初始值数组
            pso.X = np.array([self.initial_p for _ in range(self.pop_size)])
            
            # 创建进程池
            self.pool = Pool(processes=self.cpus)
            
            # 重新定义PSO的cal_y方法以使用并行计算
            original_cal_y = pso.cal_y
            
            def parallel_cal_y():
                try:
                    results = self.pool.map(self.evaluator, pso.X)
                    pso.Y = np.array(results).reshape(-1, 1)
                except Exception as e:
                    logger.error(f"并行计算错误: {str(e)}")
                    # 回退到串行计算
                    original_cal_y()
            
            pso.cal_y = parallel_cal_y
            
            # 开始迭代
            best_power = -float('inf')
            for i in range(self.max_iter):
                if not self.is_running:
                    break
                
                pso.run(1)
                
                # 更新全局最优
                if -pso.gbest_y[0] > best_power:
                    best_power = -pso.gbest_y[0]
                    best_p_SWS = pso.gbest_x
                
                # 发送进度更新
                progress = int((i + 1) / self.max_iter * 100)
                self.progress_updated.emit(progress)
                self.log_message.emit(
                    f"迭代 {i+1}/{self.max_iter} 完成, "
                    f"当前最佳功率: {best_power:.2f} W"
                )
                
                time.sleep(0.01)  # 避免GUI冻结
            
            # 返回结果
            if self.is_running:
                self.finished.emit({
                    'optimized_p_SWS': best_p_SWS,
                    'best_power': best_power
                })
        
        except Exception as e:
            self.log_message.emit(f"并行PSO出错: {str(e)}\n{traceback.format_exc()}")
            self.finished.emit(None)
        finally:
            if hasattr(self, 'pool'):
                self.pool.close()
                self.pool.join()
            self.is_running = False
    
    def stop(self):
        self.is_running = False
        if hasattr(self, 'pool'):
            self.pool.terminate()

class BaseEditor(QWidget):
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        layout = QFormLayout()
        self.widgets = {}
        
        for field in fields:
            if len(field) == 2 and field[1] == "label":
                # 纯标签字段
                label = QLabel(field[0])
                layout.addRow(label)
            else:
                # 输入字段
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
                        widget.addItem(item[0], item[1])
                
                layout.addRow(label, widget)
                self.widgets[label_text] = widget
        
        self.setLayout(layout)
    
    def get_value(self, label_text):
        """获取指定标签对应的控件值"""
        if label_text in self.widgets:
            widget = self.widgets[label_text]
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                return widget.value()
            elif isinstance(widget, QLineEdit):
                return widget.text()
            elif isinstance(widget, QComboBox):
                return widget.currentData()
        return None

    def set_value(self, label_text, value):
        """设置指定标签对应的控件值"""
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

class FixedParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("电子枪参数", "label"),
            ("电流 I (A):", "double", 0, 5, 0.3, 0.001, 4),
            ("电压 V (V):", "double", 0, 30000, 23000, 100, 0),
            
            ("慢波结构参数", "label"),
            ("初始Kc值 (Ω):", "double", 0, 10, 2.67, 0.001, 4),
            ("每单元损耗参数:", "double", 0, 1e3, 0.0, 0.001, 4),
            ("衰减段索引 (逗号分隔):", "text", "1"),
            ("各段周期数 (逗号分隔):", "text", "20,5,10,10,10,10"),
            
            ("束流参数", "label"),
            ("束流宽度 w (mm):", "double", 0, 10, 0.2, 0.001, 4),
            ("束流厚度 t (mm):", "double", 0, 10, 0.2, 0.001, 4),
            
            ("填充参数", "label"),
            ("填充率倒数 Fn_K:", "double", 1, 10, 1.0, 0.001, 4),
            
            ("频率参数", "label"),
            ("工作频率 (GHz):", "double", 0, 300, 220.0, 1, 1),
            
            ("输入/衰减参数", "label"),
            ("输入功率 (W):", "double", 0, 100, 0.1, 0.001, 4),
            ("衰减量 (dB):", "double", 0, 100, 20, 1, 1),
            
            ("相位参数", "label"),
            ("初始Vpc值 (c):", "double", 0, 1, 0.2867, 0.001, 4),
            
            ("参数调整系数", "label"),
            ("Vpc 调整系数:", "double", 0, 10, 0.82, 0.001, 4),
            ("Kc 调整系数:", "double", 0, 10, 1.6, 0.001, 4),
        ]
        super().__init__(fields, parent)
    
    def get_params(self):
        try:
            # 解析列表参数
            def parse_list(value, dtype=float):
                return [dtype(v.strip()) for v in value.split(',') if v.strip()]
            
            # 获取衰减段索引
            section_seg_idx = parse_list(
                self.get_value("衰减段索引 (逗号分隔):"), int
            )
            
            # 获取各段周期数
            n_unit = parse_list(
                self.get_value("各段周期数 (逗号分隔):"), int
            )
            
            # 验证维度匹配
            n_dim = len(n_unit)
            
            return {
                "i": self.get_value("电流 I (A):"),
                "v": self.get_value("电压 V (V):"),
                "kc": self.get_value("初始Kc值 (Ω):"),
                "loss_perunit": self.get_value("每单元损耗参数:"),
                "section_seg_idx": section_seg_idx,
                "n_unit": n_unit,
                "w": self.get_value("束流宽度 w (mm):"),
                "t": self.get_value("束流厚度 t (mm):"),
                "Fn_K": self.get_value("填充率倒数 Fn_K:"),
                "f0_GHz": self.get_value("工作频率 (GHz):"),
                "p_in": self.get_value("输入功率 (W):"),
                "loss_attu": self.get_value("衰减量 (dB):"),
                "vpc": self.get_value("初始Vpc值 (c):"),
                "vpc_coeff": self.get_value("Vpc 调整系数:"),
                "kc_coeff": self.get_value("Kc 调整系数:"),
                "n_dim": n_dim  # 参数维度
            }
        except Exception as e:
            QMessageBox.warning(self, "输入错误", f"参数格式无效: {str(e)}")
            return None

class PSOConfigEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("种群大小:", "int", 10, 100, 20),
            ("最大迭代次数:", "int", 10, 100, 20),
            ("参数下限 (逗号分隔):", "text", "0.499,0.499,0.45,0.45,0.45,0.45"),
            ("参数上限 (逗号分隔):", "text", "0.501,0.501,0.55,0.55,0.55,0.55"),
            ("使用的CPU核心数:", "int", 1, MAX_WORKERS, MAX_WORKERS),
        ]
        super().__init__(fields, parent)
    
    def get_config(self, n_dim):
        try:
            # 解析边界参数
            lb_text = self.get_value("参数下限 (逗号分隔):")
            ub_text = self.get_value("参数上限 (逗号分隔):")
            
            # 解析为浮点数列表
            lb = [float(v.strip()) for v in lb_text.split(",") if v.strip()]
            ub = [float(v.strip()) for v in ub_text.split(",") if v.strip()]
            
            # 如果只提供了一个值，扩展到所有维度
            if len(lb) == 1 and n_dim > 1:
                lb = lb * n_dim
            if len(ub) == 1 and n_dim > 1:
                ub = ub * n_dim
            
            # 验证维度匹配
            if len(lb) != n_dim or len(ub) != n_dim:
                raise ValueError(
                    f"边界维度({len(lb)}, {len(ub)})必须等于参数维度({n_dim})"
                )
            
            return {
                "pop_size": self.get_value("种群大小:"),
                "max_iter": self.get_value("最大迭代次数:"),
                "lb": lb,
                "ub": ub,
                "cpus": min(
                    self.get_value("使用的CPU核心数:"),
                    MAX_WORKERS
                )
            }
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None

class InitialParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("初始p_SWS值 (逗号分隔):", "text", "0.50,0.50,0.50,0.50,0.50,0.50"),
        ]
        super().__init__(fields, parent)
    
    def get_initial_p(self, n_dim):
        try:
            initial_p_text = self.get_value("初始p_SWS值 (逗号分隔):")
            initial_p = [float(v.strip()) for v in initial_p_text.split(",") if v.strip()]
            
            # 如果只提供了一个值，扩展到所有维度
            if len(initial_p) == 1 and n_dim > 1:
                initial_p = initial_p * n_dim
            
            # 验证维度匹配
            if len(initial_p) != n_dim:
                raise ValueError(
                    f"初始p_SWS维度({len(initial_p)})必须等于参数维度({n_dim})"
                )
            
            return initial_p
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None

class ParallelOptimizationThread(QThread):
    """高度并行化的优化线程"""
    finished = pyqtSignal(object)
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    stage_updated = pyqtSignal(str)
    
    def __init__(self, fixed_params, pso_config, initial_p):
        super().__init__()
        self.fixed_params = fixed_params
        self.pso_config = pso_config
        self.initial_p = initial_p
        self.n_dim = fixed_params["n_dim"]
        self.pso_task = None
        self.initial_power = None
        self.is_running = False
    
    def run(self):
        try:
            self.is_running = True
            
            # === 阶段1: 初始计算 ===
            self.stage_updated.emit("initial")
            self.log_message.emit("=== 初始计算阶段 ===")
            self.log_message.emit("计算初始参数性能...")
            
            # 创建适应度评估器
            evaluator = FitnessEvaluator(self.fixed_params)
            
            # 计算初始功率
            self.initial_power = -evaluator(self.initial_p)
            self.log_message.emit(f"初始功率: {self.initial_power:.2f} W")
            
            # === 阶段2: PSO优化 ===
            self.stage_updated.emit("optimization")
            self.log_message.emit("\n=== 优化阶段 ===")
            self.log_message.emit(f"使用并行PSO优化 (进程数: {self.pso_config['cpus']})")
            
            # 创建并行PSO任务
            self.pso_task = ParallelPSOTask(
                evaluator=evaluator,
                initial_p=self.initial_p,
                lb=self.pso_config["lb"],
                ub=self.pso_config["ub"],
                pop_size=self.pso_config["pop_size"],
                max_iter=self.pso_config["max_iter"],
                cpus=self.pso_config["cpus"]
            )
            
            # 连接信号
            self.pso_task.progress_updated.connect(self.progress_updated.emit)
            self.pso_task.log_message.connect(self.log_message.emit)
            self.pso_task.finished.connect(self.handle_pso_finished)
            
            # 启动PSO任务
            self.pso_task.start()
            self.pso_task.wait()
            
        except Exception as e:
            self.log_message.emit(f"优化过程中出错: {str(e)}\n{traceback.format_exc()}")
            self.finished.emit(None)
        finally:
            self.is_running = False
    
    def handle_pso_finished(self, result):
        """处理PSO任务完成"""
        if result is None:
            self.log_message.emit("并行PSO优化失败!")
            self.finished.emit(None)
            return
        
        # === 阶段3: 结果验证 ===
        self.stage_updated.emit("verification")
        self.log_message.emit("\n=== 结果验证阶段 ===")
        self.log_message.emit("验证优化结果...")
        
        try:
            optimized_p_SWS = result['optimized_p_SWS']
            verified_power = result['best_power']
            
            self.log_message.emit(f"优化完成! 最优p_SWS: {optimized_p_SWS.tolist()}")
            self.log_message.emit(f"验证功率: {verified_power:.2f} W")
            
            # 返回最终结果
            self.finished.emit({
                "initial_power": self.initial_power,
                "optimized_p_SWS": optimized_p_SWS,
                "verified_power": verified_power,
                "power_gain": verified_power - self.initial_power
            })
        
        except Exception as e:
            self.log_message.emit(f"结果验证出错: {str(e)}")
            self.finished.emit(None)
    
    def stop(self):
        """停止优化过程"""
        if self.is_running:
            self.log_message.emit("用户请求停止优化...")
            self.is_running = False
            if self.pso_task and self.pso_task.isRunning():
                self.pso_task.stop()
                self.pso_task.wait()

class TWTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行波管参数优化工具 (高度并行化版)")
        self.setGeometry(100, 100, 1000, 800)
        self.optimization_thread = None
        self.optimization_result = None
        self.init_ui()
        
        # 加载上次保存的配置
        self.load_settings()
    
    def init_ui(self):
        # 主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 分割器：左面板（配置），右面板（结果）
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # === 左侧配置面板 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 配置选项卡
        tabs = QTabWidget()
        left_layout.addWidget(tabs)
        
        # 创建参数编辑器
        self.fixed_params_editor = FixedParamsEditor()
        self.pso_editor = PSOConfigEditor()
        self.initial_editor = InitialParamsEditor()
        
        # 添加选项卡
        tabs.addTab(self.fixed_params_editor, "固定参数")
        tabs.addTab(self.pso_editor, "PSO配置")
        tabs.addTab(self.initial_editor, "初始参数")
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.run_button = QPushButton("开始优化")
        self.run_button.clicked.connect(self.start_optimization)
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.stop_button = QPushButton("停止优化")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_button.setEnabled(False)
        
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_ui)
        
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.reset_button)
        
        left_layout.addLayout(control_layout)
        
        # === 右侧结果面板 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # 阶段指示器
        stage_group = QGroupBox("优化阶段")
        stage_layout = QGridLayout(stage_group)
        
        self.stage_labels = {
            'initial': QLabel("初始计算"),
            'optimization': QLabel("并行PSO优化"),
            'verification': QLabel("结果验证")
        }
        
        # 创建状态指示器
        col = 0
        for stage, label in self.stage_labels.items():
            stage_layout.addWidget(QLabel(f"{label.text()}:"), 0, col)
            
            status = QLabel()
            status.setAlignment(Qt.AlignCenter)
            status.setStyleSheet("""
                QLabel {
                    border: 1px solid gray;
                    border-radius: 10px;
                    min-width: 20px;
                    min-height: 20px;
                    background-color: #e0e0e0;
                }
            """)
            self.stage_labels[stage] = status
            stage_layout.addWidget(status, 1, col)
            col += 1
        
        right_layout.addWidget(stage_group)
        
        # 日志区域
        log_group = QGroupBox("优化日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
            background-color: #f5f5f5;
        """)
        
        log_layout.addWidget(self.log_area)
        right_layout.addWidget(log_group, 1)  # 占据更多空间
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(True)
        right_layout.addWidget(self.progress_bar)
        
        # 设置分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.cpu_info = f"系统CPU核心数: {cpu_count()} | 优化进程数: {MAX_WORKERS}"
        self.status_bar.showMessage(f"{self.cpu_info} | 就绪")
        
        # 应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background: white;
                border-radius: 5px;
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
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #f0f0f0;
            }
            QPushButton {
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
                color: #707070;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #c0c0c0;
            }
        """)
    
    def save_settings(self):
        """保存当前配置到文件"""
        try:
            settings = {
                "fixed_params": self._collect_editor_settings(self.fixed_params_editor),
                "pso_config": self._collect_editor_settings(self.pso_editor),
                "initial_params": self._collect_editor_settings(self.initial_editor)
            }
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
                
            self.log_message(f"配置已保存到 {CONFIG_FILE}")
            return True
        except Exception as e:
            self.log_message(f"保存配置失败: {str(e)}")
            return False
    
    def load_settings(self):
        """从文件加载上次保存的配置"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                self._apply_editor_settings(self.fixed_params_editor, settings.get("fixed_params", {}))
                self._apply_editor_settings(self.pso_editor, settings.get("pso_config", {}))
                self._apply_editor_settings(self.initial_editor, settings.get("initial_params", {}))
                
                self.log_message(f"已加载上次保存的配置")
                return True
        except Exception as e:
            self.log_message(f"加载配置失败: {str(e)}")
        return False
    
    def _collect_editor_settings(self, editor):
        """收集编辑器设置"""
        settings = {}
        for label in editor.widgets.keys():
            settings[label] = editor.get_value(label)
        return settings
    
    def _apply_editor_settings(self, editor, settings):
        """应用设置到编辑器"""
        for label, value in settings.items():
            try:
                if value is not None:
                    editor.set_value(label, value)
            except Exception as e:
                self.log_message(f"设置 {label} 失败: {str(e)}")
    
    def closeEvent(self, event):
        """窗口关闭事件 - 自动保存配置"""
        if self.save_settings():
            self.log_message("自动保存配置成功")
        else:
            self.log_message("自动保存配置失败")
        
        # 停止正在运行的优化线程
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.optimization_thread.wait()
            
        event.accept()
    
    def log_message(self, message):
        """添加带时间戳的日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
    
    def start_optimization(self):
        """启动高度并行化的优化过程"""
        # 获取固定参数
        fixed_params = self.fixed_params_editor.get_params()
        if fixed_params is None:
            return
        
        n_dim = fixed_params["n_dim"]
        
        # 获取PSO配置
        pso_config = self.pso_editor.get_config(n_dim)
        if pso_config is None:
            return
        
        # 获取初始参数
        initial_p = self.initial_editor.get_initial_p(n_dim)
        if initial_p is None:
            return
        
        # 准备UI状态
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.log_area.clear()
        self.log_area.append("===== 优化开始 =====")
        self.log_area.append(f"参数维度: {n_dim}")
        self.log_area.append(f"初始p_SWS: {initial_p}")
        self.log_area.append(f"PSO配置: 种群={pso_config['pop_size']}, "
                            f"迭代={pso_config['max_iter']}, "
                            f"进程={pso_config['cpus']}")
        self.progress_bar.setValue(0)
        
        # 重置所有阶段指示器
        for stage in self.stage_labels.values():
            stage.setStyleSheet("background-color: #e0e0e0;")
        
        # 创建并启动优化线程
        self.optimization_thread = ParallelOptimizationThread(
            fixed_params,
            pso_config,
            initial_p
        )
        
        # 连接信号
        self.optimization_thread.log_message.connect(self.log_message)
        self.optimization_thread.progress_updated.connect(self.progress_bar.setValue)
        self.optimization_thread.stage_updated.connect(self.update_stage_indicator)
        self.optimization_thread.finished.connect(self.optimization_finished)
        
        self.optimization_thread.start()
    
    def stop_optimization(self):
        """停止优化过程"""
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.log_message("优化已停止")
            self.stop_button.setEnabled(False)
            self.run_button.setEnabled(True)
            self.reset_button.setEnabled(True)
    
    def update_stage_indicator(self, stage):
        """更新阶段指示器状态"""
        # 重置所有指示器
        for indicator in self.stage_labels.values():
            indicator.setStyleSheet("background-color: #e0e0e0;")
        
        # 高亮当前阶段
        if stage in self.stage_labels:
            self.stage_labels[stage].setStyleSheet("background-color: #4CAF50;")
    
    def optimization_finished(self, result):
        """优化完成处理"""
        self.optimization_result = result
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        
        if result is None:
            self.log_message("===== 优化失败! =====")
            self.status_bar.showMessage("优化失败! " + self.cpu_info)
            return
        
        # 优化成功
        self.log_message("\n===== 优化完成! =====")
        self.log_message(f"初始功率: {result['initial_power']:.2f} W")
        self.log_message(f"优化后功率: {result['verified_power']:.2f} W")
        self.log_message(f"功率提升: {result['power_gain']:.2f} W")
        self.log_message(f"优化p_SWS: {result['optimized_p_SWS'].tolist()}")
        self.log_message("优化结果已准备就绪，可以保存")
        
        self.save_button.setEnabled(True)
        self.status_bar.showMessage(f"优化完成! 功率提升: {result['power_gain']:.2f} W | {self.cpu_info}")
    
    def save_results(self):
        """保存优化结果"""
        if self.optimization_result is None:
            QMessageBox.warning(self, "保存失败", "没有可用的优化结果")
            return
        
        # 获取保存路径
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存优化结果", 
            "TWT优化结果.txt", 
            "文本文件 (*.txt);;所有文件 (*)"
        )
        
        if not path:
            return
        
        # 确保文件后缀
        if not path.endswith('.txt'):
            path += '.txt'
        
        try:
            # 保存结果
            with open(path, 'w') as f:
                f.write("===== 行波管优化结果 =====\n\n")
                f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"初始功率: {self.optimization_result['initial_power']:.4f} W\n")
                f.write(f"优化后功率: {self.optimization_result['verified_power']:.4f} W\n")
                f.write(f"功率提升: {self.optimization_result['power_gain']:.4f} W\n\n")
                
                f.write("优化后的p_SWS值 (mm):\n")
                for i, p in enumerate(self.optimization_result['optimized_p_SWS']):
                    f.write(f"段 {i+1}: {p:.6f}\n")
                
                f.write("\n复制以下内容到您的代码中:\n")
                p_list = [f"{p:.6f}" for p in self.optimization_result['optimized_p_SWS']]
                f.write(f'optimized_p_SWS = [{", ".join(p_list)}]\n')
            
            self.log_message(f"结果已保存至: {path}")
            QMessageBox.information(self, "保存成功", f"优化结果已保存到:\n{path}")
        except Exception as e:
            self.log_message(f"保存失败: {str(e)}")
            QMessageBox.critical(self, "保存失败", f"保存结果时出错:\n{str(e)}")
    
    def reset_ui(self):
        """重置UI状态"""
        # 停止正在运行的优化
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.stop_optimization()
        
        # 重置控件状态
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_area.clear()
        self.log_area.append("界面已重置")
        
        # 重置阶段指示器
        for indicator in self.stage_labels.values():
            indicator.setStyleSheet("background-color: #e0e0e0;")
        
        # 清空结果
        self.optimization_result = None
        
        # 尝试删除配置文件（如果存在）
        if os.path.exists(CONFIG_FILE):
            try:
                os.remove(CONFIG_FILE)
                self.log_message("已清除保存的配置")
            except Exception as e:
                self.log_message(f"清除配置失败: {str(e)}")
        
        # 重置状态栏
        self.status_bar.showMessage(f"已重置 | {self.cpu_info}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TWTMainWindow()
    window.show()
    sys.exit(app.exec_())