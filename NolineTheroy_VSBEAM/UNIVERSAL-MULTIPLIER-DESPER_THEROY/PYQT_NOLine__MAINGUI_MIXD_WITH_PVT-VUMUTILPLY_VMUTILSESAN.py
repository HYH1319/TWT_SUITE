import sys
import os
import json
import csv
import multiprocessing
import time
from datetime import datetime
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QGridLayout, QHeaderView,
    QSplitter, QTextEdit, QProgressBar, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QMutex
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 假设计算函数在以下模块中
from Noline_GAIN_MAINCALL_VU_MULTIPLIER_MUTILSESAN_CORE_MIXED_WITH_PVT import calculate_SEGMENT_TWT_NOLINE_SHEETBEAM

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
CONFIG_DIR = ".\config\MUTILPLYER"
RESULTS_DIR = "Results"
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # 保留1个核心给系统

# 参数配置 - 只包含真正的固定参数
PARAM_CONFIG = [
    ('i', '电流 I (A)', '0.3', '电子枪发射电流'),
    ('v', '电压 V (V)', '23000', '电子枪加速电压'),
    ('p_sws', '周期长度 p_SWS (mm)', '0.50,0.25,0.245,0.150,0.140', '各段的周期长度（逗号分隔）'),
    ('n_unit', '周期数 N_Unit', '30,5,5,5,5', '各段的周期数（逗号分隔）'),
    ('w', '束流宽度 w (mm)', '0.40', '电子束水平宽度'),
    ('t', '束流厚度 t (mm)', '0.20', '电子束垂直厚度'),
    ('fn_k', '填充因子 Fn_K', '2,1.5,1.2', '倍频前、倍频A和倍频B的填充因子（逗号分隔）'),
    ('p_in', '输入功率 P_in (W)', '0.1', '输入信号功率'),
    ('loss_attu', '衰减量 (dB)', '15', '衰减段衰减量'),
    ('section_seg_idx', '衰减段索引', '0', '衰减段索引（0开始）'),
    ('harmonic_sections', '倍频段划分', '1,3', '倍频段起始索引列表（逗号分隔，例如"1,3"表示倍频A从索引1开始，倍频B从索引3开始）'),
    ('vpc_adjust_coef', 'Vpc调整系数', '0.82', 'Vpc随周期变化的调整系数'),
    ('kc_adjust_coef', 'Kc调整系数', '1.6', 'Kc随周期变化的调整系数'),
]

class ParamWidget(QWidget):
    """精简的参数输入组件"""
    def __init__(self, param_id, label, default, tooltip):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label)
        self.line_edit = QLineEdit(default)
        self.line_edit.setMinimumWidth(120)
        self.line_edit.setMaximumWidth(200)
        
        if tooltip:
            self.label.setToolTip(tooltip)
            self.line_edit.setToolTip(tooltip)
        
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
    
    @property
    def value(self):
        return self.line_edit.text().strip()

def calculate_twt_point(params):
    """单个频率点计算函数 - 定义为模块级函数用于并行"""
    fixed, freq_params = params
    
    try:
        # 从固定参数中获取调整系数
        vpc_adjust_coef = float(fixed.get('vpc_adjust_coef', 0.82))  # 默认值0.82
        kc_adjust_coef = float(fixed.get('kc_adjust_coef', 1.6))     # 默认值1.6

        # 调试输出参数信息
        param_str = (
            f"计算参数:\n"
            f"  电流 I: {fixed['i']} A\n"
            f"  电压 V: {fixed['v']} V\n"
            f"  频率: {freq_params['freq']} GHz\n"
            f"  周期长度 p_SWS: {fixed['p_sws']} (mm)\n"
            f"  周期数 N_Unit: {fixed['n_unit']}\n"
            f"  束流宽度 w: {fixed['w']} mm\n"
            f"  束流厚度 t: {fixed['t']} mm\n"
            f"  填充因子 Fn_K: {fixed['fn_k']}\n"
            f"  输入功率 P_in: {fixed['p_in']} W\n"
            f"  衰减量 Loss_attu: {fixed['loss_attu']} dB\n"
            f"  衰减段索引 SectionedSEGMENT_IDX: {fixed['section_seg_idx']}\n"
            f"  倍频段划分: {fixed['harmonic_sections']}\n"
            f"  Vpc调整系数: {vpc_adjust_coef}\n"
            f"  Kc调整系数: {kc_adjust_coef}\n"
        )
        
        # 添加倍频段参数
        param_str += "  倍频段参数:\n"
        kc_list = []
        loss_perunit_list = []
        vpc_list = []
        
        # 按照倍频段顺序提取参数
        for section_name in fixed['harmonic_sections_names']:
            section_params = freq_params['sections'].get(section_name, {})
            kc = float(section_params.get('kc', 0))
            loss_perunit = float(section_params.get('loss_perunit', 0))
            vpc = float(section_params.get('vpc', 0))
            
            kc_list.append(kc)
            loss_perunit_list.append(loss_perunit)
            vpc_list.append(vpc)
            
            param_str += (
                f"    {section_name}:\n"
                f"      Kc: {kc} Ω\n"
                f"      Loss_perunit: {loss_perunit}\n"
                f"      Vpc: {vpc} c\n"
            )
        
        # 调用核心计算函数
        result = calculate_SEGMENT_TWT_NOLINE_SHEETBEAM(
            I=float(fixed['i']),
            V=float(fixed['v']),
            Kc=kc_list,
            Loss_perunit=loss_perunit_list,
            SectionedSEGMENT_IDX=[int(idx) for idx in fixed['section_seg_idx']],
            p_SWS=[float(p) for p in fixed['p_sws']],
            N_unit=[int(n) for n in fixed['n_unit']],
            w=float(fixed['w']),
            t=float(fixed['t']),
            Fn_K=[float(fk) for fk in fixed['fn_k']],
            f0_GHz=float(freq_params['freq']),
            Vpc=vpc_list,
            P_in=float(fixed['p_in']),
            Loss_attu=float(fixed['loss_attu']),
            harmonic_sections=[int(hs) for hs in fixed['harmonic_sections']],
            Vpc_adjust_coef=vpc_adjust_coef,
            Kc_adjust_coef=kc_adjust_coef
        )
        
        # 返回结果包含参数信息用于调试
        return {
            'freq': float(freq_params['freq']),
            'power': result['输出功率P_out'],
            'param_str': param_str
        }
        
    except Exception as e:
        # 返回错误信息包含参数详情
        return {
            'freq': float(freq_params['freq']),
            'error': f"计算错误: {str(e)}",
            'param_str': param_str
        }

class TWTCalculator(QMainWindow):
    """优化后的主窗口 - 专注于核心功能"""
    def __init__(self):
        super().__init__()
        self.history = []  # 当前计算结果
        self.history_results = []  # 所有计算结果 (用于绘图)
        self.plot_counter = 1  # 绘图曲线计数器
        self.pool = None   # 进程池对象
        self.timer = None  # 进度检查定时器
        self.mutex = QMutex()  # 线程安全锁
        self.param_widgets = {}  # 参数控件字典
        self.section_names = ["倍频前"]  # 默认倍频段名称
        self.init_ui()
        self.load_last_config()
    
    def init_ui(self):
        """紧凑的UI初始化"""
        self.setWindowTitle('TWT倍频行波管计算器 (多核并行版)')
        self.setGeometry(100, 100, 1400, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # === 左侧面板 (参数设置) ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 参数网格
        param_grid = QGroupBox("固定参数")
        grid_layout = QGridLayout(param_grid)
        
        self.param_widgets = {}
        for i, (pid, label, default, tooltip) in enumerate(PARAM_CONFIG):
            widget = ParamWidget(pid, label, default, tooltip)
            self.param_widgets[pid] = widget
            row = i // 2
            col = i % 2
            grid_layout.addWidget(widget, row, col)
        
        left_layout.addWidget(param_grid)
        
        # 生成表格按钮
        self.generate_table_btn = QPushButton("生成参数表格")
        self.generate_table_btn.clicked.connect(self.generate_param_table)
        self.generate_table_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        left_layout.addWidget(self.generate_table_btn)
        
        # 表格区域
        table_group = QGroupBox("频率相关参数配置")
        table_layout = QVBoxLayout(table_group)
        
        self.param_table = QTableWidget(0, 0)
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.param_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.param_table.verticalHeader().setDefaultSectionSize(35)
        
        # 表格操作按钮
        table_btn_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("添加行")
        self.add_row_btn.clicked.connect(self.add_table_row)
        self.del_row_btn = QPushButton("删除行")
        self.del_row_btn.clicked.connect(self.delete_table_row)
        self.clear_table_btn = QPushButton("清空表格")
        self.clear_table_btn.clicked.connect(self.clear_table)
        
        table_btn_layout.addWidget(self.add_row_btn)
        table_btn_layout.addWidget(self.del_row_btn)
        table_btn_layout.addWidget(self.clear_table_btn)
        
        table_layout.addLayout(table_btn_layout)
        table_layout.addWidget(self.param_table)
        
        # 导入按钮
        import_layout = QHBoxLayout()
        self.import_multiple_btn = QPushButton("批量导入CSV")
        self.import_multiple_btn.clicked.connect(self.import_multiple_csv)
        self.import_multiple_btn.setToolTip("按倍频段顺序导入多个CSV文件")
        
        import_layout.addWidget(self.import_multiple_btn)
        table_layout.addLayout(import_layout)
        
        left_layout.addWidget(table_group)
        
        # 操作按钮组
        control_group = QGroupBox("计算控制")
        control_layout = QGridLayout(control_group)
        
        self.calc_btn = QPushButton("开始计算")
        self.calc_btn.clicked.connect(self.start_calculation)
        self.calc_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        self.export_btn = QPushButton("导出数据")
        self.export_btn.clicked.connect(self.export_data)
        self.clear_plot_btn = QPushButton("清空绘图")
        self.clear_plot_btn.clicked.connect(self.clear_plot)
        
        control_layout.addWidget(self.calc_btn, 0, 0)
        control_layout.addWidget(self.save_btn, 0, 1)
        control_layout.addWidget(self.export_btn, 1, 0)
        control_layout.addWidget(self.clear_plot_btn, 1, 1)
        
        left_layout.addWidget(control_group)
        
        # 系统信息
        info_group = QGroupBox("系统状态")
        info_layout = QVBoxLayout(info_group)
        
        self.core_info = QLabel(f"检测到 {MAX_WORKERS} 个CPU核心可用")
        self.status_label = QLabel("就绪")
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        
        info_layout.addWidget(self.core_info)
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.progress)
        
        left_layout.addWidget(info_group)
        
        main_layout.addWidget(left_panel, 35)  # 35%宽度
        
        # === 右侧面板 (结果展示) ===
        splitter = QSplitter(Qt.Vertical)
        
        # 绘图区域
        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setup_plot()
        
        plot_layout.addWidget(self.canvas)
        
        # 文本结果区域
        result_frame = QWidget()
        result_layout = QVBoxLayout(result_frame)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("计算结果将显示在此处...")
        self.result_text.setStyleSheet("font-family: Consolas, 'Microsoft YaHei';")
        
        result_layout.addWidget(QLabel("计算结果与日志:"))
        result_layout.addWidget(self.result_text)
        
        splitter.addWidget(plot_frame)
        splitter.addWidget(result_frame)
        splitter.setSizes([500, 300])
        
        main_layout.addWidget(splitter, 65)  # 65%宽度
        
        # 初始化表格
        self.generate_param_table()
    
    def setup_plot(self):
        """初始化图表设置"""
        self.ax.clear()
        self.ax.set_title("输出功率 vs 频率", fontsize=14)
        self.ax.set_xlabel("频率 (GHz)", fontsize=12)
        self.ax.set_ylabel("输出功率 (W)", fontsize=12)
        self.ax.grid(True)
        self.canvas.draw()
    
    def generate_param_table(self):
        """根据倍频段划分生成参数表格"""
        # 获取倍频段划分
        harmonic_sections = self.param_widgets['harmonic_sections'].value
        if not harmonic_sections:
            QMessageBox.warning(self, "参数错误", "请先输入倍频段划分")
            return
            
        try:
            # 解析倍频段划分
            sections = [int(s.strip()) for s in harmonic_sections.split(',') if s.strip()]
            if not sections:
                raise ValueError("倍频段划分不能为空")
                
            # 确定倍频段数量
            num_sections = len(sections) + 1
            
            # 生成倍频段名称
            self.section_names = ["倍频前"]
            for i in range(len(sections)):
                self.section_names.append(f"倍频{chr(65+i)}")  # A, B, C, ...
            
            # 设置表格列数 (频率列 + 每个倍频段的3个参数列)
            num_columns = 1 + num_sections * 3
            self.param_table.setColumnCount(num_columns)
            
            # 设置表头
            headers = ["频率 (GHz)"]
            for section_name in self.section_names:
                headers.append(f"{section_name} Kc (Ω)")
                headers.append(f"{section_name} Loss_perunit")
                headers.append(f"{section_name} Vpc (c)")
            
            self.param_table.setHorizontalHeaderLabels(headers)
            
            # 添加默认行
            self.add_table_row()
            
            self.log_message(f"已生成表格结构: {len(self.section_names)}个倍频段")
        except Exception as e:
            QMessageBox.warning(self, "参数错误", f"倍频段划分格式无效: {str(e)}")
            self.log_message(f"生成表格时出错: {str(e)}")
    
    # =================== 表格操作 ===================
    def add_table_row(self):
        """添加行到表格"""
        row = self.param_table.rowCount()
        self.param_table.insertRow(row)
        
        # 设置默认值
        for col in range(self.param_table.columnCount()):
            # 频率列
            if col == 0:
                item = QTableWidgetItem("211")
            # Kc列
            elif col % 3 == 1:
                item = QTableWidgetItem("3.6")
            # Loss_perunit列
            elif col % 3 == 2:
                item = QTableWidgetItem("0")
            # Vpc列
            else:
                item = QTableWidgetItem("0.288")
                
            self.param_table.setItem(row, col, item)
    
    def delete_table_row(self):
        """删除表格当前行"""
        if self.param_table.rowCount() > 0:
            current_row = self.param_table.currentRow()
            row_to_delete = current_row if current_row >= 0 else self.param_table.rowCount() - 1
            self.param_table.removeRow(row_to_delete)
    
    def clear_table(self):
        """清空表格"""
        if self.param_table.rowCount() > 0:
            self.param_table.setRowCount(0)
            self.add_table_row()
    
    def import_multiple_csv(self):
        """批量导入多个CSV文件 - 按倍频段顺序填充"""
        if not self.section_names:
            QMessageBox.warning(self, "导入错误", "请先生成表格结构")
            return
            
        # 获取倍频段数量
        num_sections = len(self.section_names)
        
        # 打开文件选择对话框 - 允许多选
        paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "批量导入CSV文件", 
            "", 
            "CSV文件 (*.csv)"
        )
        
        if not paths:
            return
            
        # 确保有足够的CSV文件
        if len(paths) < num_sections:
            QMessageBox.warning(self, "导入错误", f"需要至少{num_sections}个CSV文件，每个对应一个倍频段")
            return
            
        # 清空表格
        self.param_table.setRowCount(0)
        
        # 按倍频段顺序处理每个CSV文件
        section_rows = {}
        for section_idx, path in enumerate(paths[:num_sections]):  # 只处理前num_sections个文件
            section_name = self.section_names[section_idx]
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if not rows:
                        self.log_message(f"导入文件为空: {path}")
                        continue
                        
                    # 检查是否有表头
                    has_header = any("频率" in cell for cell in rows[0])
                    if has_header:
                        start_row = 1  # 跳过表头
                        self.log_message(f"检测到CSV表头: {path}")
                    else:
                        start_row = 0
                        self.log_message(f"未检测到表头: {path}")
                    
                    # 处理数据行
                    section_rows[section_idx] = []
                    for row_idx, row_data in enumerate(rows[start_row:]):
                        if not row_data or all(not cell.strip() for cell in row_data):
                            continue  # 跳过空行
                            
                        # 只取频率列（第一列）
                        if len(row_data) > 0:
                            freq_value = row_data[0].strip()
                            section_rows[section_idx].append(freq_value)
                    
                    self.log_message(f"从 {path} 导入 {len(section_rows[section_idx])} 行数据 ({section_name})")
            except Exception as e:
                self.log_message(f"导入CSV文件时出错 ({path}): {str(e)}")
                QMessageBox.critical(self, "导入错误", f"导入文件 {path} 时出错:\n{str(e)}")
        
        # 确定最大行数
        max_rows = max(len(rows) for rows in section_rows.values()) if section_rows else 0
        
        # 填充表格
        for row_idx in range(max_rows):
            row = self.param_table.rowCount()
            self.param_table.insertRow(row)
            
            # 设置频率列
            freq_value = ""
            for section_idx in range(num_sections):
                if row_idx < len(section_rows.get(section_idx, [])):
                    freq_value = section_rows[section_idx][row_idx]
                    break
            self.param_table.setItem(row, 0, QTableWidgetItem(freq_value))
            
            # 设置倍频段参数
            for section_idx in range(num_sections):
                # 计算列索引
                kc_col = 1 + section_idx * 3
                loss_col = kc_col + 1
                vpc_col = loss_col + 1
                
                # 从CSV文件中获取参数值（如果有）
                if row_idx < len(section_rows.get(section_idx, [])):
                    # 获取该行的所有数据
                    try:
                        with open(paths[section_idx], 'r', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            rows = list(reader)
                            if not rows:
                                continue
                                
                            # 检查是否有表头
                            has_header = any("频率" in cell for cell in rows[0])
                            start_row = 1 if has_header else 0
                            
                            # 获取该行数据
                            if row_idx + start_row < len(rows):
                                row_data = rows[row_idx + start_row]
                                if len(row_data) >= 4:  # 至少包含频率和3个参数
                                    # Kc列
                                    if len(row_data) > 1:
                                        self.param_table.setItem(row, kc_col, QTableWidgetItem(row_data[1].strip()))
                                    # Loss_perunit列
                                    if len(row_data) > 2:
                                        self.param_table.setItem(row, loss_col, QTableWidgetItem(row_data[2].strip()))
                                    # Vpc列
                                    if len(row_data) > 3:
                                        self.param_table.setItem(row, vpc_col, QTableWidgetItem(row_data[3].strip()))
                    except:
                        pass
        
        self.log_message(f"已按倍频段顺序填充表格，共 {max_rows} 行数据")
    
    # =================== 核心计算功能 ===================
    def parse_params(self):
        """解析固定和可变参数"""
        # 解析固定参数
        fixed_params = {}
        for pid, widget in self.param_widgets.items():
            value = widget.value
            
            # 处理列表参数
            if pid in ['p_sws', 'n_unit', 'fn_k', 'section_seg_idx', 'harmonic_sections']:
                try:
                    # 分割字符串为列表
                    values = [v.strip() for v in value.split(',') if v.strip()]
                    if not values:
                        raise ValueError("参数不能为空")
                    
                    fixed_params[pid] = values
                except Exception as e:
                    QMessageBox.warning(self, "参数错误", f"{pid} 参数格式无效: {e}")
                    return None, None
            else:  # 数值参数
                try:
                    # 尝试转换为浮点数
                    fixed_params[pid] = float(value)
                except ValueError:
                    # 如果不能转换为浮点数，保持为字符串
                    fixed_params[pid] = value
        
        # 验证固定参数
        required_lists = ['p_sws', 'n_unit', 'fn_k', 'section_seg_idx', 'harmonic_sections']
        for param in required_lists:
            if not fixed_params.get(param) or len(fixed_params[param]) < 1:
                QMessageBox.warning(self, "参数错误", f"{param} 需要至少1个值")
                return None, None
            
        # 确保倍频段划分是整数列表
        try:
            fixed_params['harmonic_sections'] = [int(v) for v in fixed_params['harmonic_sections']]
        except:
            QMessageBox.warning(self, "参数错误", "倍频段划分应为整数列表")
            return None, None
        
        # 添加倍频段名称到固定参数
        fixed_params['harmonic_sections_names'] = self.section_names
        
        # 解析表格参数
        freq_params = []
        for row in range(self.param_table.rowCount()):
            try:
                # 获取频率值
                freq_item = self.param_table.item(row, 0)
                if not freq_item or not freq_item.text():
                    continue
                    
                freq_value = freq_item.text().strip()
                
                # 解析倍频段参数
                sections_params = {}
                for section_idx, section_name in enumerate(self.section_names):
                    # 计算列索引
                    kc_col = 1 + section_idx * 3
                    loss_col = kc_col + 1
                    vpc_col = loss_col + 1
                    
                    # 获取参数值
                    kc_item = self.param_table.item(row, kc_col)
                    loss_item = self.param_table.item(row, loss_col)
                    vpc_item = self.param_table.item(row, vpc_col)
                    
                    if kc_item and loss_item and vpc_item:
                        sections_params[section_name] = {
                            'kc': kc_item.text().strip(),
                            'loss_perunit': loss_item.text().strip(),
                            'vpc': vpc_item.text().strip()
                        }
                
                # 添加到参数列表
                freq_params.append({
                    'freq': freq_value,
                    'sections': sections_params
                })
            except (ValueError, AttributeError) as e:
                self.log_message(f"跳过无效行 {row+1}: {str(e)}")
                continue
        
        if not freq_params:
            QMessageBox.warning(self, "计算错误", "没有可计算的频率点")
            return None, None
        
        return fixed_params, freq_params
        
    def start_calculation(self):
        """启动并行计算"""
        fixed, freq_params = self.parse_params()
        if fixed is None or freq_params is None:
            return
            
        self.log_message("= 计算开始 =")
        self.log_message(f"使用 {MAX_WORKERS} 个进程并行计算")
        self.log_message(f"共 {len(freq_params)} 个频率点需要计算")
        
        # 创建任务列表 - 每个任务是一个元组 (fixed, freq_params)
        tasks = [(fixed, fp) for fp in freq_params]
        total_tasks = len(tasks)
        
        # 更新UI状态
        self.calc_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("计算中...")
        self.history = []  # 重置当前历史记录
        
        # 创建进程池
        try:
            self.pool = multiprocessing.Pool(processes=min(MAX_WORKERS, total_tasks))
        except Exception as e:
            self.log_message(f"创建进程池失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法启动并行计算:\n{str(e)}")
            self.reset_calculation_state()
            return
        
        # 异步计算结果 - 使用模块级函数
        self.results = self.pool.imap_unordered(calculate_twt_point, tasks)
        
        # 创建定时器检查进度
        self.timer = QTimer()
        self.timer.timeout.connect(partial(self.check_results, total_tasks))
        self.timer.start(10)  # 每0.3秒检查一次
        self.start_time = time.time()
        self.completed = 0
        self.success_count = 0
        self.error_count = 0
    
    def reset_calculation_state(self):
        """重置计算状态"""
        self.calc_btn.setEnabled(True)
        self.status_label.setText("计算失败")
        self.progress.setValue(0)
    
    def check_results(self, total_tasks):
        """定时检查计算结果"""
        # 使用非阻塞锁，避免死锁
        if not self.mutex.tryLock():
            return
            
        try:
            if self.results and self.completed < total_tasks:
                try:
                    # 尝试获取结果
                    result = next(self.results)
                    if result is None:
                        return
                        
                    self.completed += 1
                    
                    # 记录参数到日志
                    if 'param_str' in result:
                        self.log_message(f"开始计算频率: {result['freq']} GHz")
                        self.log_message(result['param_str'])
                    
                    # 处理结果
                    if 'error' in result:
                        self.error_count += 1
                        self.log_message(f"频率 {result['freq']} GHz: {result['error']}")
                    elif 'power' in result:
                        self.success_count += 1
                        self.history.append((result['freq'], result['power']))
                        self.log_message(f"频率 {result['freq']} GHz: 输出功率 {result['power']:.2f} W")
                    
                    # 更新进度
                    progress = int(self.completed * 100 / total_tasks)
                    elapsed = time.time() - self.start_time
                    if elapsed > 1:  # 至少3秒后计算剩余时间
                        remaining = (elapsed / max(self.completed, 1)) * (total_tasks - self.completed)
                        time_msg = f"剩余: {int(remaining)}秒"
                    else:
                        time_msg = "预估中..."
                    
                    self.status_label.setText(f"进度: {self.completed}/{total_tasks} {time_msg}")
                    self.progress.setValue(progress)
                    
                    # 所有任务完成
                    if self.completed >= total_tasks:
                        self.finish_calculation()
                        return
                except StopIteration:
                    self.finish_calculation()
                except Exception as e:
                    self.log_message(f"处理结果时出错: {str(e)}")
                    self.finish_calculation()
        finally:
            self.mutex.unlock()  # 确保总是释放锁
    
    def finish_calculation(self):
        """完成计算清理工作"""
        try:
            if self.timer:
                self.timer.stop()
                self.timer = None
            
            if self.pool:
                try:
                    self.pool.terminate()
                    self.pool.close()
                except:
                    pass
                finally:
                    self.pool = None
            
            elapsed = time.time() - self.start_time
            self.log_message(f"计算完成! 成功: {self.success_count}, 失败: {self.error_count}")
            self.log_message(f"总耗时: {elapsed:.1f} 秒")
            self.status_label.setText("计算完成")
            self.progress.setValue(100)
            
            # 保存当前结果
            if self.success_count > 0:
                self.save_results()
                
                # 按频率排序并添加到历史结果
                sorted_history = sorted(self.history, key=lambda x: x[0])
                freqs = [h[0] for h in sorted_history]
                powers = [h[1] for h in sorted_history]
                self.history_results.append((freqs, powers, f"参数组{self.plot_counter}"))
                self.plot_counter += 1
                
                # 更新图表
                self.update_plot()
        finally:
            # 确保UI状态恢复
            self.calc_btn.setEnabled(True)
    
    def update_plot(self):
        """按您要求的方式更新图表 - 保留所有结果曲线"""
        self.ax.clear()
        self.ax.set_title("输出功率 vs 频率", fontsize=14)
        self.ax.set_xlabel("频率 (GHz)", fontsize=12)
        self.ax.set_ylabel("输出功率 (W)", fontsize=12)
        self.ax.grid(True)
        
        # 如果没有历史结果，则绘制空图
        if not self.history_results:
            self.canvas.draw()
            return
            
        # 计算全局范围
        all_freqs = []
        all_powers = []
        
        # 遍历所有历史结果并绘制
        for i, (freqs, powers, label) in enumerate(self.history_results):
            # 将频率和功率组合成点并排序
            points = sorted(zip(freqs, powers), key=lambda x: x[0])
            sorted_freqs = [p[0] for p in points]
            sorted_powers = [p[1] for p in points]
            
            # 添加到全局范围
            all_freqs.extend(sorted_freqs)
            all_powers.extend(sorted_powers)
            
            # 绘制曲线
            self.ax.plot(sorted_freqs, sorted_powers, 'o-', markersize=5, label=label)
        
        # 设置图例
        self.ax.legend(loc='best')
        
        # 设置合适的数据范围
        if all_freqs and all_powers:
            min_freq = min(all_freqs) * 0.98
            max_freq = max(all_freqs) * 1.02
            min_power = min(all_powers) * 0.95
            max_power = max(all_powers) * 1.05 if max(all_powers) > 0 else max(all_powers) * 1.15
            self.ax.set_xlim(min_freq, max_freq)
            self.ax.set_ylim(min_power, max_power)
        
        self.canvas.draw()
    
    def clear_plot(self):
        """清空所有历史结果和图表"""
        self.history_results = []
        self.plot_counter = 1
        self.ax.clear()
        self.ax.set_title("输出功率 vs 频率", fontsize=14)
        self.ax.set_xlabel("频率 (GHz)", fontsize=12)
        self.ax.set_ylabel("输出功率 (W)", fontsize=12)
        self.ax.grid(True)
        self.canvas.draw()
        self.log_message("已清空所有绘图结果")
    
    # =================== 数据管理 ===================
    def log_message(self, message):
        """添加带时间戳的日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.result_text.append(f"[{timestamp}] {message}")
        # 自动滚动到底部
        self.result_text.verticalScrollBar().setValue(
            self.result_text.verticalScrollBar().maximum())
    
    def save_results(self):
        """保存计算结果到文件"""
        if not self.history:
            self.log_message("没有有效结果可以保存")
            return
            
        fixed, _ = self.parse_params()
        if not fixed:
            return
            
        # 确保结果目录存在
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(RESULTS_DIR, f"TWT_Result_{timestamp}.csv")
        
        try:
            # 创建CSV文件
            with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                # 写入固定参数头
                writer.writerow(["固定参数"])
                for key, value in fixed.items():
                    if isinstance(value, list):
                        writer.writerow([key, ','.join(map(str, value))])
                    else:
                        writer.writerow([key, value])
                
                # 写入空行分隔
                writer.writerow([])
                
                # 写入结果头
                writer.writerow(["频率 (GHz)", "输出功率 (W)"])
                
                # 按频率排序结果
                sorted_history = sorted(self.history, key=lambda x: x[0])
                
                for freq, power in sorted_history:
                    writer.writerow([freq, f"{power:.4f}"])
            
            self.log_message(f"结果已保存至: {save_path}")
        except Exception as e:
            self.log_message(f"保存结果时出错: {str(e)}")
    
    def save_config(self):
        """保存当前配置"""
        # 确保配置目录存在
        os.makedirs(CONFIG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CONFIG_DIR, f"TWT_Config_{timestamp}.json")
        
        # 准备配置数据
        config = {'fixed': {}, 'param_table': []}
        
        try:
            # 保存固定参数
            for pid, widget in self.param_widgets.items():
                config['fixed'][pid] = widget.value
            
            # 保存参数表格数据
            for row in range(self.param_table.rowCount()):
                row_data = []
                for col in range(self.param_table.columnCount()):
                    item = self.param_table.item(row, col)
                    if item and item.text():
                        row_data.append(item.text().strip())
                    else:
                        row_data.append("")
                config['param_table'].append(row_data)
            
            # 保存JSON
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.log_message(f"配置已保存至: {save_path}")
            QMessageBox.information(self, "保存成功", "当前配置已成功保存")
        except Exception as e:
            self.log_message(f"保存配置时出错: {str(e)}")
            QMessageBox.critical(self, "保存错误", f"保存配置失败:\n{str(e)}")
    
    def load_last_config(self):
        """加载最近配置"""
        try:
            if not os.path.exists(CONFIG_DIR):
                os.makedirs(CONFIG_DIR)
                return
                
            # 查找最新配置文件
            files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
            if not files:
                return
                
            # 按修改时间排序
            files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(CONFIG_DIR, f)), reverse=True)
            latest = files[0]
            config_path = os.path.join(CONFIG_DIR, latest)
            
            # 加载配置
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # 1. 加载固定参数
                for pid, value in config.get('fixed', {}).items():
                    if pid in self.param_widgets:
                        self.param_widgets[pid].line_edit.setText(str(value))
                
                # 2. 重新生成表格结构（基于加载的倍频段划分）
                # 获取倍频段划分
                harmonic_sections = config.get('fixed', {}).get('harmonic_sections', '')
                if harmonic_sections:
                    try:
                        # 解析倍频段划分
                        sections = [int(s.strip()) for s in harmonic_sections.split(',') if s.strip()]
                        if not sections:
                            raise ValueError("倍频段划分不能为空")
                            
                        # 确定倍频段数量
                        num_sections = len(sections) + 1
                        
                        # 生成倍频段名称
                        self.section_names = ["倍频前"]
                        for i in range(len(sections)):
                            self.section_names.append(f"倍频{chr(65+i)}")  # A, B, C, ...
                        
                        # 设置表格列数 (频率列 + 每个倍频段的3个参数列)
                        num_columns = 1 + num_sections * 3
                        self.param_table.setColumnCount(num_columns)
                        
                        # 设置表头
                        headers = ["频率 (GHz)"]
                        for section_name in self.section_names:
                            headers.append(f"{section_name} Kc (Ω)")
                            headers.append(f"{section_name} Loss_perunit")
                            headers.append(f"{section_name} Vpc (c)")
                        
                        self.param_table.setHorizontalHeaderLabels(headers)
                        
                        # 清空表格
                        self.param_table.setRowCount(0)
                        
                        self.log_message(f"已重新生成表格结构: {len(self.section_names)}个倍频段")
                    except Exception as e:
                        QMessageBox.warning(self, "参数错误", f"倍频段划分格式无效: {str(e)}")
                        self.log_message(f"生成表格时出错: {str(e)}")
                else:
                    # 如果没有倍频段划分，使用默认结构
                    self.section_names = ["倍频前"]
                    num_columns = 1 + len(self.section_names) * 3
                    self.param_table.setColumnCount(num_columns)
                    
                    # 设置表头
                    headers = ["频率 (GHz)"]
                    for section_name in self.section_names:
                        headers.append(f"{section_name} Kc (Ω)")
                        headers.append(f"{section_name} Loss_perunit")
                        headers.append(f"{section_name} Vpc (c)")
                    
                    self.param_table.setHorizontalHeaderLabels(headers)
                    self.param_table.setRowCount(0)
                    self.log_message("使用默认表格结构")
                
                # 3. 加载表格数据
                for row_data in config.get('param_table', []):
                    row = self.param_table.rowCount()
                    self.param_table.insertRow(row)
                    for col, value in enumerate(row_data):
                        if col < self.param_table.columnCount():
                            item = QTableWidgetItem(str(value))
                            self.param_table.setItem(row, col, item)
            
            self.log_message(f"已加载最新配置: {latest}")
        except Exception as e:
            self.log_message(f"加载配置时出错: {str(e)}")
    
    def export_data(self):
        """导出结果数据"""
        if not self.history_results:
            QMessageBox.warning(self, "导出错误", "没有可导出的数据")
            return
            
        default_name = f"TWT_Results_{datetime.now().strftime('%Y%m%d')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "导出计算结果", 
            default_name, 
            "CSV文件 (*.csv)"
        )
        
        if not path:
            return
            
        try:
            # 确保扩展名
            if not path.lower().endswith('.csv'):
                path += '.csv'
                
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["曲线名称", "频率 (GHz)", "输出功率 (W)"])
                
                # 导出所有历史结果
                for i, (freqs, powers, label) in enumerate(self.history_results):
                    # 对每组数据排序
                    sorted_points = sorted(zip(freqs, powers), key=lambda x: x[0])
                    for freq, power in sorted_points:
                        writer.writerow([label, freq, power])
                
            self.log_message(f"数据已导出至: {path}")
            QMessageBox.information(self, "导出成功", f"结果数据已导出到:\n{path}")
        except Exception as e:
            self.log_message(f"导出数据时出错: {str(e)}")
            QMessageBox.critical(self, "导出错误", f"导出数据失败:\n{str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 确保清理所有资源
        if self.timer:
            self.timer.stop()
            
        if self.pool:
            try:
                self.pool.terminate()
                self.pool.close()
            except:
                pass
            
        # 保存当前配置
        self.save_config()
        
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    calc = TWTCalculator()
    calc.show()
    sys.exit(app.exec_())