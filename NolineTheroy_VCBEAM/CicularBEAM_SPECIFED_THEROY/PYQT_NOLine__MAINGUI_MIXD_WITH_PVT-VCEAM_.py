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
    QSplitter, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QMutex
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import math

from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER_MIX_WITH_PVT import calculate_SEGMENT_TWT_NOLINE

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
CONFIG_DIR = "config"
RESULTS_DIR = "Results"
TABLE_COLUMNS = ["Kc (Ω)", "Loss_perunit", "Freq (GHz)", "Vpc (c)"]
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # 保留1个核心给系统

# 精简后的参数配置
PARAM_CONFIG = [
    ('i', '电流 I (A)', '0.3', '电子枪发射电流'),
    ('v', '电压 V (V)', '23000', '电子枪加速电压'),
    ('p_sws', '周期长度 p_SWS (mm)', '0.5,0.5,0.5,0.5', '各段慢波结构的周期长度'),
    ('n_unit', '周期数 N_Unit', '25,7,12,13', '各段的周期数'),
    ('w', '束流宽度 w (mm)', '0.2', '电子束水平宽度'),
    ('t', '束流厚度 t (mm)', '0.2', '电子束垂直厚度'),
    ('Fn_K', 'Fn_K 参数', '1', '填充率倒数'),
    ('p_in', '输入功率 P_in (W)', '0.1', '输入信号功率'),
    ('loss_attu', '衰减量 (dB)', '20', '衰减段衰减量'),
    ('section_seg_idx', '衰减段索引', '1', '衰减段索引（0开始）'),
    ('vpc_coeff', 'Vpc 调整系数', '0.3', 'Vpc调整公式系数'),
    ('kc_coeff', 'Kc 调整系数', '0.9', 'Kc调整公式系数'),
]

# ================== 核心计算函数 ==================
def adjust_params(p_SWS, idx, Vpc, Kc, coeffs):
    """参数调整函数 - 定义为模块级函数用于并行"""
    current_p = p_SWS[idx]
    first_p = p_SWS[0]
    delta = (current_p - first_p) / first_p
    return {
        "Vpc": Vpc + coeffs['vpc'] * delta * Vpc,
        "Kc": Kc + coeffs['kc'] * delta * Kc
    }

def calculate_twt_point(params):
    """单个频率点计算函数 - 定义为模块级函数用于并行"""
    fixed, var, coeffs = params
    
    try:
        # 调用核心计算函数
        result = calculate_SEGMENT_TWT_NOLINE(
            I=fixed['i'],
            V=fixed['v'],
            Kc=var['Kc'],
            Loss_perunit=var['Loss_perunit'],
            SectionedSEGMENT_IDX=fixed['section_seg_idx'],
            p_SWS=fixed['p_sws'],
            N_unit=fixed['n_unit'],
            w=fixed['w'],
            t=fixed['t'],
            Fn_K=fixed['Fn_K'],
            f0_GHz=var['Freq'],
            Vpc=var['Vpc'],
            para_func=lambda p, i, v, k: adjust_params(p, i, v, k, coeffs),
            P_in=fixed['p_in'],
            Loss_attu=fixed['loss_attu']
        )
        return (var['Freq'], result['输出功率P_out'])
    except Exception as e:
        return (var['Freq'], f"计算错误: {str(e)}")

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
        self.init_ui()
        self.load_last_config()
    
    def init_ui(self):
        """紧凑的UI初始化"""
        self.setWindowTitle('TWT行波管计算器 (多核并行版)')
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
        
        # 表格区域
        table_group = QGroupBox("频率参数配置")
        table_layout = QVBoxLayout(table_group)
        
        # 表格控件
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(35)
        
        # 表格操作按钮
        table_btn_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("添加行")
        self.add_row_btn.clicked.connect(self.add_table_row)
        self.del_row_btn = QPushButton("删除行")
        self.del_row_btn.clicked.connect(self.delete_table_row)
        self.clear_table_btn = QPushButton("清空表格")
        self.clear_table_btn.clicked.connect(self.clear_table)
        self.import_btn = QPushButton("导入CSV")
        self.import_btn.clicked.connect(self.import_csv)
        
        table_btn_layout.addWidget(self.add_row_btn)
        table_btn_layout.addWidget(self.del_row_btn)
        table_btn_layout.addWidget(self.clear_table_btn)
        table_btn_layout.addWidget(self.import_btn)
        
        table_layout.addLayout(table_btn_layout)
        table_layout.addWidget(self.table)
        
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
        self.add_table_row()
    
    def setup_plot(self):
        """初始化图表设置"""
        self.ax.clear()
        self.ax.set_title("输出功率 vs 频率", fontsize=14)
        self.ax.set_xlabel("频率 (GHz)", fontsize=12)
        self.ax.set_ylabel("输出功率 (W)", fontsize=12)
        self.ax.grid(True)
        self.canvas.draw()
    
    # =================== 表格操作 ===================
    def add_table_row(self):
        """添加行到表格"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        default_values = ["3.6", "0", "211", "0.288"]
        for col, value in enumerate(default_values):
            item = QTableWidgetItem(value)
            self.table.setItem(row, col, item)
    
    def delete_table_row(self):
        """删除表格当前行"""
        if self.table.rowCount() > 0:
            current_row = self.table.currentRow()
            row_to_delete = current_row if current_row >= 0 else self.table.rowCount() - 1
            self.table.removeRow(row_to_delete)
    
    def clear_table(self):
        """清空表格"""
        if self.table.rowCount() > 0:
            self.table.setRowCount(0)
            self.add_table_row()
    
    def import_csv(self):
        """导入CSV数据"""
        path, _ = QFileDialog.getOpenFileName(self, "导入CSV文件", "", "CSV文件 (*.csv)")
        if not path:
            return
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # 清空表格
                self.table.setRowCount(0)
                
                # 读取所有行
                rows = list(reader)
                if not rows:
                    return
                    
                # 检查是否有表头
                if any(header in TABLE_COLUMNS for header in rows[0]):
                    start_row = 1  # 跳过表头
                else:
                    start_row = 0
                
                # 添加数据行
                for row_idx, row_data in enumerate(rows[start_row:], start=start_row):
                    if not row_data:  # 跳过空行
                        continue
                        
                    self.table.insertRow(row_idx - start_row)
                    for col in range(min(len(row_data), len(TABLE_COLUMNS))):
                        item = QTableWidgetItem(row_data[col].strip())
                        self.table.setItem(row_idx - start_row, col, item)
                
                self.log_message(f"已从 {path} 导入 {self.table.rowCount()} 行数据")
                
        except Exception as e:
            QMessageBox.critical(self, "导入错误", f"导入CSV文件时出错:\n{str(e)}")
            self.log_message(f"导入CSV文件时出错: {str(e)}")
    
    # =================== 核心计算功能 ===================
    def parse_params(self):
        """解析固定和可变参数"""
        # 解析固定参数
        fixed_params = {}
        for pid, widget in self.param_widgets.items():
            value = widget.value
            
            # 处理列表参数
            if pid in ['p_sws', 'n_unit', 'section_seg_idx']:
                try:
                    values = [v.strip() for v in value.split(',') if v.strip()]
                    if not values:
                        raise ValueError("参数不能为空")
                        
                    if pid == 'p_sws':
                        fixed_params[pid] = [float(v) for v in values]
                    else:  # n_unit 和 section_seg_idx
                        fixed_params[pid] = [int(v) for v in values]
                except ValueError as e:
                    QMessageBox.warning(self, "参数错误", f"{pid} 参数格式无效: {e}")
                    return None, None
            else:  # 数值参数
                try:
                    fixed_params[pid] = float(value)
                except ValueError:
                    QMessageBox.warning(self, "参数错误", f"{pid} 应为数值")
                    return None, None
        
        # 验证固定参数
        if len(fixed_params.get('p_sws', [])) != len(fixed_params.get('n_unit', [])):
            QMessageBox.warning(self, "参数错误", "p_SWS 和 N_unit 长度不一致")
            return None, None
        
        # 解析表格参数
        var_params = []
        for row in range(self.table.rowCount()):
            try:
                # 检查此行是否有有效数据
                has_data = False
                for col in range(4):
                    item = self.table.item(row, col)
                    if item and item.text().strip():
                        has_data = True
                        break
                
                if not has_data:
                    continue
                
                # 提取数据
                row_data = {}
                row_data['Kc'] = float(self.table.item(row, 0).text())
                row_data['Loss_perunit'] = float(self.table.item(row, 1).text())
                row_data['Freq'] = float(self.table.item(row, 2).text())
                row_data['Vpc'] = float(self.table.item(row, 3).text())
                var_params.append(row_data)
            except (ValueError, AttributeError) as e:
                self.log_message(f"跳过无效行 {row+1}: {str(e)}")
                continue
        
        if not var_params:
            QMessageBox.warning(self, "计算错误", "没有可计算的频率点")
            return None, None
        
        return fixed_params, var_params
    
    def start_calculation(self):
        """启动并行计算"""
        fixed, var = self.parse_params()
        if fixed is None or var is None:
            return
            
        self.log_message("= 计算开始 =")
        self.log_message(f"使用 {MAX_WORKERS} 个进程并行计算")
        self.log_message(f"共 {len(var)} 个频率点需要计算")
        
        # 准备并行参数
        try:
            vpc_coeff = float(self.param_widgets['vpc_coeff'].value)
            kc_coeff = float(self.param_widgets['kc_coeff'].value)
            coeffs = {'vpc': vpc_coeff, 'kc': kc_coeff}
        except ValueError:
            coeffs = {'vpc': 0.3, 'kc': 0.9}
        
        # 创建任务列表
        tasks = [(fixed, v, coeffs) for v in var]
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
        
        # 异步计算结果
        self.results = self.pool.imap_unordered(calculate_twt_point, tasks)
        
        # 创建定时器检查进度
        self.timer = QTimer()
        self.timer.timeout.connect(partial(self.check_results, total_tasks))
        self.timer.start(10)  # 每0.3秒检查一次
        self.start_time = time.time()
        self.completed = 0
        self.success_count = 0
        self.error_count = 0
        self.last_update_time = 0  # 记录上次图表更新时间
        
    def check_results(self, total_tasks):
        """定时检查计算结果 - 修复图表更新问题"""
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
                    freq, power = result
                    
                    # 处理结果
                    if isinstance(power, str) and "错误" in power:
                        self.error_count += 1
                        self.log_message(f"频率 {freq} GHz: {power}")
                    else:
                        self.success_count += 1
                        self.history.append((freq, power))
                        
                        # 安全处理NaN和无穷大的功率值
                        if not isinstance(power, (int, float)):
                            self.log_message(f"频率 {freq} GHz: 输出功率 (无效值)")
                        elif math.isnan(power) or math.isinf(power):
                            self.log_message(f"频率 {freq} GHz: 输出功率 (计算结果异常)")
                        else:
                            self.log_message(f"频率 {freq} GHz: 输出功率 {power:.2f} W")
                    
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
            # 过滤掉无效功率值
            valid_points = []
            for freq, power in zip(freqs, powers):
                # 只处理有效的数值
                if isinstance(power, (int, float)) and not math.isnan(power) and not math.isinf(power):
                    valid_points.append((freq, power))
            
            # 如果没有有效数据点，跳过该组
            if not valid_points:
                continue
                
            # 将频率和功率组合成点并排序
            points = sorted(valid_points, key=lambda x: x[0])
            sorted_freqs = [p[0] for p in points]
            sorted_powers = [p[1] for p in points]
            
            # 添加到全局范围
            all_freqs.extend(sorted_freqs)
            all_powers.extend(sorted_powers)
            
            # 绘制曲线
            self.ax.plot(sorted_freqs, sorted_powers, 'o-', markersize=5, label=label)
        
        # 设置图例
        if all_powers:  # 只有存在有效数据时设置图例
            self.ax.legend(loc='best')
        
        # 设置合适的数据范围 - 修复无效范围计算
        if not all_freqs or not all_powers:
            # 没有有效数据时设置默认范围
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
        else:
            min_freq = min(all_freqs)
            max_freq = max(all_freqs)
            min_power = min(all_powers)
            max_power = max(all_powers)
            
            # 计算安全的坐标系范围（避免除以零）
            x_range = max_freq - min_freq
            y_range = max_power - min_power
            
            x_padding = max(x_range * 0.02, 0.01) if x_range > 0 else 1.0
            y_padding = max(y_range * 0.05, 0.01) if y_range > 0 else 1.0
            
            self.ax.set_xlim(min_freq - x_padding, max_freq + x_padding)
            self.ax.set_ylim(min_power - y_padding, max_power + y_padding)
        
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
        config = {'fixed': {}, 'variable': []}
        
        try:
            # 保存固定参数
            for pid, widget in self.param_widgets.items():
                config['fixed'][pid] = widget.value
            
            # 添加表格数据
            for row in range(self.table.rowCount()):
                row_data = []
                for col in range(4):
                    item = self.table.item(row, col)
                    if item and item.text():
                        row_data.append(item.text().strip())
                    else:
                        row_data.append("")
                config['variable'].append(row_data)
            
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
                
                # 加载固定参数
                for pid, value in config.get('fixed', {}).items():
                    if pid in self.param_widgets:
                        self.param_widgets[pid].line_edit.setText(str(value))
                
                # 加载表格
                self.table.setRowCount(0)
                for row_data in config.get('variable', []):
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    for col, value in enumerate(row_data):
                        if col < 4:  # 只加载前四列
                            item = QTableWidgetItem(str(value))
                            self.table.setItem(row, col, item)
            
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