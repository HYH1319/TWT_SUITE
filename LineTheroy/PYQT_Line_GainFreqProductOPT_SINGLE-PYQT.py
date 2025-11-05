import sys
import os
import json
import csv
import random
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QGridLayout,
    QHeaderView,
    QProgressBar,
    QTabWidget,
    QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from TWT_CORE_SIMP import detailed_calculation as Line_CALC
from multiprocessing import Pool, cpu_count

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "KaiTi",
    "SimSun",
    "FangSong",
]
plt.rcParams["axes.unicode_minus"] = False
# 常量定义
CONFIG_DIR = "config"
RESULTS_DIR = "Results"
OPTIMIZATION_DIR = "Optimization"
TABLE_COLUMNS = ["Kc (Ω)", "Loss_perunit", "Freq (GHz)", "Vpc (c)"]
PARAM_NAMES = {
    "i": "电流 I (A)",
    "v": "电压 V (V)",
    "p_sws": "周期长度 p_SWS (mm)",
    "n_unit": "周期数 N_Unit",
    "w": "束流宽度 w (mm)",
    "t": "束流厚度 t (mm)",
    "Fn_K": "Fn_K 参数",
}


class Particle:
    """PSO粒子类"""

    def __init__(self, dim, bounds):
        self.position = np.zeros(dim)
        self.velocity = np.zeros(dim)
        self.best_position = np.zeros(dim)
        self.best_value = -float("inf")
        self.value = -float("inf")

        # 随机初始化位置和速度
        for d in range(dim):
            min_val, max_val = bounds[d]
            self.position[d] = random.uniform(min_val, max_val)
            self.velocity[d] = random.uniform(
                -(max_val - min_val) / 2, (max_val - min_val) / 2
            )
        self.best_position = np.copy(self.position)

    def update_velocity(self, global_best_position, w, c1, c2):
        """更新粒子速度"""
        for d in range(len(self.position)):
            r1 = random.random()
            r2 = random.random()
            cognitive = c1 * r1 * (self.best_position[d] - self.position[d])
            social = c2 * r2 * (global_best_position[d] - self.position[d])
            self.velocity[d] = w * self.velocity[d] + cognitive + social

    def update_position(self, bounds):
        """更新粒子位置"""
        for d in range(len(self.position)):
            self.position[d] += self.velocity[d]

            # 确保位置在边界内
            min_val, max_val = bounds[d]
            if self.position[d] < min_val:
                self.position[d] = min_val
                self.velocity[d] *= -0.5  # 反弹
            elif self.position[d] > max_val:
                self.position[d] = max_val
                self.velocity[d] *= -0.5  # 反弹


class OptimizationWorker(QThread):
    """优化工作线程"""

    progress_updated = pyqtSignal(int, float, float)
    iteration_completed = pyqtSignal(int, float, list)
    optimization_finished = pyqtSignal(list, float, list)

    def __init__(
        self,
        fixed_params,
        base_var_params,
        bounds,
        num_particles=20,
        max_iter=100,
        w=0.8,
        c1=1.5,
        c2=1.5,
    ):
        super().__init__()
        self.fixed_params = fixed_params
        self.base_var_params = base_var_params
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = len(base_var_params)
        self.pool = Pool(processes=cpu_count())

    def run(self):
        """执行PSO优化"""
        # 初始化粒子群
        particles = [Particle(self.dim, self.bounds) for _ in range(self.num_particles)]
        global_best_value = -float("inf")
        global_best_position = None

        # 初始评估
        positions = [p.position for p in particles]
        fitness_values = self.evaluate_fitness(positions)

        for i, p in enumerate(particles):
            p.value = fitness_values[i]
            p.best_value = fitness_values[i]
            p.best_position = np.copy(p.position)

            if fitness_values[i] > global_best_value:
                global_best_value = fitness_values[i]
                global_best_position = np.copy(p.position)

        # 主优化循环
        for iter_idx in range(self.max_iter):
            # 更新粒子
            for p in particles:
                p.update_velocity(global_best_position, self.w, self.c1, self.c2)
                p.update_position(self.bounds)

            # 评估新位置
            positions = [p.position for p in particles]
            fitness_values = self.evaluate_fitness(positions)

            # 更新粒子最佳位置和全局最佳
            for i, p in enumerate(particles):
                p.value = fitness_values[i]

                if fitness_values[i] > p.best_value:
                    p.best_value = fitness_values[i]
                    p.best_position = np.copy(p.position)

                    if fitness_values[i] > global_best_value:
                        global_best_value = fitness_values[i]
                        global_best_position = np.copy(p.position)

            # 计算平均适应度
            avg_fitness = np.mean(fitness_values)

            # 发出进度更新信号
            self.progress_updated.emit(iter_idx + 1, global_best_value, avg_fitness)
            self.iteration_completed.emit(
                iter_idx + 1, global_best_value, global_best_position.tolist()
            )

        # 优化完成
        self.optimization_finished.emit(
            global_best_position.tolist(),
            global_best_value,
            [p.best_value for p in particles],
        )
        self.pool.close()
        self.pool.join()

    def evaluate_fitness(self, positions):
        """并行评估适应度函数"""
        # 准备参数
        args_list = []
        for pos in positions:
            var_params = []
            for i, vpc in enumerate(pos):
                params = self.base_var_params[i].copy()
                params["Vpc"] = vpc
                var_params.append(params)
            args_list.append((self.fixed_params, var_params))

        # 并行计算适应度
        results = self.pool.starmap(calculate_fitness, args_list)
        return results


def calculate_fitness(fixed_params, var_params):
    """计算适应度（增益曲线面积）"""
    try:
        # 计算每个频率点的增益
        results = []
        for params in var_params:
            result = Line_CALC(
                fixed_params["i"],
                fixed_params["v"],
                params["Kc"],
                params["Loss_perunit"],
                fixed_params["p_sws"],
                fixed_params["n_unit"],
                fixed_params["w"],
                fixed_params["t"],
                fixed_params["Fn_K"],
                params["Freq"],
                params["Vpc"],
            )
            results.append((params["Freq"], result["Gmax"]))

        # 按频率排序并计算积分
        if len(results) < 2:
            return 0.0

        # 按频率排序
        sorted_results = sorted(results, key=lambda x: x[0])
        freqs = [r[0] for r in sorted_results]
        gains = [r[1] for r in sorted_results]

        # 梯形法计算定积分（曲线下面积）
        area = 0.0
        for i in range(1, len(sorted_results)):
            # 确保频率递增
            if freqs[i] <= freqs[i - 1]:
                continue

            delta_freq = freqs[i] - freqs[i - 1]
            avg_gain = (gains[i] + gains[i - 1]) / 2.0
            area += avg_gain * delta_freq

        return area
    except Exception as e:
        print(f"Error in fitness calculation: {e}")
        return 0.0


class ParamInput(QWidget):
    """简化的参数输入组件"""

    def __init__(self, param_key, parent=None):
        super().__init__(parent)
        self.param_key = param_key
        self.init_ui()

    def init_ui(self):
        """初始化界面元素"""
        self.layout = QHBoxLayout(self)
        self.label = QLabel(PARAM_NAMES[self.param_key])
        self.line_edit = QLineEdit()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.line_edit)

    @property
    def value(self):
        """获取当前参数值"""
        try:
            return float(self.line_edit.text())
        except ValueError:
            return 0.0


class TWTOptimizer(QMainWindow):
    """TWT优化器主窗口"""

    def __init__(self):
        super().__init__()
        self.plot_counter = 1
        self.history_results = []  # 存储历史计算结果
        self.history_labels = []  # 存储历史结果标签
        self.optimization_worker = None
        self.init_ui()
        self.load_last_config()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("TWT并行优化器")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 使用选项卡布局
        self.tab_widget = QTabWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(self.tab_widget)

        # 创建参数配置选项卡
        param_tab = QWidget()
        param_layout = QHBoxLayout(param_tab)
        param_layout.addLayout(self.create_left_panel(), 70)
        param_layout.addLayout(self.create_right_panel(), 30)
        self.tab_widget.addTab(param_tab, "参数配置")

        # 创建优化选项卡
        self.create_optimization_tab()

        # 创建结果选项卡
        self.create_results_tab()

    def create_left_panel(self):
        """创建左侧面板"""
        left_layout = QVBoxLayout()

        # 固定参数组
        fixed_group = QGroupBox("固定参数")
        fixed_layout = QGridLayout()
        self.param_widgets = {}

        for idx, param_key in enumerate(PARAM_NAMES.keys()):
            widget = ParamInput(param_key)
            row = idx // 2
            col = (idx % 2) * 2
            fixed_layout.addWidget(widget, row, col, 1, 2)
            self.param_widgets[param_key] = widget

        fixed_group.setLayout(fixed_layout)
        left_layout.addWidget(fixed_group)

        # 绘图区域
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)

        # 历史曲线控制
        history_layout = QHBoxLayout()
        self.history_combo = QComboBox()
        self.history_combo.setToolTip("选择要显示的历史曲线")
        self.clear_history_btn = QPushButton("清空历史曲线")
        self.clear_history_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(QLabel("历史曲线:"))
        history_layout.addWidget(self.history_combo, 1)
        history_layout.addWidget(self.clear_history_btn)
        left_layout.addLayout(history_layout)

        return left_layout

    def create_right_panel(self):
        """创建右侧面板"""
        right_layout = QVBoxLayout()

        # 参数表格组
        table_group = QGroupBox("可变参数组")
        table_layout = QVBoxLayout()

        # 表格操作按钮
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("＋ 添加行", clicked=self.add_table_row)
        self.del_btn = QPushButton("－ 删除行", clicked=self.delete_table_row)
        self.import_btn = QPushButton("导入CSV", clicked=self.import_csv)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.del_btn)
        btn_layout.addWidget(self.import_btn)

        # 参数表格
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        table_layout.addLayout(btn_layout)
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        right_layout.addWidget(table_group)

        # 功能按钮
        self.add_function_buttons(right_layout)
        return right_layout

    def add_function_buttons(self, layout):
        """添加功能按钮"""
        buttons = [
            ("计算当前配置", self.calculate),
            ("保存配置", self.save_config),
            ("导出数据", self.export_data),
        ]

        btn_layout = QVBoxLayout()
        for text, handler in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

    def create_optimization_tab(self):
        """创建优化配置选项卡"""
        opt_tab = QWidget()
        opt_layout = QVBoxLayout(opt_tab)

        # 优化参数配置
        param_group = QGroupBox("优化参数")
        param_layout = QGridLayout()

        # 粒子数
        param_layout.addWidget(QLabel("粒子数量:"), 0, 0)
        self.particle_count = QLineEdit("20")
        param_layout.addWidget(self.particle_count, 0, 1)

        # 迭代次数
        param_layout.addWidget(QLabel("迭代次数:"), 1, 0)
        self.max_iterations = QLineEdit("100")
        param_layout.addWidget(self.max_iterations, 1, 1)

        # 惯性权重
        param_layout.addWidget(QLabel("惯性权重 (w):"), 2, 0)
        self.inertia_weight = QLineEdit("0.8")
        param_layout.addWidget(self.inertia_weight, 2, 1)

        # 认知系数
        param_layout.addWidget(QLabel("认知系数 (c1):"), 3, 0)
        self.cognitive_coeff = QLineEdit("1.5")
        param_layout.addWidget(self.cognitive_coeff, 3, 1)

        # 社会系数
        param_layout.addWidget(QLabel("社会系数 (c2):"), 4, 0)
        self.social_coeff = QLineEdit("1.5")
        param_layout.addWidget(self.social_coeff, 4, 1)

        # 优化范围
        param_layout.addWidget(QLabel("优化范围 (%):"), 5, 0)
        self.optimization_range = QLineEdit("20")
        param_layout.addWidget(self.optimization_range, 5, 1)

        param_group.setLayout(param_layout)
        opt_layout.addWidget(param_group)

        # 优化控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始优化", clicked=self.start_optimization)
        self.stop_btn = QPushButton("停止优化", clicked=self.stop_optimization)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        opt_layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("优化进度: %p%")
        opt_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("就绪")
        opt_layout.addWidget(self.status_label)

        # 最佳面积标签
        self.best_fitness_label = QLabel("最佳面积: -")
        opt_layout.addWidget(self.best_fitness_label)

        # 平均面积标签
        self.avg_fitness_label = QLabel("平均面积: -")
        opt_layout.addWidget(self.avg_fitness_label)

        self.tab_widget.addTab(opt_tab, "优化配置")

    def create_results_tab(self):
        """创建结果展示选项卡"""
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # 优化结果绘图
        self.results_figure = plt.figure()
        self.results_canvas = FigureCanvas(self.results_figure)
        results_layout.addWidget(self.results_canvas)

        # 结果表格
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(
            ["迭代", "增益曲线面积", "平均面积"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)

        # 结果操作按钮
        btn_layout = QHBoxLayout()
        self.save_results_btn = QPushButton(
            "保存优化结果", clicked=self.save_optimization_results
        )
        self.apply_results_btn = QPushButton(
            "应用最佳结果", clicked=self.apply_best_result
        )
        btn_layout.addWidget(self.save_results_btn)
        btn_layout.addWidget(self.apply_results_btn)
        results_layout.addLayout(btn_layout)

        self.tab_widget.addTab(results_tab, "优化结果")

    def add_table_row(self):
        """添加表格行"""
        self.table.insertRow(self.table.rowCount())

    def delete_table_row(self):
        """删除表格行"""
        current_row = self.table.currentRow()
        if current_row == -1:
            if self.table.rowCount() > 0:
                self.table.removeRow(self.table.rowCount() - 1)
        else:
            self.table.removeRow(current_row)

    def import_csv(self):
        """导入CSV文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开CSV文件", "", "CSV文件 (*.csv)"
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳过标题行
                    for row in reader:
                        if len(row) < 4:
                            continue
                        row_num = self.table.rowCount()
                        self.table.insertRow(row_num)
                        for col, value in enumerate(row[:4]):
                            self.table.setItem(
                                row_num, col, QTableWidgetItem(value.strip())
                            )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"文件读取失败: {str(e)}")

    def get_fixed_params(self):
        """获取固定参数"""
        return {key: widget.value for key, widget in self.param_widgets.items()}

    def get_var_params(self):
        """获取可变参数"""
        params = []
        for row in range(self.table.rowCount()):
            try:
                params.append(
                    {
                        "Kc": float(self.table.item(row, 0).text()),
                        "Loss_perunit": float(self.table.item(row, 1).text()),
                        "Freq": float(self.table.item(row, 2).text()),
                        "Vpc": float(self.table.item(row, 3).text()),
                    }
                )
            except (AttributeError, ValueError):
                continue
        return params

    def calculate(self):
        """执行计算"""
        try:
            fixed_params = self.get_fixed_params()
            var_params = self.get_var_params()

            if not var_params:
                QMessageBox.warning(self, "错误", "没有可计算的参数！")
                return

            results = self.run_calculations(fixed_params, var_params)
            self.save_results(results, fixed_params)
            self.update_plot(results, fixed_params)

            # 计算当前配置的面积
            area = self.calculate_area(results)

            QMessageBox.information(
                self,
                "完成",
                f"成功计算{len(var_params)}组数据\n" f"增益曲线面积: {area:.4f} dB·GHz",
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生错误：\n{str(e)}")

    def run_calculations(self, fixed_params, var_params):
        """执行批量计算"""
        results = []
        for params in var_params:
            try:
                result = Line_CALC(
                    fixed_params["i"],
                    fixed_params["v"],
                    params["Kc"],
                    params["Loss_perunit"],
                    fixed_params["p_sws"],
                    fixed_params["n_unit"],
                    fixed_params["w"],
                    fixed_params["t"],
                    fixed_params["Fn_K"],
                    params["Freq"],
                    params["Vpc"],
                )
                results.append((params["Freq"], result["Gmax"]))
            except Exception as e:
                QMessageBox.critical(self, "计算错误", f"参数错误：{str(e)}")
                raise
        return results

    def calculate_area(self, results):
        """计算增益曲线面积"""
        if len(results) < 2:
            return 0.0

        # 按频率排序
        sorted_results = sorted(results, key=lambda x: x[0])
        freqs = [r[0] for r in sorted_results]
        gains = [r[1] for r in sorted_results]

        # 梯形法计算定积分
        area = 0.0
        for i in range(1, len(sorted_results)):
            if freqs[i] <= freqs[i - 1]:
                continue
            delta_freq = freqs[i] - freqs[i - 1]
            avg_gain = (gains[i] + gains[i - 1]) / 2.0
            area += avg_gain * delta_freq

        return area

    def save_results(self, results, fixed_params):
        """保存结果文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, f"Result_{timestamp}.csv")

        # 计算面积
        area = self.calculate_area(results)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"# 增益曲线面积: {area:.4f} dB·GHz\n")
            f.write("Frequency(GHz),Gain(dB),Kc(Ohm),Loss_perunit,Vpc(c)\n")
            for params, (freq, power) in zip(self.get_var_params(), results):
                f.write(
                    f"{freq},{power},{params['Kc']},{params['Loss_perunit']},{params['Vpc']}\n"
                )

        # 保存到历史记录
        label = f"配置{self.plot_counter} (面积: {area:.2f} dB·GHz)"
        self.history_results.append((results, fixed_params, label))
        self.history_labels.append(label)
        self.plot_counter += 1

        # 更新历史曲线下拉框
        self.history_combo.clear()
        self.history_combo.addItems(self.history_labels)

    def update_plot(self, results, fixed_params):
        """更新绘图 - 包含历史曲线叠绘"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 获取当前选中的历史曲线
        selected_history = self.history_combo.currentIndex()

        # 绘制所有历史曲线
        for idx, (data, params, label) in enumerate(self.history_results):
            # 跳过当前选中的历史曲线（会在后面单独绘制）
            if idx == selected_history:
                continue

            freqs, gains = zip(*sorted(data, key=lambda x: x[0]))
            ax.plot(freqs, gains, "o-", markersize=4, label=label, alpha=0.7)

        # 绘制当前选中的历史曲线（或当前计算结果）
        if selected_history >= 0:
            data, params, label = self.history_results[selected_history]
            freqs, gains = zip(*sorted(data, key=lambda x: x[0]))
            ax.plot(freqs, gains, "o-", markersize=4, label=label, linewidth=2)
        else:
            # 绘制当前计算结果
            freqs, gains = zip(*sorted(results, key=lambda x: x[0]))
            ax.plot(freqs, gains, "o-", markersize=4, label="当前配置", linewidth=2)

        # 计算并显示当前配置的面积
        area = self.calculate_area(results)
        ax.text(
            0.95,
            0.05,
            f"面积: {area:.4f} dB·GHz",
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
        )

        ax.set_title("增益-频率曲线")
        ax.set_xlabel("频率 (GHz)")
        ax.set_ylabel("最大增益 (dB)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best")
        self.canvas.draw()

    def clear_history(self):
        """清空历史曲线"""
        self.history_results = []
        self.history_labels = []
        self.history_combo.clear()
        self.figure.clear()
        self.canvas.draw()
        QMessageBox.information(self, "完成", "历史曲线已清空")

    def start_optimization(self):
        """开始优化过程"""
        # 获取参数
        fixed_params = self.get_fixed_params()
        base_var_params = self.get_var_params()

        if not base_var_params:
            QMessageBox.warning(self, "错误", "没有可优化的参数！")
            return

        # 获取优化参数
        try:
            num_particles = int(self.particle_count.text())
            max_iter = int(self.max_iterations.text())
            w = float(self.inertia_weight.text())
            c1 = float(self.cognitive_coeff.text())
            c2 = float(self.social_coeff.text())
            range_percent = float(self.optimization_range.text()) / 100.0
        except ValueError:
            QMessageBox.critical(self, "错误", "优化参数必须是有效的数字！")
            return

        # 设置边界（Vpc ±range_percent）
        bounds = []
        for params in base_var_params:
            vpc = params["Vpc"]
            min_val = max(vpc * (1 - range_percent), 0.01)  # 最小0.01c
            max_val = vpc * (1 + range_percent)
            bounds.append((min_val, max_val))

        # 创建并启动优化线程
        self.optimization_worker = OptimizationWorker(
            fixed_params, base_var_params, bounds, num_particles, max_iter, w, c1, c2
        )

        # 连接信号
        self.optimization_worker.progress_updated.connect(
            self.update_optimization_progress
        )
        self.optimization_worker.iteration_completed.connect(
            self.update_iteration_results
        )
        self.optimization_worker.optimization_finished.connect(self.finish_optimization)

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("优化进行中...")
        self.progress_bar.setValue(0)
        self.best_fitness_label.setText("最佳面积: -")
        self.avg_fitness_label.setText("平均面积: -")

        # 清空结果表格
        self.results_table.setRowCount(0)

        # 开始优化
        self.optimization_worker.start()

    def stop_optimization(self):
        """停止优化过程"""
        if self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.terminate()
            self.status_label.setText("优化已停止")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def update_optimization_progress(self, iteration, best_fitness, avg_fitness):
        """更新优化进度"""
        max_iter = int(self.max_iterations.text())
        progress = int(iteration / max_iter * 100)
        self.progress_bar.setValue(progress)
        self.best_fitness_label.setText(f"最佳面积: {best_fitness:.6f} dB·GHz")
        self.avg_fitness_label.setText(f"平均面积: {avg_fitness:.6f} dB·GHz")
        self.status_label.setText(f"优化进行中... 迭代 {iteration}/{max_iter}")

    def update_iteration_results(self, iteration, best_fitness, best_position):
        """更新迭代结果"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        self.results_table.setItem(row, 0, QTableWidgetItem(str(iteration)))
        self.results_table.setItem(row, 1, QTableWidgetItem(f"{best_fitness:.6f}"))

        # 计算平均面积
        avg_fitness = 0
        count = 0
        for i in range(self.results_table.rowCount()):
            value = float(self.results_table.item(i, 1).text())
            avg_fitness += value
            count += 1

        if count > 0:
            avg_fitness /= count
        else:
            avg_fitness = 0

        self.results_table.setItem(row, 2, QTableWidgetItem(f"{avg_fitness:.6f}"))

        # 滚动到最后一行
        self.results_table.scrollToBottom()

    def finish_optimization(self, best_position, best_fitness, fitness_values):
        """完成优化过程"""
        self.best_position = best_position
        self.best_fitness = best_fitness
        self.fitness_values = fitness_values

        self.progress_bar.setValue(100)
        self.status_label.setText("优化完成！")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # 更新结果绘图
        self.update_results_plot()

        # 计算原始面积
        original_area = self.calculate_original_area()

        # 显示结果
        message = f"优化完成！最佳面积: {best_fitness:.4f} dB·GHz\n"
        if original_area > 0:
            improvement = (best_fitness - original_area) / original_area * 100
            message += f"相比原始配置: {'提升' if improvement > 0 else '下降'} {abs(improvement):.2f}%\n"

        QMessageBox.information(self, "完成", message)

    def calculate_original_area(self):
        """计算原始配置的面积"""
        fixed_params = self.get_fixed_params()
        var_params = self.get_var_params()

        try:
            results = []
            for params in var_params:
                result = Line_CALC(
                    fixed_params["i"],
                    fixed_params["v"],
                    params["Kc"],
                    params["Loss_perunit"],
                    fixed_params["p_sws"],
                    fixed_params["n_unit"],
                    fixed_params["w"],
                    fixed_params["t"],
                    fixed_params["Fn_K"],
                    params["Freq"],
                    params["Vpc"],
                )
                results.append((params["Freq"], result["Gmax"]))

            return self.calculate_area(results)
        except:
            return 0.0

    def update_results_plot(self):
        """更新结果绘图"""
        self.results_figure.clear()
        ax = self.results_figure.add_subplot(111)

        # 绘制面积分布
        ax.hist(self.fitness_values, bins=20, alpha=0.7)
        ax.set_title("增益曲线面积分布")
        ax.set_xlabel("面积 (dB·GHz)")
        ax.set_ylabel("粒子数量")
        ax.grid(True, linestyle="--", alpha=0.7)

        # 添加最佳值标记
        ax.axvline(
            self.best_fitness,
            color="r",
            linestyle="--",
            label=f"最佳值: {self.best_fitness:.4f} dB·GHz",
        )
        ax.legend()

        self.results_canvas.draw()

    def save_optimization_results(self):
        """保存优化结果"""
        if not hasattr(self, "best_position") or not hasattr(self, "best_fitness"):
            QMessageBox.warning(self, "警告", "没有可保存的优化结果！")
            return

        # 创建优化结果目录
        os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OPTIMIZATION_DIR, f"optimization_{timestamp}.csv")

        # 获取固定参数
        fixed_params = self.get_fixed_params()
        base_var_params = self.get_var_params()

        # 写入结果
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["优化结果", f"最佳面积: {self.best_fitness:.6f} dB·GHz"])
            writer.writerow([])
            writer.writerow(["固定参数", "值"])
            for key, value in fixed_params.items():
                writer.writerow([key, value])

            writer.writerow([])
            writer.writerow(["频率(GHz)", "原始Vpc", "优化后Vpc", "变化百分比"])

            for i, params in enumerate(base_var_params):
                freq = params["Freq"]
                orig_vpc = params["Vpc"]
                opt_vpc = self.best_position[i]
                change_percent = (opt_vpc - orig_vpc) / orig_vpc * 100
                writer.writerow([freq, orig_vpc, opt_vpc, f"{change_percent:.2f}%"])

        QMessageBox.information(self, "保存成功", f"优化结果已保存至：\n{save_path}")

    def apply_best_result(self):
        """应用最佳结果到参数表"""
        if not hasattr(self, "best_position"):
            QMessageBox.warning(self, "警告", "没有可应用的优化结果！")
            return

        base_var_params = self.get_var_params()
        if len(base_var_params) != len(self.best_position):
            QMessageBox.critical(self, "错误", "参数数量不匹配！")
            return

        # 更新表格中的Vpc值
        for row in range(self.table.rowCount()):
            if row < len(self.best_position):
                self.table.setItem(
                    row, 3, QTableWidgetItem(f"{self.best_position[row]:.6f}")
                )

        QMessageBox.information(self, "完成", "最佳结果已应用到参数表！")

    def save_config(self):
        """保存当前配置"""
        config = {
            "fixed": {k: str(widget.value) for k, widget in self.param_widgets.items()},
            "variables": [
                [self.table.item(row, col).text() for col in range(4)]
                for row in range(self.table.rowCount())
            ],
        }

        os.makedirs(CONFIG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CONFIG_DIR, f"config_{timestamp}.json")

        with open(save_path, "w") as f:
            json.dump(config, f, indent=4)

        QMessageBox.information(self, "保存成功", f"配置已保存至：\n{save_path}")

    def export_data(self):
        """导出数据文件"""
        if not self.history_results:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存数据文件", "", "数据文件 (*.dat)"
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            for idx, (data, params, label) in enumerate(self.history_results):
                f.write(f"# Parameters Set {idx+1}\n")
                f.write(f"# {'-'*50}\n")
                for key, value in params.items():
                    f.write(f"# {key}: {value}\n")
                f.write(f"# {'-'*50}\n")

                # 计算面积并输出
                area = self.calculate_area(data)
                f.write(f"# 增益曲线面积: {area:.6f} dB·GHz\n")
                f.write("Frequency(GHz)\tGain(dB)\n")
                for freq, power in sorted(data, key=lambda x: x[0]):
                    f.write(f"{freq:.6f}\t{power:.6f}\n")
                f.write("\n\n")

        QMessageBox.information(self, "导出成功", f"数据已保存至：\n{path}")

    def load_last_config(self):
        """加载最近配置"""
        try:
            if not os.path.exists(CONFIG_DIR):
                return

            config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
            if not config_files:
                return

            latest_file = max(
                config_files,
                key=lambda f: os.path.getctime(os.path.join(CONFIG_DIR, f)),
            )
            with open(os.path.join(CONFIG_DIR, latest_file), "r") as f:
                config = json.load(f)

                # 加载固定参数
                for key, value in config["fixed"].items():
                    if key in self.param_widgets:
                        self.param_widgets[key].line_edit.setText(value)

                # 加载表格数据
                self.table.setRowCount(0)
                for row in config["variables"]:
                    row_num = self.table.rowCount()
                    self.table.insertRow(row_num)
                    for col in range(4):
                        self.table.setItem(row_num, col, QTableWidgetItem(row[col]))

        except Exception as e:
            print(f"配置加载错误: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TWTOptimizer()
    window.show()
    sys.exit(app.exec_())
