import sys
import os
import json
import csv
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QSlider, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QGridLayout, QHeaderView
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Noline_GAIN_MAINCALL_VUCORE_ import Noline_CORE_CALC as Noline_CALC

# 常量定义
CONFIG_DIR = "config"
RESULTS_DIR = "Results"
TABLE_COLUMNS = ["Kc (Ω)", "Loss_perunit", "Freq (GHz)", "Vpc (c)"]
PARAM_CONFIG = {
    'i': {'label': '电流 I (A)', 'range': (0, None), 'step_type': 'percent'},
    'v': {'label': '电压 V (V)', 'range': (0, None), 'step_type': 'linear', 'scale': 100, 'slider_range': (0, 500)},
    'p_sws': {'label': '周期长度 p_SWS (mm)', 'range': (0, None), 'step_type': 'percent'},
    'n_unit': {'label': '周期数 N_Unit', 'range': (1, 1000), 'step_type': 'percent'},
    'w': {'label': '束流宽度 w (mm)', 'range': (0, None), 'step_type': 'percent'},
    't': {'label': '束流厚度 t (mm)', 'range': (0, None), 'step_type': 'percent'},
    'Fn_K': {'label': 'Fn_K 参数\n(w=t表示圆形束流,该参数为填充率倒数必须大于1，w!=t,仅为带状束流矫正)', 
            'range': (1, 10), 'step_type': 'percent'},
    'P_in': {'label': '输入功率 P_in (W)', 'range': (0, None), 'step_type': 'linear', 'scale': 100, 'slider_range': (0, 1000)}
}

class ParamWidget(QWidget):
    """参数输入组件，整合滑块和文本输入"""
    def __init__(self, param_key, parent=None):
        super().__init__(parent)
        self.param_key = param_key
        config = PARAM_CONFIG[param_key]
        self.min_val, self.max_val = config['range']
        self.step_type = config['step_type']
        self.base_value = (config['range'][0] + config['range'][1])/2 if config['range'][1] else 0
        
        self.init_ui(config)
        self.init_control(config)
        self.setup_connections()

    def init_ui(self, config):
        """初始化界面元素"""
        self.layout = QHBoxLayout(self)
        self.label = QLabel(config['label'])
        self.line_edit = QLineEdit()
        self.slider = QSlider(Qt.Horizontal)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.line_edit)
        self.layout.addWidget(self.slider)

    def init_control(self, config):
        """初始化控件参数"""
        if self.step_type == "linear":
            slider_config = config.get('slider_range', (0, 100))
            self.slider.setRange(*slider_config)
            self.slider.setValue(int(self.base_value / config.get('scale', 1)))
        elif self.step_type == "percent":
            self.slider.setRange(-50, 50)
            self.slider.setValue(0)
        self.update_display()

    def setup_connections(self):
        """建立信号连接"""
        self.slider.valueChanged.connect(self.slider_changed)
        self.line_edit.editingFinished.connect(self.text_changed)

    def slider_changed(self, value):
        """滑块变化事件处理"""
        if self.step_type == "linear":
            config = PARAM_CONFIG[self.param_key]
            actual_value = value * config.get('scale', 1)
        elif self.step_type == "percent":
            actual_value = self.base_value * (1 + 0.02 * value)
        
        self.line_edit.blockSignals(True)
        self.line_edit.setText(f"{actual_value:.4f}" if isinstance(actual_value, float) else f"{actual_value}")
        self.line_edit.blockSignals(False)

    def text_changed(self):
        """文本变化事件处理"""
        try:
            new_value = float(self.line_edit.text())
        except ValueError:
            return

        if self.step_type == "percent":
            self.base_value = new_value
            self.slider.setValue(0)
        elif self.step_type == "linear":
            config = PARAM_CONFIG[self.param_key]
            self.slider.setValue(int(new_value / config.get('scale', 1)))
        self.update_display()

    def update_display(self):
        """更新显示"""
        if self.step_type == "percent":
            self.line_edit.setText(f"{self.base_value:.3f}")

    @property
    def value(self):
        """获取当前参数值"""
        return float(self.line_edit.text())

class TWTCalculator(QMainWindow):
    """主应用程序窗口"""
    def __init__(self):
        super().__init__()
        self.plot_counter = 1
        self.history_results = []
        self.init_ui()
        self.load_last_config()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('非线性圆形束TWT计算器 v4.2 (PyQt5),带状束流圆形束流通吃')
        self.setGeometry(100, 100, 1400, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        main_layout.addLayout(self.create_left_panel(), 70)
        main_layout.addLayout(self.create_right_panel(), 30)

    def create_left_panel(self):
        """创建左侧面板"""
        left_layout = QVBoxLayout()
        
        # 固定参数组
        fixed_group = QGroupBox("固定参数")
        fixed_layout = QGridLayout()
        self.param_widgets = {}
        
        for idx, param_key in enumerate(PARAM_CONFIG.keys()):
            widget = ParamWidget(param_key)
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
            ("开始计算", self.calculate),
            ("批量处理目录", self.batch_process),
            ("清空绘图", self.clear_plot),
            ("保存配置", self.save_config),
            ("导出数据", self.export_data)
        ]
        
        btn_layout = QVBoxLayout()
        for text, handler in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

    def add_table_row(self):
        """添加表格行"""
        self.table.insertRow(self.table.rowCount())

    def delete_table_row(self):
        """删除表格行"""
        current_row = self.table.currentRow()
        if current_row == -1:
            if self.table.rowCount() > 0:
                self.table.removeRow(self.table.rowCount()-1)
        else:
            self.table.removeRow(current_row)

    def import_csv(self):
        """导入CSV文件"""
        path, _ = QFileDialog.getOpenFileName(self, "打开CSV文件", "", "CSV文件 (*.csv)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳过标题行
                    for row in reader:
                        if len(row) < 4:
                            continue
                        row_num = self.table.rowCount()
                        self.table.insertRow(row_num)
                        for col, value in enumerate(row[:4]):
                            self.table.setItem(row_num, col, QTableWidgetItem(value.strip()))
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
                params.append({
                    'Kc': float(self.table.item(row, 0).text()),
                    'Loss_perunit': float(self.table.item(row, 1).text()),
                    'Freq': float(self.table.item(row, 2).text()),
                    'Vpc': float(self.table.item(row, 3).text())
                })
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
            
            QMessageBox.information(self, "完成", f"成功计算{len(var_params)}组数据")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生错误：\n{str(e)}")

    def run_calculations(self, fixed_params, var_params):
        """执行批量计算"""
        results = []
        core_params = [
            fixed_params['i'], fixed_params['v'],
            fixed_params['p_sws'], fixed_params['n_unit'],
            fixed_params['w'], fixed_params['t'],
            fixed_params['Fn_K'],fixed_params['P_in']
        ]
        
        for params in var_params:
            try:
                result = Noline_CALC(
                    *core_params[:2],
                    params['Kc'],
                    params['Loss_perunit'],
                    *core_params[2:7],  # 更新索引范围
                    params['Freq'],
                    params['Vpc'],
                    core_params[7],  # P_in参数
                )
                results.append((params['Freq'], result['P_out']))
            except Exception as e:
                QMessageBox.critical(self, "计算错误", f"参数错误：{str(e)}")
                raise
        return results

    def save_results(self, results, fixed_params):
        """保存结果文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, f"Result_{timestamp}.csv")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Frequency(GHz),Power(W),Kc(Ohm),Loss_perunit,Vpc(c)\n")
            for params, (freq, power) in zip(self.get_var_params(), results):
                f.write(f"{freq},{power},{params['Kc']},{params['Loss_perunit']},{params['Vpc']}\n")
        
        self.history_results.append((results, fixed_params, f"param{self.plot_counter}"))
        self.plot_counter += 1

    def update_plot(self, results, fixed_params):
        """更新绘图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        for data in self.history_results:
            results, params, label = data
            freqs, powers = zip(*sorted(results, key=lambda x: x[0]))
            ax.plot(freqs, powers, 'o-', markersize=4, label=label)
        
        ax.set_title("Power-Frequency Curve")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Max Power (W)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        self.canvas.draw()

    def batch_process(self):
        """批量处理目录"""
        # 选择目录
        dir_path = QFileDialog.getExistingDirectory(self, "选择CSV文件目录")
        if not dir_path:
            return

        # 仅处理以MER开头且后缀为.csv的文件
        csv_files = [
            f for f in os.listdir(dir_path) 
            if f.endswith('.csv') and f.startswith('MER')
        ]
        if not csv_files:
            QMessageBox.warning(self, "无文件", "目录中没有符合条件的CSV文件")
            return

        # 获取固定参数（提前验证）
        try:
            fixed_params = self.get_fixed_params()
            # 验证必要参数范围
        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"固定参数错误：{str(e)}")
            return

        integral_results = []
        error_files = []

        # 处理每个CSV文件
        for csv_file in csv_files:
            csv_path = os.path.join(dir_path, csv_file)
            file_errors = []

            try:
                # 读取CSV参数
                var_params = self.read_csv_params(csv_path)
                if not var_params:
                    raise ValueError("文件内容为空或格式错误")

                # 计算每个参数组
                valid_results = []
                for param_idx, params in enumerate(var_params, 1):
                    try:
                        # 执行核心计算
                        result = Noline_CALC(
                            fixed_params['i'],
                            fixed_params['v'],
                            params['Kc'],
                            params['Loss_perunit'],
                            fixed_params['p_sws'],
                            fixed_params['n_unit'],
                            fixed_params['w'],
                            fixed_params['t'],
                            fixed_params['Fn_K'],
                            params['Freq'],
                            params['Vpc']
                        )
                        valid_results.append( (params['Freq'], result['P_out']) )

                    except Exception as e:
                        file_errors.append(f"第{param_idx}行错误: {str(e)}")
                        continue

                # 结果积分计算
                if len(valid_results) < 2:
                    raise ValueError("有效数据点不足（至少需要2个点）")

                freqs, powers = zip(*sorted(valid_results, key=lambda x: x[0]))
                integral = self.trapezoidal_integration(freqs, powers)
                integral_results.append( (csv_file, integral) )

            except Exception as e:
                error_files.append( f"{csv_file}: {str(e)}" + 
                                (f" ({'; '.join(file_errors)})" if file_errors else "") )
                continue

        # 结果展示与保存
        self.show_batch_results(integral_results, error_files, dir_path)

    def read_csv_params(self, path):
        """读取CSV参数"""
        params = []
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    if len(row) < 4:
                        continue
                    try:
                        params.append({
                            'Kc': float(row[0]),
                            'Loss_perunit': float(row[1]),
                            'Freq': float(row[2]),
                            'Vpc': float(row[3])
                        })
                    except ValueError:
                        continue
        except Exception as e:
            print(f"文件读取错误 {path}: {str(e)}")
        return params

    def trapezoidal_integration(self, x, y):
        """梯形法积分计算"""
        if len(x) != len(y):
            raise ValueError("x和y长度不一致")
        if len(x) < 2:
            raise ValueError("至少需要两个数据点")
            
        integral = 0.0
        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            avg_y = (y[i] + y[i-1]) / 2
            integral += dx * avg_y
        return integral

    def show_batch_results(self, results, errors, dir_path):
        """显示批量处理结果"""
        if errors:
            error_msg = "处理失败文件：\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n...等共{len(errors)}个错误"
            QMessageBox.warning(self, "处理警告", error_msg)

        if not results:
            QMessageBox.warning(self, "无结果", "没有有效计算结果")
            return

        # 绘制积分结果
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        files = [os.path.splitext(f)[0] for f, _ in results]
        integrals = [i for _, i in results]
        
        ax.plot(files, integrals, 'o-', markersize=5)
        ax.set_title("Power-Frequency Product")
        ax.set_xlabel("CSV File")
        ax.set_ylabel("Integral Value (W·GHz)")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        self.figure.tight_layout()
        self.canvas.draw()

        # 保存结果文件
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        save_path = os.path.join(dir_path, f"积分结果_{timestamp}.csv")
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["文件名", "功率带宽积(W·GHz)"])
            writer.writerows(results)
        
        QMessageBox.information(self, "完成", f"处理完成！结果保存至：\n{save_path}")

    def clear_plot(self):
        """清空绘图"""
        self.history_results.clear()
        self.plot_counter = 1
        self.figure.clear()
        self.canvas.draw()

    def export_data(self):
        """导出数据文件"""
        if not self.history_results:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存数据文件", "", "数据文件 (*.dat)")
        if not path:
            return

        with open(path, 'w', encoding='utf-8') as f:
            for idx, (data, params, label) in enumerate(self.history_results):
                f.write(f"# Parameters Set {idx+1}\n")
                f.write(f"# {'-'*50}\n")
                for key, value in params.items():
                    f.write(f"# {key}: {value}\n")
                f.write(f"# {'-'*50}\n")
                f.write("Frequency(GHz)\tPower(W)\n")
                for freq, power in sorted(data, key=lambda x: x[0]):
                    f.write(f"{freq:.6f}\t{power:.6f}\n")
                f.write("\n\n")
        
        QMessageBox.information(self, "导出成功", f"数据已保存至：\n{path}")

    def save_config(self):
        """保存当前配置"""
        config = {
            "fixed": {k: str(widget.value) for k, widget in self.param_widgets.items()},
            "variables": [[self.table.item(row, col).text() for col in range(4)] 
                         for row in range(self.table.rowCount())]
        }
        
        os.makedirs(CONFIG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CONFIG_DIR, f"config_{timestamp}.json")
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        QMessageBox.information(self, "保存成功", f"配置已保存至：\n{save_path}")

    def load_last_config(self):
        """加载最近配置"""
        try:
            if not os.path.exists(CONFIG_DIR):
                return
            
            config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
            if not config_files:
                return
                
            latest_file = max(config_files, key=lambda f: os.path.getctime(os.path.join(CONFIG_DIR, f)))
            with open(os.path.join(CONFIG_DIR, latest_file), 'r') as f:
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TWTCalculator()
    window.show()
    sys.exit(app.exec_())