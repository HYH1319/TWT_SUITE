import sys
import os
import json
import csv
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QLineEdit, QSlider, QTableWidget, QTableWidgetItem,
                             QPushButton, QFileDialog, QMessageBox, QGridLayout, QHeaderView)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys


from TWT_CORE_COMPLEX import detailed_calculation

class ParamWidget(QWidget):
    def __init__(self, key, label, param_range, step_type, parent=None):
        super().__init__(parent)
        self.key = key
        self.min_val, self.max_val = param_range
        self.step_type = step_type
        self.is_array = (key == 'n_unit')  # 标识数组类型参数
        self.layout = QHBoxLayout(self)
        
        self.label = QLabel(label)
        self.line_edit = QLineEdit()
        
        # 特殊处理数组参数
        if self.is_array:
            self.line_edit.setPlaceholderText("逗号分隔，如：20,30,40")
            self.slider = None
        else:
            self.slider = QSlider(Qt.Horizontal)
            if not self.is_array:
                self.layout.addWidget(self.slider)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.line_edit)
        if not self.is_array and self.slider:
            self.layout.addWidget(self.slider)
        
        self.base_value = self.calculate_base_value()
        self.init_control()
        self.setup_connections()

    def calculate_base_value(self):
        if self.is_array:
            return None
        if self.max_val is None:
            return self.min_val
        return (self.min_val + self.max_val) / 2

    def init_control(self):
        if self.is_array:
            self.line_edit.setText("20,20,30")  # 默认示例值
            return
        
        if self.step_type == "linear":
            if self.key == 'i':
                self.slider.setRange(0, 10000)
                self.slider.setValue(int(self.base_value * 100))
            elif self.key == 'v':
                self.slider.setRange(0, 500)
                self.slider.setValue(int(self.base_value / 100))
            elif self.key == 'Fn_K':
                self.slider.setRange(0, 1000)
                self.slider.setValue(int(self.base_value * 100))
            else:
                self.slider.setRange(0, 100)
                self.slider.setValue(50)
        elif self.step_type == "percent":
            self.slider.setRange(-50, 50)
            self.slider.setValue(0)
        
        self.update_display()

    def setup_connections(self):
        if not self.is_array and self.slider:
            self.slider.valueChanged.connect(self.slider_changed)
        self.line_edit.editingFinished.connect(self.text_changed)

    def slider_changed(self, value):
        if self.step_type == "linear":
            if self.key == 'i':
                actual_value = value * 0.01
            elif self.key == 'v':
                actual_value = value * 100
            elif self.key == 'Fn_K':
                actual_value = value * 0.01
            else:
                actual_value = self.min_val + (self.max_val - self.min_val) * (value / 100)
        elif self.step_type == "percent":
            actual_value = self.base_value * (1 + 0.02 * value)
        
        self.line_edit.blockSignals(True)
        self.line_edit.setText(f"{actual_value:.2f}" if isinstance(actual_value, float) else f"{actual_value}")
        self.line_edit.blockSignals(False)

    def text_changed(self):
        if self.is_array:
            return  # 数组参数不处理滑块同步
        
        text = self.line_edit.text()
        try:
            new_value = float(text)
        except ValueError:
            return

        if self.step_type == "percent":
            self.base_value = new_value
            self.slider.setValue(0)
        
        if self.step_type == "linear":
            if self.key == 'i':
                self.slider.setValue(int(new_value * 100))
            elif self.key == 'v':
                self.slider.setValue(int(new_value / 100))
            elif self.key == 'Fn_K':
                self.slider.setValue(int(new_value * 100))
            else:
                slider_value = int((new_value - self.min_val) / (self.max_val - self.min_val) * 100)
                self.slider.setValue(slider_value)
        self.update_display()

    def update_display(self):
        if self.step_type == "percent":
            self.line_edit.setText(f"{self.base_value:.3f}")

class TWTCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.param_ranges = {
            'i': (0, None), 'v': (0, None), 
            'p_sws': (0, None), 'n_unit': (1, 1000),
            'w': (0, None), 't': (0, None),
            'Kc': (0, None), 'Loss_perunit': (0, None),
            'Freq': (0.1, None), 'Vpc': (0.1, 0.99),
            'Fn_K': (0, None)
        }
        self.step_types = {
            'i': 'linear', 'v': 'linear', 'Fn_K': 'linear',
            'p_sws': 'percent', 'n_unit': None,  # 禁用n_unit滑块
            'w': 'percent', 't': 'percent'
        }
        self.plot_counter = 1
        self.history_results = []
        self.init_ui()
        self.load_last_config()

    def init_ui(self):
        self.setWindowTitle('TWT计算器 v4.3 (支持多段计算)')
        self.setGeometry(100, 100, 1400, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        left_panel = QWidget()
        right_panel = QWidget()
        main_layout.addWidget(left_panel, 70)
        main_layout.addWidget(right_panel, 30)

        left_layout = QVBoxLayout(left_panel)
        
        fixed_group = QGroupBox("固定参数")
        fixed_layout = QGridLayout()
        params = [
            ('i', '电流 I (A)'), ('v', '电压 V (V)'),
            ('p_sws', '周期长度 p_SWS (mm)'), ('n_unit', '周期数 N_Unit（逗号分隔）'),
            ('w', '束流宽度 w (mm)'), ('t', '束流厚度 t (mm)'),
            ('Fn_K', 'Fn_K 参数')
        ]
        
        self.param_widgets = {}
        for idx, (key, label) in enumerate(params):
            widget = ParamWidget(
                key, label, 
                self.param_ranges[key],
                self.step_types.get(key, 'linear'),
                self
            )
            row = idx // 2
            col = (idx % 2) * 2
            fixed_layout.addWidget(widget, row, col, 1, 2)
            self.param_widgets[key] = widget
        
        fixed_group.setLayout(fixed_layout)
        left_layout.addWidget(fixed_group)

        self.figure = plt.figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)

        right_layout = QVBoxLayout(right_panel)
        
        table_group = QGroupBox("可变参数组")
        table_layout = QVBoxLayout()
        
        table_btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("＋ 添加行", clicked=self.add_table_row)
        self.del_btn = QPushButton("－ 删除行", clicked=self.delete_table_row)
        self.import_btn = QPushButton("导入CSV", clicked=self.import_csv)
        table_btn_layout.addWidget(self.add_btn)
        table_btn_layout.addWidget(self.del_btn)
        table_btn_layout.addWidget(self.import_btn)
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Kc (Ω)", "Loss_perunit", "Freq (GHz)", "Vpc (c)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        table_layout.addLayout(table_btn_layout)
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        right_layout.addWidget(table_group)

        btn_layout = QVBoxLayout()
        self.calc_btn = QPushButton("开始计算", clicked=self.calculate)
        self.clear_btn = QPushButton("清空绘图", clicked=self.clear_plot)
        self.save_btn = QPushButton("保存配置", clicked=self.save_config)
        self.export_btn = QPushButton("导出数据", clicked=self.export_data)
        
        btn_layout.addWidget(self.calc_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.export_btn)
        right_layout.addLayout(btn_layout)

    def validate_n_units(self, text):
        try:
            values = [float(x.strip()) for x in text.split(',')]
            if not values:
                raise ValueError("至少需要一个周期数")
            min_val, max_val = self.param_ranges['n_unit']
            for val in values:
                if val < min_val or (max_val is not None and val > max_val):
                    raise ValueError(f"值 {val:.1f} 超出范围 ({min_val}-{max_val})")
            return values
        except Exception as e:
            QMessageBox.critical(self, "输入错误", 
                f"周期数格式无效：\n{str(e)}\n"
                "正确示例：20,30,40")
            return None

    def calculate(self):
        try:
            fixed_params = {}
            for key in self.param_widgets:
                widget = self.param_widgets[key]
                text = widget.line_edit.text().strip()
                if key == 'n_unit':
                    n_units = self.validate_n_units(text)
                    if n_units is None:
                        return
                    fixed_params[key] = n_units
                else:
                    try:
                        fixed_params[key] = float(text)
                    except ValueError:
                        QMessageBox.critical(self, "错误", f"参数 {widget.label.text()} 输入无效")
                        return

            var_params = []
            for row in range(self.table.rowCount()):
                try:
                    params = {
                        'Kc': float(self.table.item(row, 0).text()),
                        'Loss_perunit': float(self.table.item(row, 1).text()),
                        'Freq': float(self.table.item(row, 2).text()),
                        'Vpc': float(self.table.item(row, 3).text())
                    }
                    var_params.append(params)
                except:
                    QMessageBox.warning(self, "格式错误", f"第{row+1}行数据不完整或格式错误")
                    return

            save_dir = os.path.join(os.getcwd(), "Results")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"Result_{timestamp}.csv")

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("Frequency(GHz),Combined_Gain(dB),Kc(Ohm),Loss_perunit,Vpc(c),N_Units\n")
                
                current_results = []
                for p in var_params:
                    total_gain = 0.0
                    accumulated_ab = 0.0
                    A_intbeamloss=0.0
                    n_units = fixed_params['n_unit']
                    
                    for idx, n in enumerate(n_units):
                        full_args = [
                            fixed_params['i'],
                            fixed_params['v'],
                            p['Kc'],
                            p['Loss_perunit'],
                            fixed_params['p_sws'],
                            n,
                            fixed_params['w'],
                            fixed_params['t'],
                            fixed_params['Fn_K'],
                            p['Freq'],
                            p['Vpc']
                        ]
                        
                        try:
                            result = detailed_calculation(*full_args)
                        except ValueError as e:
                            QMessageBox.critical(self, "计算错误", 
                                f"参数：{full_args}\n错误：{str(e)}")
                            return
                        
                        total_gain += result['线性最大增益Gmax']
                        A_intbeamloss=result['初始化调制增益降低量A']
                        if idx < len(n_units) - 1:
                                accumulated_ab += result['衰减降低增益量Ab']
                    
                    final_gain =total_gain + accumulated_ab+A_intbeamloss
                    current_results.append((p['Freq'], final_gain))
                    f.write(f"{p['Freq']},{final_gain},{p['Kc']},{p['Loss_perunit']},{p['Vpc']},\"{n_units}\"\n")

                self.history_results.append((current_results, fixed_params, f"N={n_units}"))
                self.plot_counter += 1

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            for data, params, label in self.history_results:
                freqs, gains = zip(*sorted(data, key=lambda x: x[0]))
                ax.plot(freqs, gains, 'o-', markersize=4, label=label)
            
            ax.set_title(f"Combined-Gain\n(Time {timestamp.replace('_', ' ')})")
            ax.set_xlabel("Frequncy (GHz)")
            ax.set_ylabel("Combined-Gain (dB)")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            self.canvas.draw()

            QMessageBox.information(self, "完成", 
                                  f"成功计算{len(var_params)}组数据\n结果已保存至：\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生错误：\n{str(e)}")

    def add_table_row(self):
        self.table.insertRow(self.table.rowCount())

    def delete_table_row(self):
        current_row = self.table.currentRow()
        if current_row == -1:
            if self.table.rowCount() > 0:
                self.table.removeRow(self.table.rowCount()-1)
        else:
            self.table.removeRow(current_row)

    def import_csv(self):
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
                        for col in range(4):
                            self.table.setItem(row_num, col, QTableWidgetItem(row[col].strip()))
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败：{str(e)}")

    def clear_plot(self):
        self.history_results.clear()
        self.plot_counter = 1
        self.figure.clear()
        self.canvas.draw()

    def export_data(self):
        if not self.history_results:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存数据文件", "", "数据文件 (*.dat)")
        if path:
            with open(path, 'w') as f:
                for idx, (data, params, label) in enumerate(self.history_results):
                    f.write(f"# Parameters={{\n")
                    for key, value in params.items():
                        if key == 'n_unit':
                            f.write(f"#\t{key}={list(value)},\n")
                        else:
                            f.write(f"#\t{key}={value},\n")
                    f.write(f"# }}\n")
                    
                    f.write(f"# Frequency(GHz)\tGain(dB)\n")
                    f.write("# ----------------------------------------\n")
                    for freq, gain in sorted(data, key=lambda x: x[0]):
                        f.write(f"{freq:.2f}\t{gain:.2f}\n")
                    f.write("\n\n")

            QMessageBox.information(self, "导出成功", f"数据已保存至：\n{path}")

    def save_config(self):
        config = {
            "fixed": {k: w.line_edit.text() for k, w in self.param_widgets.items()},
            "variables": [[self.table.item(row, col).text() for col in range(4)] 
                         for row in range(self.table.rowCount())]
        }
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        path = os.path.join(config_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, 'w') as f:
            json.dump(config, f)
        QMessageBox.information(self, "保存成功", f"配置已保存至：\n{path}")

    def load_last_config(self):
        config_dir = "config"
        try:
            configs = [f for f in os.listdir(config_dir) if f.endswith(".json")]
            if configs:
                latest = max(configs, key=lambda f: os.path.getctime(os.path.join(config_dir, f)))
                with open(os.path.join(config_dir, latest), 'r') as f:
                    config = json.load(f)
                    for key, value in config["fixed"].items():
                        self.param_widgets[key].line_edit.setText(value)
                    self.table.setRowCount(0)
                    for row in config["variables"]:
                        row_num = self.table.rowCount()
                        self.table.insertRow(row_num)
                        for col, value in enumerate(row[:4]):
                            self.table.setItem(row_num, col, QTableWidgetItem(value))
        except:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TWTCalculator()
    ex.show()
    sys.exit(app.exec_())