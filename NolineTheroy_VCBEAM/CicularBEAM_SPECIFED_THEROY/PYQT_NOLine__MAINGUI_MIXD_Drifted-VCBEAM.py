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
import numpy as np
from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER_MIX_Drift import calculate_SEGMENT_TWT_NOLINE

class ParamWidget(QWidget):
    def __init__(self, key, label, param_range, step_type, parent=None):
        super().__init__(parent)
        self.key = key
        self.min_val, self.max_val = param_range
        self.step_type = step_type
        self.is_array = (key == 'n_unit')  # 数组类型特殊处理
        self.layout = QHBoxLayout(self)
        
        # 初始化控件
        self.label = QLabel(label)
        self.line_edit = QLineEdit()
        self.slider = QSlider(Qt.Horizontal) if not self.is_array else None
        
        # 构建布局
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.line_edit)
        if self.slider:
            self.layout.addWidget(self.slider)
        
        # 初始化数值系统
        self.base_value = self.initialize_base_value()
        self.init_control()
        self.setup_connections()

    def initialize_base_value(self):
        """获取并验证初始基准值"""
        if self.is_array:
            return None
        
        # 尝试读取已有输入
        if self.line_edit.text():
            try:
                val = float(self.line_edit.text())
                return max(self.min_val, min(val, self.max_val)) if self.max_val else val
            except:
                pass
                
        # 默认值策略
        if self.min_val >= 0 and self.max_val:
            return (self.min_val + self.max_val) / 2
        return self.min_val

    def init_control(self):
        """初始化控件状态"""
        if self.is_array:
            self.line_edit.setPlaceholderText("逗号分隔，如：20,30,40")
            return
        
        # 电压特殊处理
        if self.key == 'v':
            self.init_voltage_control()
        else:
            self.init_percent_control()
        
        self.update_display()

    def init_voltage_control(self):
        """电压参数初始化逻辑"""
        # 计算合理的滑块范围
        max_slider_val = int((self.max_val if self.max_val else 40000) / 100)
        self.slider.setRange(0, max_slider_val)
        self.slider.setValue(int(self.base_value / 100))
        self.slider.setSingleStep(1)  # 每步100V

    def init_percent_control(self):
        """百分比参数初始化逻辑"""
        self.slider.setRange(-50, 50)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)  # 每步1%

    def setup_connections(self):
        """信号连接"""
        if self.slider:
            self.slider.valueChanged.connect(self.slider_changed)
        self.line_edit.editingFinished.connect(self.text_changed)

    def slider_changed(self, value):
        """处理滑块变化事件"""
        if self.key == 'v':
            actual_value = value * 100  # 电压线性模式
        else:
            # 百分比模式计算
            actual_value = self.base_value * (1 + value/100)
            actual_value = max(self.min_val, min(actual_value, self.max_val)) if self.max_val else actual_value
        
        self.update_line_edit(actual_value)

    def text_changed(self):
        """处理文本输入变化"""
        if self.is_array or not self.slider:
            return
        
        # 获取并验证新值
        try:
            new_value = float(self.line_edit.text())
        except ValueError:
            QMessageBox.warning(self.parent(), "输入错误", "请输入有效数字")
            self.line_edit.setText(f"{self.base_value:.3f}")
            return
        
        # 范围验证
        error_msg = None
        if self.min_val is not None and new_value < self.min_val:
            error_msg = f"数值不能小于{self.min_val}"
        elif self.max_val and new_value > self.max_val:
            error_msg = f"数值不能超过{self.max_val}"
        
        if error_msg:
            QMessageBox.warning(self.parent(), "范围错误", error_msg)
            self.line_edit.setText(f"{self.base_value:.3f}")
            return
        
        # 更新基准值并重置滑块
        self.base_value = new_value
        if self.key == 'v':
            self.slider.setValue(int(new_value / 100))
        else:
            self.slider.setValue(0)  # 百分比模式归零
        
        self.update_display()

    def update_line_edit(self, value):
        """更新文本框显示"""
        self.line_edit.blockSignals(True)
        
        # 智能格式显示
        if abs(value) >= 1e4 or 0 < abs(value) < 0.01:
            display_text = f"{value:.4e}"
        elif value == int(value):
            display_text = f"{int(value)}"
        else:
            display_text = f"{value:.4f}"
            
        self.line_edit.setText(display_text)
        self.line_edit.blockSignals(False)

    def update_display(self):
        """统一更新显示状态"""
        self.update_line_edit(self.base_value)

class TWTCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.param_ranges = {
            'Lenth_att': (0, None),  # 衰减段数量
            'Loss_attu': (0, 100),   # 衰减量 (dB)
            'i': (0, 1e0), 'v': (0, 4e4), 
            'p_sws': (0, None), 'n_unit': (1, 1000),
            'w': (0, None), 't': (0, None),
            'Kc': (0, None), 'Loss_perunit': (0, None),
            'Freq': (0.1, None), 'Vpc': (0.1, 0.99),
            'Fn_K': (0, None),
            'P_in': (0, None)  # 输入功率
        }

        self.step_types = {
            'Lenth_att': 'linear',
            'Loss_attu': 'linear',
            'i': 'percent',
            'v': 'linear',  # 仅电压保持线性
            'Fn_K': 'percent',
            'p_sws': 'percent',
            'n_unit': None,
            'w': 'percent',
            't': 'percent',
            'P_in': 'percent'
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
        
        # 修正的参数列表，添加了Loss_attu和P_in
        params = [
            ('i', '电流 I (A)'), 
            ('v', '电压 V (V)'),
            ('Lenth_att', '慢波分段 Lenth_att (个),\n例如其为2时，n_unit形式应为互作用段长度-衰减漂移段长度-互作用段长度'),
            ('Loss_attu', '衰减量 Loss_attu (dB)'),    # 添加的缺失参数
            ('p_sws', '周期长度 p_SWS (mm)'), 
            ('n_unit', '周期数 N_Unit（逗号分隔）'),
            ('w', '束流宽度 w (mm)'), 
            ('t', '束流厚度 t (mm)'),
            ('Fn_K', 'Fn_K 参数'),
            ('P_in', '输入功率 P_in (W)')             # 添加的输入功率参数
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
                f.write("Frequency(GHz),Output_Power(W),Kc(Ohm),Loss_perunit,Vpc(c),N_Units,Lenth_att,Loss_attu,P_in\n")
                
                current_results = []
                for p in var_params:
                    # 构建计算函数所需的参数
                    args = [
                        fixed_params['i'],    # I
                        fixed_params['v'],    # V
                        p['Kc'],              # Kc
                        p['Loss_perunit'],            # Loss_perunit
                        fixed_params['Lenth_att'],  # Lenth_att
                        fixed_params['p_sws'],      # p_SWS
                        fixed_params['n_unit'],     # N_unit (数组)
                        fixed_params['w'],          # w
                        fixed_params['t'],          # t
                        fixed_params['Fn_K'],       # Fn_K
                        p['Freq'],             # f0_GHz
                        p['Vpc'],              # Vpc
                        fixed_params['P_in'],       # P_in (新添加)
                        fixed_params['Loss_attu']   # Loss_attu (新添加)
                    ]
                    
                    # 执行计算
                    try:
                        # 对每个分段进行计算（这里使用原始计算函数）
                        n_units = fixed_params['n_unit']
                        num_segments = len(n_units)
                        
                        # 为简化示例，这里只调用一次计算函数
                        # 实际应用中应该为每个分段单独计算
                        result = calculate_SEGMENT_TWT_NOLINE(*args)
                        
                        # 获取输出功率
                        P_Out = result["输出功率P_out"]
                        
                    except Exception as e:
                        QMessageBox.critical(self, "计算错误", 
                            f"参数：{args}\n错误：{str(e)}")
                        return

                    # 记录结果
                    current_results.append((p['Freq'], P_Out))
                    
                    # 写入CSV
                    n_units_str = ",".join(str(int(n)) for n in fixed_params['n_unit'])
                    f.write(f"{p['Freq']},{P_Out},{p['Kc']},{p['Loss_perunit']},{p['Vpc']},\"{n_units_str}\",{fixed_params['Lenth_att']},{fixed_params['Loss_attu']},{fixed_params['P_in']}\n")
                
                # 保存当前结果用于绘图
                self.history_results.append((current_results, fixed_params, f"N={fixed_params['n_unit']}"))
                self.plot_counter += 1

            # 更新图表
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            for data, params, label in self.history_results:
                freqs, powers = zip(*sorted(data, key=lambda x: x[0]))
                ax.plot(freqs, powers, 'o-', markersize=4, label=label)
            
            ax.set_title(f"Output Power vs Frequency\n(Time {timestamp.replace('_', ' ')})")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Output Power (W)")
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
                    
                    f.write(f"# Frequency(GHz)\tOutput Power(W)\n")
                    f.write("# ----------------------------------------\n")
                    for freq, power in sorted(data, key=lambda x: x[0]):
                        f.write(f"{freq:.4f}\t{power:.4f}\n")
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