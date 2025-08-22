import sys
import math as m
import numpy as np
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGridLayout, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QTabWidget, QFileDialog, QMessageBox)
from PyQt5.QtGui import QDoubleValidator, QValidator
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.figure import Figure
from Calc_Beam_EquipErb_FUN import calculate_dielectric



# 物理常量
YITA = 1.7588e11       # 电子电荷质量比
ERB = 8.854e-12        # 真空介电常数 (F/m)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("波导介电常数计算器")
        self.setGeometry(100, 100, 900, 700)
        
        # 主部件和布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # 标签页
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # 创建参数输入页面
        self.create_input_tab()
        
        # 创建结果页面
        self.create_results_tab()
        
        # 初始化参数
        self.params = {
            'I': 0.30,
            'U': 23000,
            'f0_GHz': 216,
            'Fill_Rate': 1,
            'w_beam': 0.20,
            't_beam': 0.20
        }
        
        # 初始化输入字段
        self.init_input_fields()
        
        # 状态栏
        self.statusBar()
        
        # 为表格添加初始示例数据
        self.add_example_data()
        
    def add_example_data(self):
        """添加初始示例数据到表格"""
        self.table.setRowCount(5)
        A_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        u_values = [0.05, 0.15, 0.25, 0.35, 0.45]
        
        for i in range(5):
            self.table.setItem(i, 0, QTableWidgetItem(str(A_values[i])))
            self.table.setItem(i, 1, QTableWidgetItem(str(u_values[i])))
    
    def create_input_tab(self):
        """创建输入参数标签页"""
        self.input_tab = QWidget()
        self.tabs.addTab(self.input_tab, "参数输入")
        
        tab_layout = QVBoxLayout(self.input_tab)
        
        # 参数输入组
        param_group = QGroupBox("基本参数")
        tab_layout.addWidget(param_group)
        
        # 使用网格布局组织输入字段
        param_layout = QGridLayout(param_group)
        
        # 创建标签和输入字段
        self.param_fields = {}
        
        param_definitions = [
            ('I', '电流 (A)', 0.01, 100),
            ('U', '电压 (V)', 100, 50000),
            ('f0_GHz', '频率 (GHz)', 1, 2000),
            ('Fill_Rate', '填充率倒数', 0.1, 100),
            ('w_beam', '束宽 (mm)', 0.01, 10),
            ('t_beam', '束厚 (mm)', 0.01, 10)
        ]
        
        for i, (key, label, min_val, max_val) in enumerate(param_definitions):
            param_layout.addWidget(QLabel(label), i, 0)
            self.param_fields[key] = QLineEdit()
            # 设置验证器
            validator = QDoubleValidator(min_val, max_val, 6)
            validator.setNotation(QDoubleValidator.StandardNotation)
            self.param_fields[key].setValidator(validator)
            param_layout.addWidget(self.param_fields[key], i, 1)
        
        # A和u表
        arrays_group = QGroupBox("可变参数 (A和u数组)")
        tab_layout.addWidget(arrays_group)
        arrays_layout = QVBoxLayout(arrays_group)
        
        self.table = QTableWidget(10, 2)  # 初始10行
        self.table.setHorizontalHeaderLabels(["A", "u"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 添加按钮
        btn_layout = QHBoxLayout()
        
        add_row_btn = QPushButton("添加行")
        add_row_btn.clicked.connect(lambda: self.table.insertRow(self.table.rowCount()))
        btn_layout.addWidget(add_row_btn)
        
        remove_row_btn = QPushButton("删除行")
        remove_row_btn.clicked.connect(self.remove_selected_row)
        btn_layout.addWidget(remove_row_btn)
        
        import_csv_btn = QPushButton("导入CSV")
        import_csv_btn.clicked.connect(self.import_csv)
        btn_layout.addWidget(import_csv_btn)
        
        clear_btn = QPushButton("清除数据")
        clear_btn.clicked.connect(self.clear_table)
        btn_layout.addWidget(clear_btn)
        
        arrays_layout.addWidget(self.table)
        arrays_layout.addLayout(btn_layout)
        
        # 计算按钮
        self.calculate_btn = QPushButton("计算")
        self.calculate_btn.clicked.connect(self.calculate)
        tab_layout.addWidget(self.calculate_btn)
    
    def create_results_tab(self):
        """创建结果展示标签页"""
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "计算结果")
        
        results_layout = QVBoxLayout(self.results_tab)
        
        # 结果图表
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)
        
        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["序号", "A", "u", "ε_res"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        # 导出按钮
        export_btn = QPushButton("导出结果到CSV")
        export_btn.clicked.connect(self.export_results)
        results_layout.addWidget(export_btn)
    
    def init_input_fields(self):
        """使用默认值初始化输入字段"""
        for key, value in self.params.items():
            self.param_fields[key].setText(str(value))
    
    def collect_params(self):
        """从输入字段收集参数"""
        params = {}
        for key, field in self.param_fields.items():
            try:
                text = field.text()
                if text:
                    value = float(text)
                    # 检查验证器范围
                    validator = field.validator()
                    if validator and validator.validate(text, 0)[0] == QValidator.Acceptable:
                        params[key] = value
                    else:
                        # 超出范围时使用默认值
                        min_val = validator.bottom()
                        max_val = validator.top()
                        value = min(max(value, min_val), max_val)  # 截断到范围
                        field.setText(str(value))
                        params[key] = value
                else:
                    # 空输入使用默认值
                    params[key] = self.params[key]
                    field.setText(str(self.params[key]))
            except ValueError:
                # 输入无效时使用默认值
                params[key] = self.params[key]
                field.setText(str(self.params[key]))
        return params
    
    def collect_arrays(self):
        """从表格中收集A和u值"""
        A_list = []
        u_list = []
        invalid_rows = []
        
        for row in range(self.table.rowCount()):
            a_item = self.table.item(row, 0)
            u_item = self.table.item(row, 1)
            valid_row = True
            
            try:
                if a_item and a_item.text():
                    a_value = float(a_item.text())
                else:
                    valid_row = False
                
                if u_item and u_item.text():
                    u_value = float(u_item.text())
                else:
                    valid_row = False
                
                if valid_row:
                    A_list.append(a_value)
                    u_list.append(u_value)
                else:
                    invalid_rows.append(row)
            except ValueError:
                invalid_rows.append(row)
        
        # 标记无效行
        for row in range(self.table.rowCount()):
            for col in range(2):  # 只有两列
                item = self.table.item(row, col)
                if row in invalid_rows:
                    if item:
                        item.setBackground(Qt.red)
                elif item:
                    item.setBackground(Qt.white)
        
        return A_list, u_list
    
    def calculate(self):
        """执行计算并显示结果"""
        # 收集输入参数
        self.params = self.collect_params()
        A_list, u_list = self.collect_arrays()
        
        # 验证输入
        if not A_list or not u_list:
            self.show_error("错误：A和u数组不能为空！")
            return
            
        if len(A_list) != len(u_list):
            self.show_error("错误：A和u数组长度必须相同！")
            return
        
        # 执行计算
        try:
            results = calculate_dielectric(self.params, A_list, u_list)
        except Exception as e:
            self.show_error(f"计算错误: {str(e)}")
            return
        
        # 更新结果标签页
        self.update_results_tab(results)
        self.tabs.setCurrentIndex(1)  # 切换到结果标签页
    
    def update_results_tab(self, results):
        """更新结果标签页的内容"""
        # 清除图表
        self.figure.clear()
        
        # 绘制图表
        ax = self.figure.add_subplot(111)
        
        # 绘制相对介电常数随A的变化
        ax.plot(results['A'], results['epsilon_rs'], 'b-', marker='o', label='ε_res')
        ax.set_xlabel('A')
        ax.set_ylabel(' ε_res')
        ax.set_title('ε_res')
        ax.grid(True)
        ax.legend()
        
        self.canvas.draw()
        
        # 更新结果表格
        n_points = len(results['A'])
        self.results_table.setRowCount(n_points)
        
        # 设置表头工具提示
        self.results_table.horizontalHeaderItem(0).setToolTip("数据点索引")
        self.results_table.horizontalHeaderItem(1).setToolTip("参数A值")
        self.results_table.horizontalHeaderItem(2).setToolTip("参数u值")
        self.results_table.horizontalHeaderItem(3).setToolTip("计算得到的相对介电常数")
        
        for i in range(n_points):
            # 点号
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            # A值
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{results['A'][i]:.4f}"))
            
            # u值
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{results['u'][i]:.4f}"))
            
            # ε_res值
            epsilon_value = results['epsilon_rs'][i]
            item = QTableWidgetItem(f"{epsilon_value:.4f}")
            
            # 标记异常值
            if abs(epsilon_value) > 100:
                item.setBackground(Qt.red)  # 高亮红色
            elif abs(epsilon_value) > 10:
                item.setBackground(Qt.yellow)  # 高亮黄色
                
            self.results_table.setItem(i, 3, item)
        
        # 添加一些关键值到表格顶部
        self.results_table.insertRow(0)
        self.results_table.setItem(0, 0, QTableWidgetItem("Wp"))
        self.results_table.setItem(0, 1, QTableWidgetItem(f"{results['Wp']:.4e}"))
    
    def import_csv(self):
        """从CSV文件导入A和u值"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开CSV文件", "", "CSV文件 (*.csv)"
        )
        
        if not filename:
            return
        
        try:
            A_values = []
            u_values = []
            
            with open(filename, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                for row_num, row in enumerate(csvreader, 1):
                    if len(row) >= 2:
                        try:
                            A_value = float(row[0])
                            u_value = float(row[1])
                            A_values.append(A_value)
                            u_values.append(u_value)
                        except ValueError:
                            self.show_error(f"警告：跳过无效数据行 {row_num}: {row}")
            
            if not A_values or not u_values:
                self.show_error("错误：CSV文件没有有效数据")
                return
            
            # 清除现有表格
            self.table.setRowCount(len(A_values))
            
            # 填充新数据
            for i, (A_val, u_val) in enumerate(zip(A_values, u_values)):
                self.table.setItem(i, 0, QTableWidgetItem(str(A_val)))
                self.table.setItem(i, 1, QTableWidgetItem(str(u_val)))
            
            self.statusBar().showMessage(f"成功导入 {len(A_values)} 行数据", 3000)
        
        except Exception as e:
            self.show_error(f"导入错误: {str(e)}")
    
    def export_results(self):
        """导出结果到CSV文件"""
        if self.results_table.rowCount() <= 1:  # 只有标题行
            self.show_error("没有可导出的结果数据")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "CSV文件 (*.csv)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                # 写入标题行
                csvwriter.writerow(['等离子体频率Wp', 'A', 'u', '相对介电常数 ε_res'])
                
                # 写入数据行（跳过第一行的摘要信息）
                for row in range(1, self.results_table.rowCount()):
                    index_item = self.results_table.item(row, 0)
                    A_item = self.results_table.item(row, 1)
                    u_item = self.results_table.item(row, 2)
                    epsilon_item = self.results_table.item(row, 3)
                    
                    if all([index_item, A_item, u_item, epsilon_item]):
                        csvwriter.writerow([
                            index_item.text(),
                            A_item.text(),
                            u_item.text(),
                            epsilon_item.text()
                        ])
            
            self.statusBar().showMessage(f"结果已成功导出到 {filename}", 5000)
        except Exception as e:
            self.show_error(f"导出错误: {str(e)}")
    
    def clear_table(self):
        """清除表格数据"""
        self.table.setRowCount(0)
        self.table.setRowCount(5)  # 保留5行空白行
    
    def remove_selected_row(self):
        """删除选中的行"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
    
    def show_error(self, message):
        """显示错误消息"""
        QMessageBox.critical(self, "错误", message)
        self.statusBar().showMessage(message, 5000)  # 同时在状态栏显示5秒

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())