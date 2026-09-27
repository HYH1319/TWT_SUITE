import sys
import numpy as np
import math as m
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit,
                            QGroupBox, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class BeamCalculatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('束流参数计算器')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中心widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 输入参数组
        input_group = QGroupBox("输入参数")
        input_layout = QGridLayout()
        
        # 束流宽度
        input_layout.addWidget(QLabel("束流宽度 (mm):"), 0, 0)
        self.w_input = QLineEdit()
        self.w_input.setPlaceholderText("请输入束流宽度")
        input_layout.addWidget(self.w_input, 0, 1)
        
        # 束流厚度
        input_layout.addWidget(QLabel("束流厚度 (mm):"), 1, 0)
        self.t_input = QLineEdit()
        self.t_input.setPlaceholderText("请输入束流厚度")
        input_layout.addWidget(self.t_input, 1, 1)
        
        # 工作频率
        input_layout.addWidget(QLabel("工作频率 (GHz):"), 2, 0)
        self.f0_input = QLineEdit()
        self.f0_input.setPlaceholderText("请输入工作频率")
        input_layout.addWidget(self.f0_input, 2, 1)
        
        # 束流电压
        input_layout.addWidget(QLabel("束流电压:"), 3, 0)
        self.Ue_input = QLineEdit()
        self.Ue_input.setPlaceholderText("请输入束流电压")
        input_layout.addWidget(self.Ue_input, 3, 1)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # 计算按钮
        self.calc_button = QPushButton("开始计算")
        self.calc_button.clicked.connect(self.calculate_step1)
        self.calc_button.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        main_layout.addWidget(self.calc_button)
        
        # 结果显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        main_layout.addWidget(self.result_text)
        
        # 填充校正因子输入（初始隐藏）
        self.fill_group = QGroupBox("填充校正因子")
        fill_layout = QHBoxLayout()
        fill_layout.addWidget(QLabel("请输入正确的填充校正因子:"))
        self.fill_input = QLineEdit()
        self.fill_input.setPlaceholderText("根据查表结果输入")
        fill_layout.addWidget(self.fill_input)
        
        self.fill_button = QPushButton("继续计算")
        self.fill_button.clicked.connect(self.calculate_step2)
        fill_layout.addWidget(self.fill_button)
        
        self.fill_group.setLayout(fill_layout)
        self.fill_group.setVisible(False)
        main_layout.addWidget(self.fill_group)
        
        # 存储中间计算结果
        self.intermediate_results = {}
        
    def Vpc_calc(self, U):
        """计算Vp_c，保持原逻辑不变"""
        gamma = U / 5.11e5
        Vp_c = 1 * m.pow(1 - 1 / m.pow(1 + gamma, 2), 0.5)
        return Vp_c
    
    def calculate_step1(self):
        """第一步计算，获取基本参数并计算gamma0*t"""
        try:
            # 获取输入值
            w = float(self.w_input.text()) * 1e-3
            t = float(self.t_input.text()) * 1e-3
            f0_Hz = float(self.f0_input.text()) * 1e9
            Ue = float(self.Ue_input.text())
            
            # 执行计算（保持原逻辑）
            Vec = self.Vpc_calc(Ue)
            omega = 2 * m.pi * f0_Hz
            beta_e = omega / (Vec * 299792458)
            K_wave = omega / 299792458
            gamma0 = np.sqrt(beta_e**2 - K_wave**2)
            
            # 存储中间结果
            self.intermediate_results = {
                'w': w, 't': t, 'f0_Hz': f0_Hz, 'Ue': Ue,
                'Vec': Vec, 'omega': omega, 'beta_e': beta_e,
                'K_wave': K_wave, 'gamma0': gamma0
            }
            
            # 显示结果
            result_text = f"输入参数:\n"
            result_text += f"束流宽度: {w*1e3} mm\n"
            result_text += f"束流厚度: {t*1e3} mm\n"
            result_text += f"工作频率: {f0_Hz/1e9} GHz\n"
            result_text += f"束流电压: {Ue}\n\n"
            
            result_text += f"请根据此参数查表得到正确的填充校正因子: {gamma0*t}\n"
            
            self.result_text.setText(result_text)
            
            # 显示填充校正因子输入框
            self.fill_group.setVisible(True)
            self.fill_input.setFocus()
            
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", "请输入有效的数值！")
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算过程中出现错误：{str(e)}")
    
    def calculate_step2(self):
        """第二步计算，使用填充校正因子完成计算"""
        try:
            # 获取填充校正因子
            Fill_Rate_Ture = float(self.fill_input.text())
            
            # 获取存储的中间结果
            results = self.intermediate_results
            w = results['w']
            t = results['t']
            beta_e = results['beta_e']
            
            # 继续计算（保持原逻辑）
            
            Fn_tmp = np.sqrt((np.pi / w) ** 2 + (np.pi / t) ** 2)
            Fn_FullFill = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
            Fn_tureFix = Fn_FullFill * Fill_Rate_Ture  # 填充矫正Fn
            
            # 带状束流Rowe特征值R介于[Fn_tureFix,np.sqrt(Fn_tureFix)]之间,但没有解析方法，近似数值计算
            Fn_needFix = (Fn_tureFix * (Fn_tureFix**0.5)) ** 0.5
            Rowe_Sheetbeam = Fn_needFix  # 用一个修正的Fn_needFix替代Rowe_Sheetbeam
            
            Fill_Rate_need = Fn_needFix / Fn_FullFill
            
            # 显示完整结果
            result_text = self.result_text.toPlainText()
            result_text += f"\n输入的填充校正因子: {Fill_Rate_Ture}\n\n"
            result_text += f"填充校正后绝对正确的Fn_TureFIX={Fn_tureFix}\n\n"
            result_text += f"大信号分析需要的Rowe特征值Rowe_Sheetbeam={Rowe_Sheetbeam}\n\n"
            result_text += f"大信号分析需要填充因子Fill_Rate_need={Fill_Rate_need}\n"
            
            self.result_text.setText(result_text)
            
            # 隐藏填充校正因子输入框
            self.fill_group.setVisible(False)
            
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", "请输入有效的填充校正因子！")
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算过程中出现错误：{str(e)}")

def main():
    app = QApplication(sys.argv)
    calculator = BeamCalculatorGUI()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
