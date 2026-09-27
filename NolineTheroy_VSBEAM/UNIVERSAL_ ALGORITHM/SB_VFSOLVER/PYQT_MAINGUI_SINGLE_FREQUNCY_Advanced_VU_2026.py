#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行波管计算器 — PyQt5 现代化重构版
功能：参数配置保存/自动加载、实时计算日志、矩阵独立Sheet导出Excel
"""

import sys
import os
import json
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QTabWidget,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QHeaderView,
    QAbstractItemView,
    QAction,
    QToolBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QIcon, QColor

import matplotlib

matplotlib.use("Qt5Agg")
# ====== 解决中文字体缺失警告 ======
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",  # Windows 微软雅黑
    "PingFang SC",  # macOS 苹方
    "Heiti TC",  # macOS 黑体
    "WenQuanYi Micro Hei",  # Linux 文泉驿
    "Noto Sans CJK SC",  # Linux 思源
    "SimHei",  # Windows 黑体备用
    "DejaVu Sans",  # 最终回退
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 修复负号显示
# 👇 这里调整全局默认字号
matplotlib.rcParams["font.size"] = 15  # 全局基础字号
matplotlib.rcParams["axes.titlesize"] = 15  # 标题字号
matplotlib.rcParams["axes.labelsize"] = 15  # 坐标轴标签字号
matplotlib.rcParams["xtick.labelsize"] = 15  # x轴刻度字号
matplotlib.rcParams["ytick.labelsize"] = 15  # y轴刻度字号
matplotlib.rcParams["legend.fontsize"] = 15  # 图例字号


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from datetime import datetime

from PYQT_MAINGUI_SINGLE_FREQUNCY_AdvancedCALC_VU_2026 import TWTCalculator


# ═══════════════════════════════════════════════════════════════
# 现代化深色 QSS 主题
# ═══════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow {
    background-color: #f5f6fa;
}
QWidget {
    background-color: #f5f6fa;
    color: #2c3e50;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #bdc3c7;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #2980b9;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 4px 8px;
    color: #2c3e50;
    selection-background-color: #3498db;
    selection-color: #ffffff;
}
QLineEdit:focus {
    border: 2px solid #3498db;
}
QPushButton {
    background-color: #ecf0f1;
    border: 1px solid #bdc3c7;
    border-radius: 5px;
    padding: 6px 16px;
    color: #2c3e50;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #d5dbdb;
    border: 1px solid #3498db;
}
QPushButton:pressed {
    background-color: #3498db;
    color: #ffffff;
}
QPushButton#calcBtn {
    background-color: #27ae60;
    color: #ffffff;
    font-size: 14px;
    border: none;
}
QPushButton#calcBtn:hover {
    background-color: #2ecc71;
}
QPushButton#resetBtn {
    background-color: #e74c3c;
    color: #ffffff;
    border: none;
}
QPushButton#resetBtn:hover {
    background-color: #c0392b;
}
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f8f9fa;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    gridline-color: #dfe6e9;
    selection-background-color: #3498db;
    selection-color: #ffffff;
}
QTableWidget::item {
    padding: 2px;
}
QHeaderView::section {
    background-color: #dfe6e9;
    color: #2c3e50;
    border: 1px solid #bdc3c7;
    padding: 4px;
    font-weight: bold;
}
QTabWidget::pane {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: #ffffff;
}
QTabBar::tab {
    background-color: #ecf0f1;
    border: 1px solid #bdc3c7;
    padding: 8px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    color: #7f8c8d;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #2980b9;
    font-weight: bold;
}
QTabBar::tab:hover {
    background-color: #d5dbdb;
}
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    color: #2c3e50;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 4px;
}
QStatusBar {
    background-color: #ecf0f1;
    color: #2c3e50;
    border-top: 1px solid #bdc3c7;
}
QComboBox {
    background-color: #ffffff;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 4px 8px;
    color: #2c3e50;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #2c3e50;
    selection-background-color: #3498db;
    selection-color: #ffffff;
}
QToolBar {
    background-color: #ecf0f1;
    border-bottom: 1px solid #bdc3c7;
    spacing: 6px;
    padding: 4px;
}
QSplitter::handle {
    background-color: #bdc3c7;
    width: 3px;
}
QScrollBar:vertical {
    background-color: #f5f6fa;
    width: 10px;
}
QScrollBar::handle:vertical {
    background-color: #bdc3c7;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #95a5a6;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QToolTip {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #bdc3c7;
    padding: 4px;
}
"""


# ═══════════════════════════════════════════════════════════════
# 辅助：表格中放置 ComboBox 的委托
# ═══════════════════════════════════════════════════════════════
class ComboBoxDelegate(QComboBox):
    """嵌入 QTableWidget 的下拉框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItems(["initial", "attenuator", "O"])
        self.setFocusPolicy(Qt.StrongFocus)

    def get_value(self):
        return self.currentText()

    def set_value(self, val):
        idx = self.findText(val)
        if idx >= 0:
            self.setCurrentIndex(idx)
        else:
            self.setCurrentText(val)


# ═══════════════════════════════════════════════════════════════
# 计算线程 — 防止 GUI 卡死
# ═══════════════════════════════════════════════════════════════
class CalcWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, common_params, segments, loss_attu):
        super().__init__()
        self.common_params = common_params
        self.segments = segments
        self.loss_attu = loss_attu

    def run(self):
        try:
            result = TWTCalculator.calculate(
                self.common_params,
                self.segments,
                self.loss_attu,
                log_callback=lambda msg: self.log_signal.emit(msg),
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════
class TWTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 行波管 (TWT) 非线性计算平台")
        self.setGeometry(80, 60, 1500, 900)
        self.last_calc_result = None
        self.calc_thread = None
        self.calc_worker = None

        self._init_ui()
        self._create_toolbar()
        self._load_defaults()
        self._load_config(silent=True)
        self.statusBar().showMessage("就绪")

    # ─────────────────── UI 构建 ───────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ========== 左侧参数面板 ==========
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_panel.setFixedWidth(520)

        # 全局参数
        global_group = QGroupBox("⚙ 全局参数")
        gl = QVBoxLayout(global_group)
        gl.setSpacing(6)

        # 用网格布局排列参数
        grid_widget = QWidget()
        grid = QHBoxLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        col1 = QVBoxLayout()
        col2 = QVBoxLayout()
        col3 = QVBoxLayout()

        self.global_vars = {}
        params_def = [
            ("电流 I (A)", "I", 0.3),
            ("电压 V (V)", "V", 23000),
            ("频率 f0 (GHz)", "f0", 211),
            ("宽度 w (mm)", "w", 0.2),
            ("厚度 t (mm)", "t", 0.2),
            ("输入功率 P_in (W)", "P_in", 0.10),
            ("Loss_attu", "Loss_attu", 0),
        ]

        for i, (label_text, key, default) in enumerate(params_def):
            lbl = QLabel(label_text)
            lbl.setFixedHeight(20)
            edit = QLineEdit(str(default))
            edit.setFixedHeight(28)
            edit.setAlignment(Qt.AlignCenter)
            self.global_vars[key] = edit

            if i < 3:
                col1.addWidget(lbl)
                col1.addWidget(edit)
            elif i < 5:
                col2.addWidget(lbl)
                col2.addWidget(edit)
            else:
                col3.addWidget(lbl)
                col3.addWidget(edit)

        grid.addLayout(col1)
        grid.addLayout(col2)
        grid.addLayout(col3)
        gl.addWidget(grid_widget)
        left_layout.addWidget(global_group)

        # 分段参数表格
        seg_group = QGroupBox("📊 分段参数")
        sl = QVBoxLayout(seg_group)
        sl.setSpacing(4)

        self.seg_table = QTableWidget()
        self.seg_table.setColumnCount(7)
        self.seg_table.setHorizontalHeaderLabels(
            ["周期数", "Vpc", "螺距", "耦合阻抗", "每单元损耗", "填充因子", "类型"]
        )
        self.seg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.seg_table.setAlternatingRowColors(True)
        self.seg_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        sl.addWidget(self.seg_table)

        # 分段操作按钮行
        seg_btn_row = QHBoxLayout()
        seg_btn_row.setSpacing(6)

        btn_add = QPushButton("➕ 添加分段")
        btn_add.clicked.connect(self._add_segment_row)
        btn_rem = QPushButton("➖ 删除选中")
        btn_rem.clicked.connect(self._remove_selected_rows)
        btn_default = QPushButton("🔄 默认值")
        btn_default.clicked.connect(self._load_defaults)

        seg_btn_row.addWidget(btn_add)
        seg_btn_row.addWidget(btn_rem)
        seg_btn_row.addWidget(btn_default)
        seg_btn_row.addStretch()
        sl.addLayout(seg_btn_row)

        left_layout.addWidget(seg_group, stretch=1)

        # 主操作按钮
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.calc_btn = QPushButton("🚀 开始计算")
        self.calc_btn.setObjectName("calcBtn")
        self.calc_btn.setFixedHeight(42)
        self.calc_btn.clicked.connect(self._start_calculation)

        self.reset_btn = QPushButton("🗑 重置")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.setFixedHeight(42)
        self.reset_btn.setFixedWidth(100)
        self.reset_btn.clicked.connect(self._reset)

        action_row.addWidget(self.calc_btn, stretch=1)
        action_row.addWidget(self.reset_btn)
        left_layout.addLayout(action_row)

        main_layout.addWidget(left_panel)

        # ========== 右侧结果面板 ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)

        # 日志文本
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        splitter.addWidget(self.log_text)

        # 图表 Tab
        self.plot_tabs = QTabWidget()
        self._init_plot_tabs()
        splitter.addWidget(self.plot_tabs)

        splitter.setSizes([280, 500])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        right_layout.addWidget(splitter)
        main_layout.addWidget(right_panel, stretch=1)

    def _init_plot_tabs(self):
        """初始化 6 个独立图表 Tab"""
        self.figures = {}
        self.canvases = {}
        self.axes = {}

        plot_configs = [
            ("amplitude", "📈 振幅演化 A(y)"),
            ("phase", "🌀 相位演化 θ(y)"),
            ("velocity", "🔵 速度分布 u-φ"),
            ("e_phase", "🟣 电子相位 φ(y,φ₀)"),
            ("e_vel", "🟠 电子速度 u(y)"),
            ("power", "⚡ 功率演化 P(z)"),
        ]

        for key, title in plot_configs:
            fig = plt.figure(figsize=(7, 4.5), dpi=100, facecolor="#ffffff")
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.set_facecolor("#ffffff")
            ax.tick_params(colors="#2c3e50")
            ax.xaxis.label.set_color("#2c3e50")
            ax.yaxis.label.set_color("#2c3e50")
            ax.title.set_color("#2980b9")
            for spine in ax.spines.values():
                spine.set_color("#bdc3c7")
            ax.grid(True, alpha=0.3, color="#bdc3c7")
            ax.text(
                0.5,
                0.5,
                "等待计算...",
                fontsize=14,
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#bdc3c7",
            )
            ax.axis("off")

            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(0, 0, 0, 0)

            nav_tb = NavigationToolbar(canvas, tab)
            nav_tb.setStyleSheet("background-color: #ecf0f1; color: #2c3e50;")
            tab_layout.addWidget(nav_tb)
            tab_layout.addWidget(canvas)

            self.plot_tabs.addTab(tab, title)
            self.figures[key] = fig
            self.canvases[key] = canvas
            self.axes[key] = ax

    def _create_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QToolBar().iconSize())
        self.addToolBar(toolbar)

        toolbar.addAction("💾 保存配置", self._save_config)
        toolbar.addAction("📂 加载配置", lambda: self._load_config(silent=False))
        toolbar.addSeparator()
        toolbar.addAction("📊 导出 Excel", self._export_excel)
        toolbar.addSeparator()
        toolbar.addAction("🖼 保存所有图表", self._save_all_plots)

    # ─────────────────── 分段表格操作 ───────────────────
    def _add_segment_row(self, data=None):
        row = self.seg_table.rowCount()
        self.seg_table.insertRow(row)
        self.seg_table.setCellWidget(row, 6, ComboBoxDelegate())

        if data:
            for col, key in enumerate(
                ["len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K"]
            ):
                item = QTableWidgetItem(str(data.get(key, "")))
                item.setTextAlignment(Qt.AlignCenter)
                self.seg_table.setItem(row, col, item)
            combo = self.seg_table.cellWidget(row, 6)
            if combo:
                combo.set_value(data.get("type", "initial"))
        else:
            for col in range(6):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignCenter)
                self.seg_table.setItem(row, col, item)

    def _remove_selected_rows(self):
        rows = set(item.row() for item in self.seg_table.selectedItems())
        for row in sorted(rows, reverse=True):
            self.seg_table.removeRow(row)

    # ─────────────────── 参数读写 ───────────────────
    def _get_common_params(self):
        params = {}
        for key, edit in self.global_vars.items():
            try:
                params[key] = float(edit.text())
            except ValueError:
                raise ValueError(f"全局参数 [{key}] 的值无效: {edit.text()}")
        return params

    def _get_segments(self):
        segments = []
        keys = ["len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K"]
        for row in range(self.seg_table.rowCount()):
            seg = {}
            has_data = False
            for col, key in enumerate(keys):
                item = self.seg_table.item(row, col)
                val = item.text().strip() if item else ""
                if val:
                    has_data = True
                try:
                    seg[key] = float(val)
                except ValueError:
                    seg[key] = 0.0
            combo = self.seg_table.cellWidget(row, 6)
            seg["type"] = combo.get_value() if combo else "initial"
            if has_data:
                segments.append(seg)
        return segments

    def _load_defaults(self):
        defaults = {
            "I": "0.3",
            "V": "23000",
            "f0": "211",
            "P_in": "0.10",
            "w": "0.2",
            "t": "0.2",
            "Loss_attu": "0",
        }
        for key, val in defaults.items():
            if key in self.global_vars:
                self.global_vars[key].setText(val)

        self.seg_table.setRowCount(0)
        self._add_segment_row(
            data={
                "len": 50,
                "Vpc": 0.2893,
                "p_SWS": 0.50,
                "Kc": 3.88,
                "Loss_perunit": 0,
                "Fn_K": 1,
                "type": "initial",
            }
        )

    def _reset(self):
        for edit in self.global_vars.values():
            edit.clear()
        self.seg_table.setRowCount(0)
        self.log_text.clear()

        # ✅ 只清空现有图表内容，不重建 Tab/Canvas
        for key in self.axes:
            ax = self.axes[key]
            ax.clear()
            ax.set_facecolor("#ffffff")
            ax.tick_params(colors="#2c3e50")
            for spine in ax.spines.values():
                spine.set_color("#bdc3c7")
            ax.grid(True, alpha=0.3, color="#dfe6e9")
            ax.text(
                0.5,
                0.5,
                "等待计算...",
                fontsize=14,
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#bdc3c7",
            )
            ax.axis("off")
            self.canvases[key].draw_idle()
            self.canvases[key].flush_events()

        self.last_calc_result = None
        self.statusBar().showMessage("已重置")

    # ─────────────────── 配置保存/加载 ───────────────────
    def _save_config(self):
        config = {
            "common": {k: v.text() for k, v in self.global_vars.items()},
            "segments": [],
        }
        keys = ["len", "Vpc", "p_SWS", "Kc", "Loss_perunit", "Fn_K"]
        for row in range(self.seg_table.rowCount()):
            seg = {}
            for col, key in enumerate(keys):
                item = self.seg_table.item(row, col)
                seg[key] = item.text() if item else ""
            combo = self.seg_table.cellWidget(row, 6)
            seg["type"] = combo.get_value() if combo else "initial"
            config["segments"].append(seg)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存参数配置", "", "JSON 文件 (*.json)"
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            # 自动保存一份
            with open("twt_config_autoload.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.statusBar().showMessage(f"配置已保存: {file_path}")

    def _load_config(self, silent=False):
        file_path = "twt_config_autoload.json" if silent else None
        if not silent:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "加载参数配置", "", "JSON 文件 (*.json)"
            )
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for key, val in config.get("common", {}).items():
                if key in self.global_vars:
                    self.global_vars[key].setText(str(val))
            self.seg_table.setRowCount(0)
            for seg in config.get("segments", []):
                self._add_segment_row(data=seg)
            if not silent:
                self.statusBar().showMessage(f"配置已加载: {file_path}")
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "加载错误", str(e))

    # ─────────────────── 计算 ───────────────────
    def _append_log(self, msg):
        self.log_text.moveCursor(self.log_text.textCursor().End)
        self.log_text.insertPlainText(msg + "\n")
        self.log_text.ensureCursorVisible()
        QApplication.processEvents()

    def _start_calculation(self):
        try:
            common_params = self._get_common_params()
            segments = self._get_segments()
            loss_attu = common_params.pop("Loss_attu", 0)
        except Exception as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return

        if not segments:
            QMessageBox.warning(self, "参数错误", "请至少输入一个分段参数")
            return

        self.log_text.clear()
        self._append_log("🚀 开始计算...")
        self.calc_btn.setEnabled(False)
        self.statusBar().showMessage("⏳ 正在计算中...")

        # 在子线程中运行计算
        self.calc_thread = QThread()
        self.calc_worker = CalcWorker(common_params, segments, loss_attu)
        self.calc_worker.moveToThread(self.calc_thread)

        self.calc_thread.started.connect(self.calc_worker.run)
        self.calc_worker.log_signal.connect(self._append_log)
        self.calc_worker.finished.connect(self._on_calc_finished)
        self.calc_worker.error.connect(self._on_calc_error)
        self.calc_worker.finished.connect(self.calc_thread.quit)
        self.calc_worker.error.connect(self.calc_thread.quit)

        self.calc_thread.start()

    def _on_calc_finished(self, result):
        self.last_calc_result = result
        self.calc_btn.setEnabled(True)
        common = self._get_common_params()

        self._append_log("\n" + "=" * 50)
        self._append_log("✅ 计算完成!")
        self._append_log(
            f"工作频率: {common['f0']} GHz  |  输入功率: {common['P_in']} W"
        )
        self._append_log(f"非线性增益: {result['Gain_dB']:.4f} dB")
        self._append_log(f"输出功率: {result['P_out_end']:.4f} W")
        self._append_log(f"最大效率: {result['Eff_max']:.4f}%")
        self._append_log(f"最大功率: {result['P_max']:.4f} W")
        self.statusBar().showMessage(
            f"✅ 计算完成 — 增益 {result['Gain_dB']:.2f} dB | 效率 {result['Eff_max']:.2f}%"
        )

        self._update_plots(result, common["P_in"])

    def _on_calc_error(self, err_msg):
        self.calc_btn.setEnabled(True)
        self._append_log(f"\n❌ 计算错误:\n{err_msg}")
        self.statusBar().showMessage("❌ 计算失败")
        QMessageBox.critical(self, "计算错误", err_msg)

    # ─────────────────── 图表更新 ───────────────────
    def _style_ax(self, ax, xlabel, ylabel, title):
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#2c3e50", labelsize=10)
        ax.xaxis.label.set_color("#2c3e50")
        ax.yaxis.label.set_color("#2c3e50")
        ax.title.set_color("#2980b9")
        for spine in ax.spines.values():
            spine.set_color("#bdc3c7")
        ax.grid(True, alpha=0.3, color="#dfe6e9")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # ✅ 确保坐标轴可见（覆盖 axis('off') 的残留状态）
        ax.set_axis_on()

    def _update_plots(self, res, P_in):
        Y = res["Y"]
        Lenth = res["Lenth"]
        final_seg = res["final_seg"]

        # 1. 振幅
        ax = self.axes["amplitude"]
        ax.clear()
        ax.plot(Y, res["A"], color="#2980b9", linewidth=1.5)
        self._style_ax(ax, "Position y", "Amplitude A(y)", "Amplitude Growth")
        self.figures["amplitude"].tight_layout()
        self.canvases["amplitude"].draw_idle()
        self.canvases["amplitude"].flush_events()

        # 2. 相位
        ax = self.axes["phase"]
        ax.clear()
        ax.plot(Y, res["theta"], color="#c0392b", linewidth=1.5)
        self._style_ax(ax, "Position y", "Phase Shift θ(y)", "Phase Evolution")
        self.figures["phase"].tight_layout()
        self.canvases["phase"].draw_idle()
        self.canvases["phase"].flush_events()

        # 3. 速度分布散点
        ax = self.axes["velocity"]
        ax.clear()
        sc = ax.scatter(
            final_seg["phi_final"],
            final_seg["u_final"],
            c=final_seg["phi_final"],
            cmap="hsv",
            s=18,
            edgecolor="#2c3e50",
            linewidth=0.3,
        )
        self._style_ax(ax, "Final Phase φ", "Final Velocity u", "Velocity Distribution")
        self.figures["velocity"].tight_layout()
        self.canvases["velocity"].draw_idle()
        self.canvases["velocity"].flush_events()

        # 4. 电子相位
        ax = self.axes["e_phase"]
        ax.clear()
        ax.plot(Y, res["phi"], color="#8e44ad", linewidth=1.2)
        self._style_ax(
            ax, "Position y", "Electron Phase φ(y,φ0)", "Electron Phase Space"
        )
        self.figures["e_phase"].tight_layout()
        self.canvases["e_phase"].draw_idle()
        self.canvases["e_phase"].flush_events()

        # 5. 电子速度
        ax = self.axes["e_vel"]
        ax.clear()
        ax.plot(Lenth, res["u"], color="#d35400", linewidth=1.2)
        self._style_ax(
            ax, "Position Z", "Electron Velocity u(y)", "Electron Velocity Space"
        )
        self.figures["e_vel"].tight_layout()
        self.canvases["e_vel"].draw_idle()
        self.canvases["e_vel"].flush_events()

        # 6. 功率
        ax = self.axes["power"]
        ax.clear()
        ax.plot(Lenth, res["P_Out"], color="#27ae60", linewidth=1.5)
        ax.axhline(
            y=P_in, color="#95a5a6", linestyle="--", alpha=0.8, label="Input Power"
        )
        ax.legend(facecolor="#ffffff", edgecolor="#bdc3c7", labelcolor="#2c3e50")
        self._style_ax(ax, "Position Z", "Output Power (W)", "Power Evolution")
        self.figures["power"].tight_layout()
        self.canvases["power"].draw_idle()
        self.canvases["power"].flush_events()

    # ─────────────────── 导出 Excel ───────────────────
    def _export_excel(self):
        if not self.last_calc_result:
            QMessageBox.warning(self, "提示", "请先进行计算！")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel", "", "Excel 文件 (*.xlsx)"
        )
        if not file_path:
            return

        res = self.last_calc_result
        try:
            y_data = res.get("Y", np.array([]))

            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                # Sheet: A(y)
                if "A" in res:
                    df = pd.DataFrame({"y": y_data, "A(y)": res["A"]})
                    df.to_excel(writer, sheet_name="A(y)", index=False)

                # Sheet: theta(y)
                if "theta" in res:
                    df = pd.DataFrame({"y": y_data, "theta(y)": res["theta"]})
                    df.to_excel(writer, sheet_name="theta(y)", index=False)

                # Sheet: u(y, phi0) — 矩阵原样导出
                if "u" in res:
                    mat = res["u"]
                    if mat.ndim == 2:
                        cols = [f"phi0_{i+1}" for i in range(mat.shape[1])]
                        df = pd.DataFrame(mat, columns=cols)
                        df.insert(0, "y", y_data)
                    else:
                        df = pd.DataFrame({"y": y_data, "u(y)": mat})
                    df.to_excel(writer, sheet_name="u(y,phi0)", index=False)

                # Sheet: phi(y, phi0) — 矩阵原样导出
                if "phi" in res:
                    mat = res["phi"]
                    if mat.ndim == 2:
                        cols = [f"phi0_{i+1}" for i in range(mat.shape[1])]
                        df = pd.DataFrame(mat, columns=cols)
                        df.insert(0, "y", y_data)
                    else:
                        df = pd.DataFrame({"y": y_data, "phi(y)": mat})
                    df.to_excel(writer, sheet_name="phi(y,phi0)", index=False)

                # Sheet: P_Out(y)
                if "P_Out" in res:
                    df = pd.DataFrame({"y": y_data, "P_Out(W)": res["P_Out"]})
                    df.to_excel(writer, sheet_name="P_Out(y)", index=False)

                # Sheet: Summary
                common = self._get_common_params()
                df = pd.DataFrame(
                    {
                        "指标": [
                            "工作频率(GHz)",
                            "输入功率(W)",
                            "非线性增益(dB)",
                            "输出功率(W)",
                            "最大效率(%)",
                            "最大功率(W)",
                        ],
                        "值": [
                            common["f0"],
                            common["P_in"],
                            res["Gain_dB"],
                            res["P_out_end"],
                            res["Eff_max"],
                            res["P_max"],
                        ],
                    }
                )
                df.to_excel(writer, sheet_name="Summary", index=False)

            self.statusBar().showMessage(f"📊 Excel 已导出: {file_path}")
            QMessageBox.information(self, "导出成功", f"结果已导出至:\n{file_path}")

        except Exception as e:
            import traceback

            QMessageBox.critical(
                self, "导出错误", f"{str(e)}\n\n{traceback.format_exc()}"
            )

    # ─────────────────── 保存所有图表 ───────────────────
    def _save_all_plots(self):
        if not self.last_calc_result:
            QMessageBox.warning(self, "提示", "请先进行计算！")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not folder:
            return
        timestamp = datetime.now().strftime("%Y%m")
        save_dir = os.path.join(folder, f"TWT_Plots_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        names = {
            "amplitude": "1_Amplitude_Growth",
            "phase": "2_Phase_Evolution",
            "velocity": "3_Velocity_Distribution",
            "e_phase": "4_Electron_Phase",
            "e_vel": "5_Electron_Velocity",
            "power": "6_Power_Evolution",
        }
        for key, name in names.items():
            path = os.path.join(save_dir, f"{name}.png")
            self.figures[key].savefig(
                path, dpi=600, bbox_inches="tight", facecolor="#ffffff"
            )

        self.statusBar().showMessage(f"🖼 图表已保存至: {save_dir}")
        QMessageBox.information(self, "保存成功", f"所有图表已保存至:\n{save_dir}")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = TWTMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
