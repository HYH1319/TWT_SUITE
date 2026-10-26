# -*- coding: utf-8 -*-
"""
行波管多频点 RVEA 多目标优化 —— PyQt5 图形界面
依赖: pip install PyQt5 numpy, 同目录下 rvae_core.py

页签结构:
  ① 物理参数      - 电子枪 / 束流 / 慢波结构 / 输入衰减 / 调整系数
  ② 多频点参数    - 每个频点 Kc/Vpc/freq + 求解精度
  ③ 初始值与边界  - 初始 p_SWS + 参数上下限 (p_SWS 的搜索空间唯一入口)
  ④ RVEA 核心     - 分区数 p / 最大迭代 / 并行 (必调)
  ⑤ RVEA 进阶     - SBX / PM / α (默认即可)
"""

import sys, os, json, logging, time
import multiprocessing as mp
from datetime import datetime
from typing import List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar,
    QFileDialog, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush

from PYQT_NOLine__PVTOPT_MUTILFREQ_MORVEA_AL_VCBEAMS_CORE import (
    FreqParam, FixedParams, Coeffs, RVEAConfig, OptimizationResult,
    RVEAOptimizer,
)

logger = logging.getLogger(__name__)

N_FREQ       = 3
MAX_WORKERS  = max(1, mp.cpu_count())
CONFIG_FILE  = "./configUDF/OPT/twt_RVEA_config_MUTIL.json"

_LBL_PREC  = "求解精度等级:"
_FILE_FLT  = "文本文件 (*.txt);;所有文件 (*)"


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Qt 线程封装                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class RVEATask(QThread):
    finished  = pyqtSignal(object)
    prog_sig  = pyqtSignal(int, object)
    iter_sig  = pyqtSignal(int, object)
    init_sig  = pyqtSignal(object)
    log_sig   = pyqtSignal(str)

    def __init__(self, opt: RVEAOptimizer, ip=None):
        super().__init__()
        self.opt = opt
        self.ip  = ip

    def run(self):
        self.opt._on_progress  = self.prog_sig.emit
        self.opt._on_iteration = self.iter_sig.emit
        self.opt._on_initial   = self.init_sig.emit
        self.opt._on_log       = self.log_sig.emit
        try:
            result = self.opt.run(initial_p=self.ip)
            self.finished.emit(result)
        except Exception as e:
            self.log_sig.emit("异常: %s" % e)
            self.finished.emit(None)

    def stop(self):
        self.opt.stop()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  通用编辑器基类                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

class BaseEditor(QWidget):
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.widgets = {}
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        for field in fields:
            if len(field) == 2 and field[1] == "label":
                lbl = QLabel(field[0])
                lbl.setStyleSheet(
                    "font-weight:bold;color:#1565C0;"
                    "padding-top:8px;font-size:10pt;")
                layout.addRow(lbl)
                continue
            key, wtype, *args = field
            label  = QLabel(key)
            widget = None
            if wtype == "double":
                widget = QDoubleSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
                if len(args) > 3: widget.setSingleStep(args[3])
                if len(args) > 4: widget.setDecimals(args[4])
            elif wtype == "int":
                widget = QSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
            elif wtype == "text":
                widget = QLineEdit()
                if args: widget.setText(args[0])
            elif wtype == "combo":
                widget = QComboBox()
                options = args[0] if len(args) > 0 else []
                widget.addItems(options)
                default = args[1] if len(args) > 1 else (options[0] if options else "")
                if default in options:
                    widget.setCurrentText(default)
            if widget is not None:
                layout.addRow(label, widget)
                self.widgets[key] = widget
        self.setLayout(layout)

    def get(self, key):
        w = self.widgets.get(key)
        if w is None: return None
        if isinstance(w, (QDoubleSpinBox, QSpinBox)): return w.value()
        if isinstance(w, QComboBox):  return w.currentText()
        if isinstance(w, QLineEdit):  return w.text()
        return None

    def put(self, key, val):
        w = self.widgets.get(key)
        if w is None: return
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            try: w.setValue(float(val))
            except Exception: pass
        elif isinstance(w, QComboBox):
            idx = w.findText(str(val))
            if idx >= 0: w.setCurrentIndex(idx)
        elif isinstance(w, QLineEdit):
            w.setText(str(val))

    def collect(self):
        return {k: self.get(k) for k in self.widgets}

    def apply(self, data):
        if not isinstance(data, dict): return
        for k, v in data.items():
            self.put(k, v)

    @staticmethod
    def _parse_list(s, dtype=float):
        return [dtype(v.strip()) for v in s.split(",") if v.strip()]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ① 物理参数 (p_SWS 不在这里, 它是优化变量)                       ║
# ╚══════════════════════════════════════════════════════════════════╝

class PhysParamEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("电子枪", "label"),
            ("电流 I (A):",              "double", 0, 1e7, 0.3,    0.001, 4),
            ("电压 V (V):",              "double", 0, 1e7, 23000,  1,     0),
            ("束流", "label"),
            ("束流宽度 w (mm):",         "double", 0, 1e7, 0.2,    0.001, 4),
            ("束流厚度 t (mm):",         "double", 0, 1e7, 0.2,    0.001, 4),
            ("填充率倒数 Fn_K:",         "double", 1, 1e7, 1.0,    0.001, 4),
            ("慢波结构 (N_Unit 的长度即优化维度)", "label"),
            ("各段周期数 N_Unit (逗号分隔):",
             "text", "20,5,10,10,10,10"),
            ("每单元损耗:",              "double", 0, 1e7, 0.0,    0.001, 4),
            ("输入/衰减", "label"),
            ("输入功率 P_in (W):",       "double", 0, 1e7, 0.1,    0.001, 4),
            ("衰减量 (dB):",             "double", 0, 1e7, 20,     1,     1),
            ("衰减段索引 (逗号分隔):",    "text",   "1"),
            ("参数调整系数", "label"),
            ("Vpc 调整系数:",            "double", 0, 1e7, 0.82,   0.001, 4),
            ("Kc 调整系数:",             "double", 0, 1e7, 1.6,    0.001, 4),
        ]
        super().__init__(fields, parent)

    def get_dict(self):
        try:
            return dict(
                i=float(self.get("电流 I (A):")),
                v=float(self.get("电压 V (V):")),
                loss_perunit=float(self.get("每单元损耗:")),
                section_seg_idx=self._parse_list(
                    self.get("衰减段索引 (逗号分隔):"), int),
                n_unit=self._parse_list(
                    self.get("各段周期数 N_Unit (逗号分隔):"), int),
                w=float(self.get("束流宽度 w (mm):")),
                t=float(self.get("束流厚度 t (mm):")),
                Fn_K=float(self.get("填充率倒数 Fn_K:")),
                p_in=float(self.get("输入功率 P_in (W):")),
                loss_attu=float(self.get("衰减量 (dB):")),
                vpc_coeff=float(self.get("Vpc 调整系数:")),
                kc_coeff=float(self.get("Kc 调整系数:")),
            )
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ② 多频点参数                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class FreqParamEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("频点 1", "label"),
            ("频点1 Kc (Ω):",     "double", 0, 1e7, 2.67,   0.001, 4),
            ("频点1 Vpc (c):",    "double", 0, 1,   0.2867, 0.001, 4),
            ("频点1 频率 (GHz):", "double", 0, 1e7, 220.0,  1,     1),
            ("频点 2", "label"),
            ("频点2 Kc (Ω):",     "double", 0, 1e7, 2.67,   0.001, 4),
            ("频点2 Vpc (c):",    "double", 0, 1,   0.2867, 0.001, 4),
            ("频点2 频率 (GHz):", "double", 0, 1e7, 225.0,  1,     1),
            ("频点 3", "label"),
            ("频点3 Kc (Ω):",     "double", 0, 1e7, 2.67,   0.001, 4),
            ("频点3 Vpc (c):",    "double", 0, 1,   0.2867, 0.001, 4),
            ("频点3 频率 (GHz):", "double", 0, 1e7, 230.0,  1,     1),
            ("求解", "label"),
            (_LBL_PREC, "combo",
             ["标准精度", "高精度", "极高精度"], "高精度"),
        ]
        super().__init__(fields, parent)

    def get_freqs(self):
        try:
            return [
                FreqParam(
                    Kc=float(self.get("频点%d Kc (Ω):" % i)),
                    Vpc=float(self.get("频点%d Vpc (c):" % i)),
                    freq=float(self.get("频点%d 频率 (GHz):" % i)),
                )
                for i in (1, 2, 3)
            ]
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None

    def get_precision(self) -> str:
        return self.get(_LBL_PREC) or "高精度"


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ③ 初始值与边界 (p_SWS 的唯一入口)                              ║
# ╚══════════════════════════════════════════════════════════════════╝

class InitBoundsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("初始种群种子", "label"),
            ("初始 p_SWS 向量 (逗号分隔):",
             "text", "0.50,0.50,0.50,0.50,0.50,0.50"),
            ("搜索边界", "label"),
            ("参数下限 (逗号分隔):",
             "text", "0.499,0.490,0.49,0.49,0.49,0.49"),
            ("参数上限 (逗号分隔):",
             "text", "0.501,0.510,0.51,0.51,0.51,0.51"),
        ]
        super().__init__(fields, parent)

    def get_init_p(self, n_dim):
        try:
            vals = self._parse_list(self.get("初始 p_SWS 向量 (逗号分隔):"), float)
            if len(vals) == 1 and n_dim > 1:
                vals = vals * n_dim
            if len(vals) != n_dim:
                raise ValueError("维度不匹配 (需要 %d 维)" % n_dim)
            return vals
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None

    def get_bounds(self, n_dim):
        try:
            lb = self._parse_list(self.get("参数下限 (逗号分隔):"), float)
            ub = self._parse_list(self.get("参数上限 (逗号分隔):"), float)
            if len(lb) == 1 and n_dim > 1: lb = lb * n_dim
            if len(ub) == 1 and n_dim > 1: ub = ub * n_dim
            if len(lb) != n_dim or len(ub) != n_dim:
                raise ValueError("边界维度不匹配 (需要 %d 维)" % n_dim)
            if any(l > u for l, u in zip(lb, ub)):
                raise ValueError("存在下限 > 上限的维度")
            return list(zip(lb, ub))
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ④ RVEA 核心参数 (必调)                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class RVEABasicEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("参考向量与种群", "label"),
            ("分区数 p [关键] (决定参考向量数与种群大小):",
             "int", 2, 100, 12),
            ("收敛控制", "label"),
            ("最大迭代 [关键] (10~20维建议 200~500):",
             "int", 10, 5000, 200),
            ("并行计算", "label"),
            ("并行进程数 [关键] (建议 = CPU核数-1):",
             "int", 1, 128, min(32, MAX_WORKERS)),
        ]
        super().__init__(fields, parent)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ⑤ RVEA 进阶参数 (默认即可)                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

class RVEAAdvancedEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("SBX 交叉 [默认即可]", "label"),
            ("交叉概率:",  "double", 0.0, 1.0, 0.9, 0.01, 2),
            ("交叉 eta:",  "double", 1,  100,  15,  1, 0),
            ("PM 变异 [默认即可]", "label"),
            ("变异 eta:",  "double", 1,  100,  20,  1, 0),
            ("APD 惩罚 [默认即可]", "label"),
            ("惩罚指数 α:", "double", 0.1, 10.0, 2.0, 0.1, 1),
        ]
        super().__init__(fields, parent)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Pareto 前沿面板                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

class ParetoPanel(QWidget):
    solution_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pop  = None
        self._fval = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.info_lbl = QLabel("等待优化结果...")
        layout.addWidget(self.info_lbl)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setStyleSheet(
            "QTableWidget{font-family:Consolas,monospace;font-size:10pt;}"
            "QHeaderView::section{background:#e8e8e8;font-weight:bold;padding:4px;}")
        self.table.selectionModel().selectionChanged.connect(self._on_sel)
        layout.addWidget(self.table, 1)

    def update_result(self, result: OptimizationResult, freq_labels: List[str]):
        self._pop  = result.pareto_pop
        self._fval = result.pareto_F
        ns = result.n_solutions
        no = result.pareto_F.shape[1]
        nd = result.pareto_pop.shape[1]
        bi = result.best_idx

        hdr = list(freq_labels) + ["总功率"] + ["p[%d]" % i for i in range(nd)]
        self.table.setColumnCount(len(hdr))
        self.table.setHorizontalHeaderLabels(hdr)
        self.table.setRowCount(ns)

        cb_color = QColor("#c8e6c9")
        cn_color = QColor("#ffffff")

        for row in range(ns):
            is_best = (row == bi)
            bg = cb_color if is_best else cn_color
            for col in range(no):
                item = QTableWidgetItem("%.3f" % result.pareto_F[row, col])
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QBrush(bg))
                self.table.setItem(row, col, item)

            total_item = QTableWidgetItem("%.3f" % result.pareto_F[row].sum())
            total_item.setTextAlignment(Qt.AlignCenter)
            total_item.setBackground(QBrush(bg))
            self.table.setItem(row, no, total_item)

            for k in range(nd):
                pi = QTableWidgetItem("%.6f" % result.pareto_pop[row, k])
                pi.setTextAlignment(Qt.AlignCenter)
                pi.setBackground(QBrush(bg))
                self.table.setItem(row, no + 1 + k, pi)

        if bi >= 0:
            self.table.scrollToItem(self.table.item(bi, 0))
            self.table.selectRow(bi)

        self.info_lbl.setText(
            "共%d方案 | 推荐行%d 总%.3fW" % (ns, bi, result.best_total_power))

    def _on_sel(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.solution_selected.emit(rows[0].row())

    def get_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows and self._pop is not None:
            idx = rows[0].row()
            return self._pop[idx], self._fval[idx]
        return None, None


# ╔══════════════════════════════════════════════════════════════════╗
# ║  主窗口                                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行波管 RVEA 多目标优化工具")
        self.setGeometry(100, 100, 1360, 880)
        self.task          = None
        self.result        = None
        self.t0            = None
        self._freq_params  = None
        self._init_ui()
        self._load_cfg()
        self.statusBar().showMessage("就绪 | CPU:%d核" % mp.cpu_count())

    @staticmethod
    def _wrap_tab(editor, hint_text):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        hint = QLabel(hint_text)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color:#444;background:#fffde7;padding:6px;"
            "border:1px solid #ffe082;border-radius:4px;font-size:9pt;")
        lay.addWidget(hint)
        lay.addWidget(editor)
        lay.addStretch()
        return w

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(2, 2, 2, 2)

        tabs = QTabWidget()
        tabs.setFont(QFont("Arial", 10))

        # ① 物理参数
        self.edit_phys = PhysParamEditor()
        tabs.addTab(
            self._wrap_tab(
                self.edit_phys,
                "① 物理参数: 电子枪 / 束流 / 慢波结构 / 输入衰减 / 调整系数。\n"
                "注意: p_SWS 是优化变量, 不在此处配置 —— 请到 ③ 初始值与边界 设定。\n"
                "N_Unit 的长度即优化维度, 决定 ③ 中 init_p / bounds 应有的长度。"
            ),
            "① 物理参数")

        # ② 多频点参数
        self.edit_freq = FreqParamEditor()
        tabs.addTab(
            self._wrap_tab(
                self.edit_freq,
                "② 多频点参数: 每个频点独立配置 Kc / Vpc / 频率。\n"
                "求解精度影响每次物理评估的空间电荷场截断项数: 标准=20 / 高=50 / 极高=100。"
            ),
            "② 多频点参数")

        # ③ 初始值与边界
        self.edit_init = InitBoundsEditor()
        tabs.addTab(
            self._wrap_tab(
                self.edit_init,
                "③ 初始值与边界 (p_SWS 的唯一入口): 初始向量为种群第 0 个个体播种;\n"
                "上下限决定搜索空间。向量长度必须等于 ① 中的 N_Unit 长度。\n"
                "10~20 维时边界建议按实际加工能力设为 ±2% ~ ±10%, "
                "过窄会失去优化空间, 过宽会拖慢收敛。"
            ),
            "③ 初始值与边界")

        # ④ RVEA 核心 (必调)
        self.edit_rc_basic = RVEABasicEditor()
        self.ref_lbl = QLabel("")
        self.ref_lbl.setStyleSheet(
            "color:#1565C0;font-weight:bold;padding:4px;"
            "background:#e3f2fd;border-radius:4px;")
        tab_basic = QWidget()
        tb_lay = QVBoxLayout(tab_basic)
        tb_lay.setContentsMargins(4, 4, 4, 4)
        tb_hint = QLabel(
            "④ RVEA 核心参数: 显著影响收敛质量与效率, 建议根据维度/目标数显式调整。\n"
            "分区数 p 决定参考向量数 = 种群大小 (见下方实时值); "
            "最大迭代控制收敛深度; 并行进程数决定吞吐。")
        tb_hint.setWordWrap(True)
        tb_hint.setStyleSheet(
            "color:#444;background:#fffde7;padding:6px;"
            "border:1px solid #ffe082;border-radius:4px;font-size:9pt;")
        tb_lay.addWidget(tb_hint)
        tb_lay.addWidget(self.edit_rc_basic)
        tb_lay.addWidget(self.ref_lbl)
        tb_lay.addStretch()
        tabs.addTab(tab_basic, "④ RVEA 核心 [必调]")

        self.edit_rc_basic.widgets[
            "分区数 p [关键] (决定参考向量数与种群大小):"
        ].valueChanged.connect(self._update_ref_info)

        # ⑤ RVEA 进阶 (默认)
        self.edit_rc_adv = RVEAAdvancedEditor()
        tabs.addTab(
            self._wrap_tab(
                self.edit_rc_adv,
                "⑤ RVEA 进阶参数: 标准设置下默认值已能很好契合 10~20 维、2~4 目标优化。\n"
                "仅当需要对种群多样性/收敛速度做特殊权衡时才修改。"
            ),
            "⑤ RVEA 进阶 [默认]")

        left_layout.addWidget(tabs)

        btn_layout = QHBoxLayout()
        self.btn_run  = self._make_btn("开始优化", "#4CAF50", self._start)
        self.btn_stop = self._make_btn("停止",     "#f44336", self._stop)
        self.btn_stop.setEnabled(False)
        self.btn_save = self._make_btn("保存结果",   "#2196F3", self._save)
        self.btn_save.setEnabled(False)
        self.btn_use  = self._make_btn("使用选中方案", "#2196F3", self._use)
        self.btn_use.setEnabled(False)
        for b in [self.btn_run, self.btn_stop, self.btn_save, self.btn_use]:
            btn_layout.addWidget(b)
        left_layout.addLayout(btn_layout)

        splitter.addWidget(left_widget)

        right_splitter = QSplitter(Qt.Vertical)

        right_top = QWidget()
        rt_layout = QVBoxLayout(right_top)
        rt_layout.setContentsMargins(2, 2, 2, 2)

        status_grp = QGroupBox("优化状态")
        grid = QGridLayout(status_grp)
        self._status = {}
        status_defs = [
            ("ps", "Pareto大小",   "#e0e0e0"),
            ("mt", "最大总功率(W)", "#c8e6c9"),
            ("rv", "参考向量数",    "#e3f2fd"),
        ]
        for row, (key, text, color) in enumerate(status_defs):
            grid.addWidget(QLabel(text + ":"), row, 0)
            val_lbl = QLabel("--")
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet(
                "border:1px solid #b0b0b0;border-radius:4px;"
                "padding:4px;font-weight:bold;background:%s;"
                "font-size:10pt;" % color)
            grid.addWidget(val_lbl, row, 1)
            self._status[key] = val_lbl

        info_row = QHBoxLayout()
        self.lbl_time = QLabel("用时:--")
        self.lbl_iter = QLabel("迭代:--")
        for w in [self.lbl_time, self.lbl_iter]:
            w.setStyleSheet("font-weight:bold;")
            info_row.addWidget(w)
        grid.addLayout(info_row, len(status_defs), 0, 1, 2)
        rt_layout.addWidget(status_grp)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        rt_layout.addWidget(self.progress)

        log_grp = QGroupBox("优化日志")
        log_layout = QVBoxLayout(log_grp)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "font-family:Consolas,monospace;font-size:10pt;"
            "background:#f8f8f8;border-radius:4px;")
        log_layout.addWidget(self.log_box)
        rt_layout.addWidget(log_grp, 1)

        right_splitter.addWidget(right_top)

        self.pareto_panel = ParetoPanel()
        self.pareto_panel.solution_selected.connect(
            lambda _: self.btn_use.setEnabled(True))
        pareto_grp = QGroupBox("Pareto前沿")
        pg_layout = QVBoxLayout(pareto_grp)
        pg_layout.addWidget(self.pareto_panel)
        right_splitter.addWidget(pareto_grp)

        right_splitter.setSizes([350, 500])
        splitter.addWidget(right_splitter)
        splitter.setSizes([420, 880])

        self._update_ref_info()

    @staticmethod
    def _make_btn(text, color, callback):
        btn = QPushButton(text)
        btn.setStyleSheet(
            "QPushButton{background:%s;color:#fff;font-weight:bold;"
            "padding:8px;font-size:11pt;border-radius:4px;}"
            "QPushButton:hover{background:%s;}"
            "QPushButton:disabled{background:#ccc;color:#888;}" % (color, color))
        btn.clicked.connect(callback)
        return btn

    def _update_ref_info(self):
        try:
            p  = self.edit_rc_basic.get(
                "分区数 p [关键] (决定参考向量数与种群大小):")
            nr = RVEAConfig.ref_dir_count(p, N_FREQ)
            self.ref_lbl.setText(
                "当前 Das-Dennis(p=%d) → %d 个参考向量 (= 种群大小)" % (p, nr))
            if "rv" in self._status:
                self._status["rv"].setText(str(nr))
        except Exception:
            pass

    def _freq_labels(self) -> List[str]:
        if not self._freq_params:
            return ["频点%d" % (i + 1) for i in range(N_FREQ)]
        return ["频点%d(%.1fGHz)" % (i + 1, fp.freq)
                for i, fp in enumerate(self._freq_params)]

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append("[%s] %s" % (ts, msg))
        self.log_box.ensureCursorVisible()

    def _collect_all(self):
        phys_dict = self.edit_phys.get_dict()
        if phys_dict is None:
            return None
        freqs = self.edit_freq.get_freqs()
        if freqs is None:
            return None
        precision = self.edit_freq.get_precision()
        try:
            fixed = FixedParams(precision=precision, **phys_dict)
        except Exception as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return None

        bounds = self.edit_init.get_bounds(fixed.n_dim)
        if bounds is None:
            return None
        init_p = self.edit_init.get_init_p(fixed.n_dim)
        if init_p is None:
            return None

        cfg = RVEAConfig(
            n_partitions   = self.edit_rc_basic.get(
                "分区数 p [关键] (决定参考向量数与种群大小):"),
            pop_size       = 0,
            max_iter       = self.edit_rc_basic.get(
                "最大迭代 [关键] (10~20维建议 200~500):"),
            workers        = min(
                self.edit_rc_basic.get("并行进程数 [关键] (建议 = CPU核数-1):"),
                MAX_WORKERS),
            prob_crossover = self.edit_rc_adv.get("交叉概率:"),
            eta_crossover  = self.edit_rc_adv.get("交叉 eta:"),
            eta_mutation   = self.edit_rc_adv.get("变异 eta:"),
            alpha          = self.edit_rc_adv.get("惩罚指数 α:"),
        )
        return fixed, freqs, cfg, bounds, init_p

    def _start(self):
        collected = self._collect_all()
        if collected is None:
            return
        fixed, freqs, cfg, bounds, init_p = collected

        coeffs = Coeffs(vpc=fixed.vpc_coeff, kc=fixed.kc_coeff)
        self._save_cfg()
        self.log_box.clear()
        self._log("===== 开始 RVEA 优化 =====")
        self._log("求解精度: %s" % fixed.precision)

        self._freq_params = freqs

        tmp = RVEAOptimizer(fixed, freqs, coeffs, bounds, cfg)
        self._log("计算初始功率...")
        QApplication.processEvents()
        try:
            pw = tmp.evaluate_initial(init_p)
            self._log("--- 初始功率 ---")
            for j, fp in enumerate(freqs):
                self._log("  频点%d(%.1fGHz): %.3fW" % (j + 1, fp.freq, pw[j]))
            self._log("  总功率: %.3fW" % sum(pw))
        except Exception as e:
            self._log("初始评估失败: %s" % e)
            return

        nr = cfg.n_ref_dirs_for(len(freqs))
        self._status["rv"].setText(str(nr))

        optimizer = RVEAOptimizer(fixed, freqs, coeffs, bounds, cfg)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_use.setEnabled(False)

        self._log(
            "维度:%d | 频点:%d | 种群:%d(=n_ref) | 迭代:%d | 进程:%d | 精度:%s" % (
                fixed.n_dim, len(freqs), nr, cfg.max_iter, cfg.workers,
                fixed.precision))

        self.progress.setValue(0)
        self.t0 = time.time()

        self.task = RVEATask(optimizer, init_p)
        self.task.prog_sig.connect(self._on_prog)
        self.task.iter_sig.connect(self._on_iter)
        self.task.init_sig.connect(
            lambda f: self._status["ps"].setText(str(len(f))))
        self.task.log_sig.connect(self._log)
        self.task.finished.connect(self._on_done)
        self.task.start()

    def _stop(self):
        if self.task is not None and self.task.isRunning():
            self.task.stop()
            self._log("停止中...")
            self.btn_stop.setEnabled(False)
            self.btn_run.setEnabled(True)

    def _on_prog(self, pct, pf_F):
        self.progress.setValue(pct)
        if pf_F is not None and len(pf_F) > 0:
            self._status["mt"].setText("%.2f" % pf_F.sum(axis=1).max())
        if self.t0 is not None:
            elapsed = time.time() - self.t0
            m, s = divmod(elapsed, 60)
            self.lbl_time.setText("用时:%02d:%02d" % (int(m), int(s)))

    def _on_iter(self, gen, pf_F):
        self.lbl_iter.setText("迭代:%d" % gen)
        if pf_F is not None and len(pf_F) > 0:
            self._status["ps"].setText(str(len(pf_F)))

    def _on_done(self, result):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(100)

        if result is None:
            self._log("===== 优化未完成 =====")
            return
        self.result = result
        if result.n_solutions == 0:
            self._log("===== 无可行解 =====")
            return

        self._log("===== 优化完成 =====")
        self._log("Pareto前沿: %d方案" % result.n_solutions)
        self._log("推荐方案%d: 总功率%.3fW" % (
            result.best_idx, result.best_total_power))

        self.pareto_panel.update_result(result, self._freq_labels())
        self._status["ps"].setText(str(result.n_solutions))
        self._status["mt"].setText("%.2f" % result.best_total_power)
        self.btn_save.setEnabled(True)

    def _use(self):
        p, F = self.pareto_panel.get_selected()
        if p is None:
            QMessageBox.information(self, "提示", "请先选择方案")
            return
        p_str = ",".join("%.6f" % v for v in p)
        self.edit_init.put("初始 p_SWS 向量 (逗号分隔):", p_str)
        self._log("已回填 p_SWS: %s" % p_str)

    def _save(self):
        if self.result is None:
            return
        default_name = "TWT_RVEA_%s.txt" % datetime.now().strftime(
            "%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存", default_name, _FILE_FLT)
        if not path:
            return
        r = self.result
        labels = self._freq_labels()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("RVEA优化结果 | 时间:%s | 耗时:%.1fs | Pareto:%d\n" % (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    r.elapsed_sec, r.n_solutions))
                nd  = r.pareto_pop.shape[1]
                hdr = ["方案"] + labels + ["总功率"]
                hdr += ["p[%d]" % i for i in range(nd)]
                f.write("\t".join(hdr) + "\n")
                for k in range(r.n_solutions):
                    row = [str(k)]
                    for j in range(r.pareto_F.shape[1]):
                        row.append("%.4f" % r.pareto_F[k, j])
                    row.append("%.4f" % r.pareto_F[k].sum())
                    for v in r.pareto_pop[k]:
                        row.append("%.6f" % v)
                    f.write("\t".join(row) + "\n")
                f.write("\n推荐方案%d: 总功率%.4fW\n" % (
                    r.best_idx, r.best_total_power))
                p_list = ["%.6f" % v for v in r.best_p_SWS]
                f.write("p_SWS = np.array([%s])\n" % ", ".join(p_list))
            self._log("已保存: %s" % path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _save_cfg(self):
        try:
            data = {
                "phys":     self.edit_phys.collect(),
                "freq":     self.edit_freq.collect(),
                "init":     self.edit_init.collect(),
                "rc_basic": self.edit_rc_basic.collect(),
                "rc_adv":   self.edit_rc_adv.collect(),
            }
            cfg_dir = os.path.dirname(CONFIG_FILE)
            if cfg_dir:
                os.makedirs(cfg_dir, exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error("配置保存失败: %s", e)

    def _load_cfg(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, editor in [
                ("phys",     self.edit_phys),
                ("freq",     self.edit_freq),
                ("init",     self.edit_init),
                ("rc_basic", self.edit_rc_basic),
                ("rc_adv",   self.edit_rc_adv),
            ]:
                editor.apply(data.get(key, {}))
        except Exception as e:
            logger.error("配置加载失败: %s", e)

    def closeEvent(self, event):
        self._save_cfg()
        if self.task is not None and self.task.isRunning():
            self.task.stop()
            self.task.wait()
        event.accept()


if __name__ == "__main__":
    mp.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())