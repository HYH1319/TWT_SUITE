"""
行波管多频点 **多目标差分进化 (MODE)** 优化工具
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心算法：MODE = DE变异/交叉 + NSGA-II非支配排序 + 拥挤距离选择
目标：每个频点功率作为独立目标，输出完整 Pareto 前沿
界面：PyQt5，结果以表格列出所有可行方案并标注最优解

依赖：pip install numpy PyQt5
"""

import sys
import os
import json
import logging
import time
import traceback
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from datetime import datetime
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox,
    QProgressBar, QFileDialog, QGridLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush

from Noline_GAIN_MAINCALL_VUCORE_MIX_WITH_PVT import calculate_SEGMENT_TWT_NOLINE

# ── 全局常量 ──────────────────────────────────────────────────────
DE_STRATEGIES = [
    ("best/1/bin",             "best1bin"),
    ("rand/1/bin",             "rand1bin"),
    ("current-to-best/1/bin",  "currenttobest1bin"),
]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_WORKERS = max(1, cpu_count())
CONFIG_FILE = "./configUDF/OPT/twt_MODE_config_MUTIL.json"
N_FREQ = 3

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def adjust_params(p_SWS, idx, Vpc, Kc, coeffs):
    """螺距→电参数映射"""
    delta = (p_SWS[idx] - p_SWS[0]) / p_SWS[0]
    return {
        "Vpc": Vpc + coeffs['vpc'] * delta * Vpc,
        "Kc":  Kc  + coeffs['kc']  * delta * Kc,
    }


def evaluate_single_freq(args):
    """
    【最细粒度并行单元】单个体 × 单频点 的 TWT 评估。
    进程池调度的最小任务，消除了原版"个体内部频点串行"的瓶颈。

    args: (p_SWS, fixed_params, freq_data, coeffs, ind_idx, freq_idx)
    返回: (ind_idx, freq_idx, -power)  — 负号→最小化
    """
    p_SWS, fixed_params, freq_data, coeffs, ind_idx, freq_idx = args
    try:
        para_func = partial(adjust_params, coeffs=coeffs)
        result = calculate_SEGMENT_TWT_NOLINE(
            I=fixed_params["i"],
            V=fixed_params["v"],
            Kc=freq_data['Kc'],
            Loss_perunit=fixed_params["loss_perunit"],
            SectionedSEGMENT_IDX=fixed_params["section_seg_idx"],
            p_SWS=p_SWS,
            N_unit=fixed_params["n_unit"],
            w=fixed_params["w"],
            t=fixed_params["t"],
            Fn_K=fixed_params["Fn_K"],
            f0_GHz=freq_data['freq'],
            Vpc=freq_data['Vpc'],
            para_func=para_func,
            P_in=fixed_params["p_in"],
            Loss_attu=fixed_params["loss_attu"],
        )
        return (ind_idx, freq_idx, -result["输出功率P_out"])
    except Exception as e:
        logger.error(f"评估出错 ind={ind_idx} freq={freq_idx}: {e}")
        return (ind_idx, freq_idx, float("inf"))


# ═══════════════════════════════════════════════════════════════════
# NSGA-II 核心：非支配排序 + 拥挤距离
# ═══════════════════════════════════════════════════════════════════

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a 支配 b：a 在所有目标上 ≤ b，且至少一个目标 < b"""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F: np.ndarray):
    """
    NSGA-II 快速非支配排序——numpy 向量化版。
    利用广播一次性计算全部 (i,j) 支配关系，消除 O(N²) 的 Python 双重循环。

    F: (pop_size, n_obj) 目标值矩阵（最小化）
    返回: 层级列表，fronts[0] 为 Pareto 前沿个体索引列表
    """
    n = len(F)
    # Fi[i,j,:] = F[i] - F[j]，broadcasting: (n,1,m) - (1,n,m)
    diff = F[:, None, :] - F[None, :, :]        # (n, n, m)
    # i 支配 j：F[i] 在所有维度 ≤ F[j] 且至少一维 <
    leq  = (diff <= 0).all(axis=2)              # (n, n)  F[i]≤F[j]
    lt   = (diff <  0).any(axis=2)              # (n, n)  F[i]<F[j] 某维
    dom  = leq & lt                             # dom[i,j]=True 表示 i 支配 j

    # 被支配计数
    n_dom = dom.sum(axis=0).astype(int)         # dom 列和 = 支配 j 的人数
    # 支配集合：用列表存储（后续需要逐层传播）
    S = [list(np.where(dom[i])[0]) for i in range(n)]

    fronts = [list(np.where(n_dom == 0)[0])]
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(F: np.ndarray, front: list) -> np.ndarray:
    """计算某层前沿内各个体的拥挤距离"""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n)
    F_front = F[front]
    n_obj = F.shape[1]

    for m in range(n_obj):
        order = np.argsort(F_front[:, m])
        f_min = F_front[order[0], m]
        f_max = F_front[order[-1], m]
        dist[order[0]] = dist[order[-1]] = np.inf
        rng = f_max - f_min if f_max != f_min else 1e-12
        for k in range(1, n - 1):
            dist[order[k]] += (F_front[order[k + 1], m] - F_front[order[k - 1], m]) / rng

    return dist


def nsga2_select(pop: np.ndarray, F: np.ndarray, pop_size: int):
    """
    从合并种群（父代+子代）中用 NSGA-II 选择下一代。
    返回选中个体的索引（长度 = pop_size）。
    """
    fronts = fast_non_dominated_sort(F)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            cd = crowding_distance(F, front)
            order = np.argsort(-cd)           # 拥挤距离大的优先
            needed = pop_size - len(selected)
            selected.extend([front[i] for i in order[:needed]])
            break

    return selected


# ═══════════════════════════════════════════════════════════════════
# MODE 优化线程（替换原 DistributedDETask）
# ═══════════════════════════════════════════════════════════════════

class MODETask(QThread):
    """
    多目标差分进化（MODE）优化线程。

    算法流程：
      1. 初始化种群并评估多目标适应度
      2. 每代：DE变异 → 交叉 → 评估子代 → 合并 → NSGA-II选择
      3. 输出 Pareto 前沿解集
    """
    finished          = pyqtSignal(object)   # dict 或 None
    progress_updated  = pyqtSignal(int, object)  # (进度%, pareto_F)
    log_message       = pyqtSignal(str)
    iteration_complete = pyqtSignal(int, object)  # (iter, pareto_F)
    initial_evaluated  = pyqtSignal(object)       # initial pareto_F

    def __init__(self, fixed_params, freq_params, coeffs, bounds,
                 pop_size, max_iter, workers,
                 strategy='best1bin', F=0.5, CR=0.7, initial_p=None):
        super().__init__()
        self.fixed_params = fixed_params
        self.freq_params  = freq_params
        self.coeffs       = coeffs
        self.bounds       = bounds
        self.pop_size     = pop_size
        self.max_iter     = max_iter
        self.workers      = workers
        self.strategy     = strategy
        self.F            = F
        self.CR           = CR
        self.initial_p    = initial_p
        self.is_running   = False
        self.pool         = None

        self.bounds_low  = np.array([b[0] for b in bounds])
        self.bounds_high = np.array([b[1] for b in bounds])
        self.n_dim       = len(bounds)

    # ── 主流程 ────────────────────────────────────────────────────
    def run(self):
        try:
            self.is_running = True
            self.log_message.emit(f"启动进程池 ({self.workers} 进程)…")
            self.pool = Pool(processes=self.workers)

            # 初始化种群
            pop = np.random.uniform(self.bounds_low, self.bounds_high,
                                    (self.pop_size, self.n_dim))
            if self.initial_p is not None and len(self.initial_p) == self.n_dim:
                pop[0] = np.array(self.initial_p)
                self.log_message.emit("已植入用户初始参数到种群[0]")

            # 初始评估
            self.log_message.emit("评估初始种群…")
            fitness = self._batch_evaluate(pop)   # (pop_size, n_obj)

            # 初始 Pareto 前沿
            fronts = fast_non_dominated_sort(fitness)
            pf_idx = fronts[0]
            pf_F   = -fitness[pf_idx]   # 转回正功率供显示
            self.initial_evaluated.emit(pf_F)
            self.log_message.emit(
                f"初始 Pareto 前沿大小: {len(pf_idx)}  "
                f"| 最大各频点功率: {(-fitness[pf_idx].min(axis=0)).tolist()}"
            )

            # 主迭代
            for it in range(1, self.max_iter + 1):
                if not self.is_running:
                    break

                # DE变异 + 交叉 → 子代
                mutant = self._mutate(pop, fitness)
                trial  = self._crossover(pop, mutant)

                # 评估子代
                trial_fitness = self._batch_evaluate(trial)

                # 合并父代+子代 → NSGA-II 选择
                combined_pop = np.vstack([pop,   trial])
                combined_fit = np.vstack([fitness, trial_fitness])

                selected = nsga2_select(combined_pop, combined_fit, self.pop_size)
                pop     = combined_pop[selected]
                fitness = combined_fit[selected]

                # 当代 Pareto 前沿
                fronts = fast_non_dominated_sort(fitness)
                pf_idx = fronts[0]
                pf_F   = -fitness[pf_idx]    # 正功率

                progress = int(it / self.max_iter * 100)
                self.progress_updated.emit(progress, pf_F)
                self.iteration_complete.emit(it, pf_F)

                if it % 1 == 0 or it == self.max_iter:
                    self.log_message.emit(
                        f"迭代 {it}/{self.max_iter} | "
                        f"Pareto前沿大小={len(pf_idx)} | "
                        f"当前最大各频点功率={pf_F.max(axis=0).tolist()}"
                    )

            # 最终输出：直接用最后一代的 Pareto 前沿（主循环末尾已更新）
            fronts     = fast_non_dominated_sort(fitness)
            pf_idx     = fronts[0]
            pareto_pop = pop[pf_idx]
            pareto_F   = -fitness[pf_idx]   # 正功率，shape=(n_pareto, n_freq)

            if self.is_running:
                self.finished.emit({
                    'pareto_pop': pareto_pop,
                    'pareto_F':   pareto_F,
                    'freq_labels': [f"频点{i+1}({self.freq_params[i]['freq']:.1f}GHz)"
                                    for i in range(len(self.freq_params))]
                })

        except Exception as e:
            self.log_message.emit(f"MODE 出错: {e}\n{traceback.format_exc()}")
            self.finished.emit(None)
        finally:
            self._cleanup()

    # ── 批量评估（个体 × 频点 全并行）────────────────────────────
    def _batch_evaluate(self, pop: np.ndarray) -> np.ndarray:
        """
        将 pop_size × n_freq 个任务展平成一个大列表，
        一次 pool.map 让所有 CPU 核心同时处理「个体-频点」组合，
        并行任务数 = pop_size × N_FREQ，比原版多 N_FREQ 倍。
        使用 chunksize 减少进程间通信开销。
        """
        n_pop  = len(pop)
        n_freq = len(self.freq_params)

        # 展平：每个 (个体i, 频点j) 作为独立任务
        flat_args = [
            (pop[i], self.fixed_params, self.freq_params[j],
             self.coeffs, i, j)
            for i in range(n_pop)
            for j in range(n_freq)
        ]

        # chunksize 经验值：总任务 / (4 × workers)，平衡调度开销与负载均衡
        chunksize = max(1, len(flat_args) // (4 * self.workers))
        raw = self.pool.map(evaluate_single_freq, flat_args, chunksize=chunksize)

        # 重新组装 (pop_size, n_freq)
        fitness = np.full((n_pop, n_freq), float("inf"))
        for ind_idx, freq_idx, val in raw:
            fitness[ind_idx, freq_idx] = val
        return fitness

    # ── DE 变异（向量化）─────────────────────────────────────────
    def _mutate(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        向量化变异，保证每个个体采样到的 r1/r2/r3 互不相同且不等于自身。

        正确方法：
          1. 构造 (n_pop, n_pop) 的候选矩阵，每行是 [0..n_pop-1]
          2. 将对角线元素（自身）与最后一列交换，然后截掉最后一列
          3. 对每行的 n_pop-1 个候选独立随机排列，取前3列作为 r1/r2/r3
        """
        n_pop, n_dim = pop.shape

        if 'best' in self.strategy:
            fronts   = fast_non_dominated_sort(fitness)
            best_vec = pop[np.random.choice(fronts[0])]

        # 构造候选矩阵，去掉自身
        candidates = np.tile(np.arange(n_pop), (n_pop, 1))        # (n_pop, n_pop)
        diag_vals  = candidates[np.arange(n_pop), np.arange(n_pop)].copy()
        last_vals  = candidates[:, -1].copy()
        candidates[np.arange(n_pop), np.arange(n_pop)] = last_vals  # 对角线换到末尾
        candidates[:, -1] = diag_vals
        others = candidates[:, :-1]                                 # (n_pop, n_pop-1)

        # 每行独立随机打乱，取前3个
        perm = np.argsort(np.random.rand(n_pop, n_pop - 1), axis=1)
        r1 = others[np.arange(n_pop), perm[:, 0]]
        r2 = others[np.arange(n_pop), perm[:, 1]]
        r3 = others[np.arange(n_pop), perm[:, 2]]

        if self.strategy == 'rand1bin':
            mutants = pop[r1] + self.F * (pop[r2] - pop[r3])
        elif self.strategy == 'best1bin':
            mutants = best_vec + self.F * (pop[r1] - pop[r2])
        elif self.strategy == 'currenttobest1bin':
            mutants = (pop
                       + self.F * (best_vec - pop)
                       + self.F * (pop[r1] - pop[r2]))
        else:
            mutants = pop[r1] + self.F * (pop[r2] - pop[r3])

        return np.clip(mutants, self.bounds_low, self.bounds_high)

    # ── DE 交叉 ───────────────────────────────────────────────────
    def _crossover(self, pop: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        n_pop = pop.shape[0]
        mask = np.random.rand(n_pop, self.n_dim) < self.CR
        rand_dims = np.random.randint(0, self.n_dim, n_pop)
        mask[np.arange(n_pop), rand_dims] = True
        return np.where(mask, mutant, pop)

    # ── 停止 & 清理 ───────────────────────────────────────────────
    def stop(self):
        self.is_running = False
        self._cleanup()

    def _cleanup(self):
        if self.pool:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception:
                pass
            self.pool = None

    def __del__(self):
        self._cleanup()


# ═══════════════════════════════════════════════════════════════════
# 基础 UI 组件（与原版一致）
# ═══════════════════════════════════════════════════════════════════

class BaseEditor(QWidget):
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.widgets = {}
        layout = QFormLayout()

        for field in fields:
            if len(field) == 2 and field[1] == "label":
                layout.addRow(QLabel(field[0]))
                continue

            label_text, widget_type, *args = field
            label = QLabel(label_text)

            if widget_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
                if len(args) > 3: widget.setSingleStep(args[3])
                if len(args) > 4: widget.setDecimals(args[4])
            elif widget_type == "int":
                widget = QSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
            elif widget_type == "text":
                widget = QLineEdit()
                if args: widget.setText(args[0])
            elif widget_type == "combo":
                widget = QComboBox()
                for item in args:
                    if isinstance(item, tuple) and len(item) == 2:
                        widget.addItem(item[0], item[1])
                    else:
                        widget.addItem(str(item))

            layout.addRow(label, widget)
            self.widgets[label_text] = widget

        self.setLayout(layout)

    def get_value(self, key):
        w = self.widgets.get(key)
        if w is None: return None
        if isinstance(w, (QDoubleSpinBox, QSpinBox)): return w.value()
        if isinstance(w, QLineEdit): return w.text()
        if isinstance(w, QComboBox):
            return w.currentData() if w.currentData() else w.currentText()
        return None

    def set_value(self, key, value):
        w = self.widgets.get(key)
        if w is None: return
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            try: w.setValue(float(value))
            except Exception: pass
        elif isinstance(w, QLineEdit):
            w.setText(str(value))
        elif isinstance(w, QComboBox):
            idx = w.findData(value)
            if idx >= 0: w.setCurrentIndex(idx)
            else:
                idx = w.findText(str(value))
                if idx >= 0: w.setCurrentIndex(idx)


class FixedParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("电子枪参数", "label"),
            ("电流 I (A):",      "double", 0, 3e3,     0.3,    0.001, 4),
            ("电压 V (V):",      "double", 0, 3e6, 23000,  1,   0),
            ("慢波结构参数", "label"),
            ("每单元损耗参数:", "double", 0, 3e3,  0.0,    0.001, 4),
            ("衰减段索引 (逗号分隔):", "text", "1"),
            ("各段周期数 (逗号分隔):", "text", "20,5,10,10,10,10"),
            ("束流参数", "label"),
            ("束流宽度 w (mm):", "double", 0, 3e3,   0.2,    0.001, 4),
            ("束流厚度 t (mm):", "double", 0, 3e3,   0.2,    0.001, 4),
            ("填充参数", "label"),
            ("填充率倒数 Fn_K:", "double", 1, 3e3,   1.0,    0.001, 4),
            ("输入/衰减参数", "label"),
            ("输入功率 (W):",    "double", 0, 3e3,  0.1,    0.001, 4),
            ("衰减量 (dB):",     "double", 0, 3e3,  20,     1,     1),
            ("参数调整系数", "label"),
            ("Vpc 调整系数:",    "double", 0, 3e3,   0.82,   0.001, 4),
            ("Kc 调整系数:",     "double", 0, 3e3,   1.6,    0.001, 4),
            ("频点1参数", "label"),
            ("频点1 Kc值 (Ω):", "double", 0, 3e3,   2.67,   0.001, 4),
            ("频点1 Vpc值 (c):", "double", 0, 1,    0.2867, 0.001, 4),
            ("频点1 频率 (GHz):","double", 0, 3e3, 220.0,  1,     1),
            ("频点2参数", "label"),
            ("频点2 Kc值 (Ω):", "double", 0, 3e3,   2.67,   0.001, 4),
            ("频点2 Vpc值 (c):", "double", 0, 1,    0.2867, 0.001, 4),
            ("频点2 频率 (GHz):","double", 0, 3e3, 225.0,  1,     1),
            ("频点3参数", "label"),
            ("频点3 Kc值 (Ω):", "double", 0, 3e3,   2.67,   0.001, 4),
            ("频点3 Vpc值 (c):", "double", 0, 1,    0.2867, 0.001, 4),
            ("频点3 频率 (GHz):","double", 0, 3e3, 230.0,  1,     1),
        ]
        super().__init__(fields, parent)

    def get_params(self):
        try:
            def parse_list(s, dtype=float):
                return [dtype(v.strip()) for v in s.split(',') if v.strip()]

            section_seg_idx = parse_list(self.get_value("衰减段索引 (逗号分隔):"), int)
            n_unit          = parse_list(self.get_value("各段周期数 (逗号分隔):"), int)
            n_dim = len(n_unit)

            params = {
                "i":            self.get_value("电流 I (A):"),
                "v":            self.get_value("电压 V (V):"),
                "loss_perunit": self.get_value("每单元损耗参数:"),
                "section_seg_idx": section_seg_idx,
                "n_unit":       n_unit,
                "w":            self.get_value("束流宽度 w (mm):"),
                "t":            self.get_value("束流厚度 t (mm):"),
                "Fn_K":         self.get_value("填充率倒数 Fn_K:"),
                "p_in":         self.get_value("输入功率 (W):"),
                "loss_attu":    self.get_value("衰减量 (dB):"),
                "vpc_coeff":    self.get_value("Vpc 调整系数:"),
                "kc_coeff":     self.get_value("Kc 调整系数:"),
                "n_dim":        n_dim,
            }
            for i in range(1, 4):
                params[f'freq{i}_kc']  = self.get_value(f"频点{i} Kc值 (Ω):")
                params[f'freq{i}_vpc'] = self.get_value(f"频点{i} Vpc值 (c):")
                params[f'freq{i}_freq']= self.get_value(f"频点{i} 频率 (GHz):")
            return params
        except Exception as e:
            QMessageBox.warning(self, "输入错误", f"参数格式无效: {e}")
            return None


class MODEConfigEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("优化策略:",           "combo", *DE_STRATEGIES),
            ("种群大小:",           "int",    10, 1000, 100),
            ("最大迭代次数:",       "int",    10, 5000, 200),
            ("变异因子 (F):",       "double", 0.1, 2.0, 0.5, 0.01, 3),
            ("交叉概率 (CR):",      "double", 0.0, 1.0, 0.7, 0.01, 2),
            ("参数下限 (逗号分隔):","text",  "0.499,0.490,0.49,0.49,0.49,0.49"),
            ("参数上限 (逗号分隔):","text",  "0.501,0.510,0.51,0.51,0.51,0.51"),
            ("并行进程数:",         "int",    1, MAX_WORKERS, MAX_WORKERS),
        ]
        super().__init__(fields, parent)

    def get_config(self, n_dim):
        try:
            lb = [float(v.strip()) for v in self.get_value("参数下限 (逗号分隔):").split(",") if v.strip()]
            ub = [float(v.strip()) for v in self.get_value("参数上限 (逗号分隔):").split(",") if v.strip()]
            if len(lb) == 1 and n_dim > 1: lb = lb * n_dim
            if len(ub) == 1 and n_dim > 1: ub = ub * n_dim
            if len(lb) != n_dim or len(ub) != n_dim:
                raise ValueError(f"边界维度必须等于参数维度({n_dim})")
            return {
                "strategy": self.get_value("优化策略:"),
                "pop_size": self.get_value("种群大小:"),
                "max_iter": self.get_value("最大迭代次数:"),
                "F":        self.get_value("变异因子 (F):"),
                "CR":       self.get_value("交叉概率 (CR):"),
                "bounds":   list(zip(lb, ub)),
                "workers":  min(self.get_value("并行进程数:"), MAX_WORKERS),
            }
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


class InitialParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [("初始p_SWS值 (逗号分隔):", "text", "0.50,0.50,0.50,0.50,0.50,0.50")]
        super().__init__(fields, parent)

    def get_initial_p(self, n_dim):
        try:
            vals = [float(v.strip()) for v in
                    self.get_value("初始p_SWS值 (逗号分隔):").split(",") if v.strip()]
            if len(vals) == 1 and n_dim > 1: vals = vals * n_dim
            if len(vals) != n_dim:
                raise ValueError(f"维度({len(vals)})必须等于参数维度({n_dim})")
            return vals
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


# ═══════════════════════════════════════════════════════════════════
# Pareto 前沿展示面板
# ═══════════════════════════════════════════════════════════════════

class ParetoPanel(QWidget):
    """
    Pareto 前沿解集展示面板。
    - 纯表格，每行一个方案，列为各频点功率 + 总功率 + p_SWS
    - 总功率最大行以绿色背景标注为推荐最优解
    - 点击任意行可回填到初始参数编辑器
    """
    solution_selected = pyqtSignal(int)

    # 颜色
    COLOR_BEST   = QColor("#c8e6c9")   # 淡绿：推荐最优解
    COLOR_NORMAL = QColor("#ffffff")
    COLOR_SELECT = QColor("#bbdefb")   # 淡蓝：当前选中行

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pareto_pop   = None
        self._pareto_F     = None
        self._freq_labels  = []
        self._best_row     = -1
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 说明标签
        self.info_lbl = QLabel("等待优化结果…")
        self.info_lbl.setStyleSheet("color:#555; font-style:italic; padding:2px;")
        layout.addWidget(self.info_lbl)

        # 表格
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet("""
            QTableWidget { font-family:'Consolas','Courier New',monospace; font-size:10pt; }
            QHeaderView::section { background:#e8e8e8; font-weight:bold; padding:4px; }
        """)
        self.table.selectionModel().selectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

        self.hint_lbl = QLabel('点击行选择方案 → 点击"使用选中方案"回填到初始参数')
        self.hint_lbl.setStyleSheet("color:#666; font-size:9pt; padding:2px;")
        layout.addWidget(self.hint_lbl)

    def update_pareto(self, pareto_pop: np.ndarray, pareto_F: np.ndarray,
                      freq_labels: list):
        self._pareto_pop  = pareto_pop
        self._pareto_F    = pareto_F
        self._freq_labels = freq_labels
        n_sol, n_obj = pareto_F.shape
        n_dim = pareto_pop.shape[1]

        # 推荐行（总功率最大）
        self._best_row = int(pareto_F.sum(axis=1).argmax())

        # 列定义：各频点功率 | 总功率 | p_SWS各分量
        psws_headers = [f"p[{i}]" for i in range(n_dim)]
        col_headers  = freq_labels + ["总功率(W)"] + psws_headers
        self.table.setColumnCount(len(col_headers))
        self.table.setHorizontalHeaderLabels(col_headers)
        self.table.setRowCount(n_sol)

        for row in range(n_sol):
            is_best = (row == self._best_row)
            bg = self.COLOR_BEST if is_best else self.COLOR_NORMAL

            # 各频点功率
            for col in range(n_obj):
                item = QTableWidgetItem(f"{pareto_F[row, col]:.3f}")
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QBrush(bg))
                if is_best:
                    font = item.font(); font.setBold(True); item.setFont(font)
                self.table.setItem(row, col, item)

            # 总功率
            total_item = QTableWidgetItem(f"{pareto_F[row].sum():.3f}")
            total_item.setTextAlignment(Qt.AlignCenter)
            total_item.setBackground(QBrush(bg))
            if is_best:
                font = total_item.font(); font.setBold(True); total_item.setFont(font)
            self.table.setItem(row, n_obj, total_item)

            # p_SWS 各分量
            for k, pv in enumerate(pareto_pop[row]):
                pitem = QTableWidgetItem(f"{pv:.6f}")
                pitem.setTextAlignment(Qt.AlignCenter)
                pitem.setBackground(QBrush(bg))
                self.table.setItem(row, n_obj + 1 + k, pitem)

        # 自动滚动到推荐行
        self.table.scrollToItem(self.table.item(self._best_row, 0))
        self.table.selectRow(self._best_row)

        total_powers = pareto_F.sum(axis=1)
        best_total   = total_powers[self._best_row]
        self.info_lbl.setText(
            f"共 {n_sol} 个 Pareto 最优方案  |  "
            f"推荐最优解（行 {self._best_row}，总功率最大）: {best_total:.3f} W  "
            f"[绿色行]"
        )
        self.info_lbl.setStyleSheet("color:#2e7d32; font-weight:bold; padding:2px;")

    def _on_selection(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.solution_selected.emit(rows[0].row())

    def get_selected_solution(self):
        rows = self.table.selectionModel().selectedRows()
        if rows and self._pareto_pop is not None:
            idx = rows[0].row()
            return self._pareto_pop[idx], self._pareto_F[idx]
        return None, None

    def get_best_solution(self):
        """返回推荐最优解（总功率最大）"""
        if self._pareto_pop is not None and self._best_row >= 0:
            return self._pareto_pop[self._best_row], self._pareto_F[self._best_row]
        return None, None


# ═══════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════

BUTTON_STYLE = {
    'green':  ("#4CAF50", "#45a049", "#a5d6a7"),
    'red':    ("#f44336", "#d32f2f", "#ffcdd2"),
    'blue':   ("#2196F3", "#1976D2", "#bbdefb"),
    'orange': ("#ff9800", "#f57c00", "#ffe0b2"),
}

def make_button(text, color_key, callback=None, enabled=True) -> QPushButton:
    normal, hover, disabled = BUTTON_STYLE[color_key]
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {normal}; color: white;
            font-weight: bold; padding: 10px; font-size: 12pt; border-radius: 5px;
        }}
        QPushButton:hover      {{ background-color: {hover}; }}
        QPushButton:disabled   {{ background-color: {disabled}; color: #888; }}
    """)
    if callback: btn.clicked.connect(callback)
    btn.setEnabled(enabled)
    return btn


class TWTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行波管多频点 MODE 多目标差分进化优化工具")
        self.setGeometry(100, 100, 1400, 900)
        self.optimization_thread = None
        self.optimization_result = None
        self.start_time = None
        self._init_ui()
        self._load_settings()

        self.status_bar = self.statusBar()
        self.cpu_info = f"CPU核心: {cpu_count()} | 最大进程: {MAX_WORKERS}"
        self.status_bar.showMessage(f"就绪 [MODE多目标版] | {self.cpu_info}")


    # ── 界面初始化 ────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── 左侧：参数面板 ─────────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        tabs.setFont(QFont("Arial", 10))
        self.fixed_params_editor = FixedParamsEditor()
        self.mode_editor         = MODEConfigEditor()
        self.initial_editor      = InitialParamsEditor()
        tabs.addTab(self.fixed_params_editor, "固定参数")
        tabs.addTab(self.mode_editor,         "MODE配置")
        tabs.addTab(self.initial_editor,      "初始参数")
        ll.addWidget(tabs)

        ctrl = QHBoxLayout()
        self.run_btn   = make_button("▶ 开始优化",  'green',  self.start_optimization)
        self.stop_btn  = make_button("■ 停止",      'red',    self.stop_optimization, False)
        self.save_btn  = make_button("💾 保存结果", 'blue',   self.save_results,       False)
        self.reset_btn = make_button("↺ 重置",      'orange', self.reset_ui)
        for b in [self.run_btn, self.stop_btn, self.save_btn, self.reset_btn]:
            ctrl.addWidget(b)
        ll.addLayout(ctrl)

        # 方案选择按钮
        self.use_solution_btn = make_button("✔ 使用选中方案", 'blue', self._use_selected_solution, False)
        ll.addWidget(self.use_solution_btn)

        splitter.addWidget(left)

        # ── 右侧：状态 + 日志 + Pareto 面板 ──────────────────────
        right_splitter = QSplitter(Qt.Vertical)

        # 状态区
        top_right = QWidget()
        trl = QVBoxLayout(top_right)
        trl.setContentsMargins(4, 4, 4, 4)

        status_grp = QGroupBox("优化状态")
        sg = QGridLayout(status_grp)
        self._status_vals = {}
        for row, (key, label, color) in enumerate([
            ("pareto_size", "Pareto前沿大小", "#e0e0e0"),
            ("max_total",   "最大总功率(W)",   "#c8e6c9"),
        ]):
            sg.addWidget(QLabel(f"{label}:"), row, 0)
            vl = QLabel("—")
            vl.setAlignment(Qt.AlignCenter)
            vl.setStyleSheet(f"border:1px solid #b0b0b0; border-radius:5px; padding:5px; "
                             f"min-height:28px; font-weight:bold; background:{color}; font-size:11pt;")
            sg.addWidget(vl, row, 1)
            self._status_vals[key] = vl

        info_row = QHBoxLayout()
        self.elapsed_lbl = QLabel("用时: —")
        self.iter_lbl    = QLabel("迭代: —")
        for lbl in [self.elapsed_lbl, self.iter_lbl]:
            lbl.setStyleSheet("font-weight:bold;")
            info_row.addWidget(lbl)
        sg.addLayout(info_row, 2, 0, 1, 2)
        trl.addWidget(status_grp)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { text-align:center; border:1px solid #b0b0b0;
                           border-radius:5px; height:22px; font-weight:bold; }
            QProgressBar::chunk { background-color:#4CAF50; }
        """)
        trl.addWidget(self.progress_bar)

        log_grp = QGroupBox("优化日志")
        log_lay = QVBoxLayout(log_grp)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet(
            "font-family:'Consolas','Courier New',monospace; font-size:10pt; "
            "background:#f8f8f8; border:1px solid #d0d0d0; border-radius:4px;")
        log_lay.addWidget(self.log_area)
        trl.addWidget(log_grp, 1)

        right_splitter.addWidget(top_right)

        # Pareto 面板
        self.pareto_panel = ParetoPanel()
        self.pareto_panel.solution_selected.connect(self._on_solution_selected)
        pareto_wrap = QGroupBox("Pareto 前沿分析")
        pw = QVBoxLayout(pareto_wrap)
        pw.addWidget(self.pareto_panel)
        right_splitter.addWidget(pareto_wrap)
        right_splitter.setSizes([350, 550])

        splitter.addWidget(right_splitter)
        splitter.setSizes([420, 980])

        self._apply_global_style()

    def _apply_global_style(self):
        self.setStyleSheet("""
            QMainWindow { background:#f5f5f5; font-family:'Segoe UI','Microsoft YaHei',sans-serif; }
            QTabWidget::pane { border:1px solid #c0c0c0; background:white; border-radius:6px; }
            QTabBar::tab { background:#e0e0e0; border:1px solid #c0c0c0; padding:8px 15px;
                           border-top-left-radius:5px; border-top-right-radius:5px; font-weight:bold; }
            QTabBar::tab:selected { background:white; border-bottom-color:white; }
            QGroupBox { border:1px solid #b0b0b0; border-radius:6px; margin-top:15px;
                        font-weight:bold; color:#303030; padding-top:15px; background:white; }
            QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top center;
                               padding:0 5px; background:#f0f0f0; border:1px solid #b0b0b0;
                               border-radius:4px; margin-top:-12px; }
        """)

    # ── 优化控制 ──────────────────────────────────────────────────
    def start_optimization(self):
        fixed_params = self.fixed_params_editor.get_params()
        if fixed_params is None: return

        n_dim = fixed_params["n_dim"]
        cfg   = self.mode_editor.get_config(n_dim)
        if cfg is None: return

        initial_p = self.initial_editor.get_initial_p(n_dim)
        if initial_p is None: return

        freq_params = [{'Kc': fixed_params[f'freq{i}_kc'],
                        'Vpc': fixed_params[f'freq{i}_vpc'],
                        'freq': fixed_params[f'freq{i}_freq']} for i in range(1, 4)]
        coeffs = {'vpc': fixed_params['vpc_coeff'], 'kc': fixed_params['kc_coeff']}

        # ── 启动前：并行计算初始参数各频点功率 ────────────────────
        self.log_area.clear()
        self.log_message("===== MODE 多目标差分进化优化 =====")
        self.log_message("计算初始参数各频点功率…")
        QApplication.processEvents()

        try:
            p_arr = np.array(initial_p)
            flat_args = [
                (p_arr, fixed_params, freq_params[j], coeffs, 0, j)
                for j in range(len(freq_params))
            ]
            with Pool(processes=min(len(freq_params), MAX_WORKERS)) as pool:
                raw = pool.map(evaluate_single_freq, flat_args)

            init_powers = {}
            for _, freq_idx, neg_pwr in raw:
                init_powers[freq_idx] = -neg_pwr

            self.log_message("─── 初始参数各频点功率 ───")
            total_init = 0.0
            for j, fp in enumerate(freq_params):
                pwr = init_powers.get(j, float('nan'))
                total_init += pwr
                self.log_message(
                    f"  频点{j+1} ({fp['freq']:.1f} GHz): {pwr:.3f} W"
                )
            self.log_message(f"  初始总功率: {total_init:.3f} W")
            self.log_message("─────────────────────────")
        except Exception as e:
            self.log_message(f"初始功率计算失败: {e}")
            QMessageBox.critical(self, "计算错误", f"初始功率计算失败:\n{e}")
            return

        # ── 启动优化线程 ───────────────────────────────────────────
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.use_solution_btn.setEnabled(False)
        self.log_message(f"参数维度: {n_dim} | 频点数: {N_FREQ} | "
                         f"种群: {cfg['pop_size']} | 迭代: {cfg['max_iter']} | "
                         f"进程: {cfg['workers']}")
        self.progress_bar.setValue(0)
        self.start_time = time.time()

        self.optimization_thread = MODETask(
            fixed_params=fixed_params,
            freq_params=freq_params,
            coeffs=coeffs,
            bounds=cfg["bounds"],
            pop_size=cfg["pop_size"],
            max_iter=cfg["max_iter"],
            workers=cfg["workers"],
            strategy=cfg["strategy"],
            F=cfg["F"], CR=cfg["CR"],
            initial_p=initial_p,
        )
        self.optimization_thread.log_message.connect(self.log_message)
        self.optimization_thread.progress_updated.connect(self._on_progress)
        self.optimization_thread.iteration_complete.connect(self._on_iteration)
        self.optimization_thread.initial_evaluated.connect(self._on_initial)
        self.optimization_thread.finished.connect(self._on_finished)
        self.optimization_thread.start()

    def stop_optimization(self):
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.log_message("优化已停止")
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)

    # ── 信号处理 ──────────────────────────────────────────────────
    def _on_initial(self, pf_F: np.ndarray):
        self._status_vals["pareto_size"].setText(str(len(pf_F)))

    def _on_progress(self, progress: int, pf_F: np.ndarray):
        self.progress_bar.setValue(progress)
        if pf_F is not None and len(pf_F) > 0:
            self._status_vals["max_total"].setText(f"{pf_F.sum(axis=1).max():.2f}")
        if self.start_time:
            elapsed = time.time() - self.start_time
            m, s = divmod(elapsed, 60)
            self.elapsed_lbl.setText(f"用时: {int(m):02d}:{int(s):02d}")

    def _on_iteration(self, it: int, pf_F: np.ndarray):
        self.iter_lbl.setText(f"迭代: {it}")
        if pf_F is not None and len(pf_F) > 0:
            self._status_vals["pareto_size"].setText(str(len(pf_F)))

    def _on_finished(self, result):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        if result is None:
            self.log_message("===== 优化失败 =====")
            self.status_bar.showMessage("优化失败 | " + self.cpu_info)
            return

        self.optimization_result = result
        pareto_F    = result['pareto_F']
        pareto_pop  = result['pareto_pop']
        freq_labels = result['freq_labels']
        best_idx    = int(pareto_F.sum(axis=1).argmax())

        self.log_message(f"\n===== 优化完成 =====")
        self.log_message(f"Pareto 前沿共 {len(pareto_F)} 个方案")
        self.log_message("─── 推荐最优解（总功率最大，方案 "
                         f"{best_idx}）───")
        for i, lbl in enumerate(freq_labels):
            self.log_message(f"  {lbl}: {pareto_F[best_idx, i]:.3f} W")
        self.log_message(f"  总功率: {pareto_F[best_idx].sum():.3f} W")
        p_best = pareto_pop[best_idx]
        self.log_message(f"  p_SWS: [{', '.join(f'{v:.6f}' for v in p_best)}]")
        self.log_message("─────────────────────────────────────")
        self.log_message("各方案功率范围（供参考）：")
        for i, lbl in enumerate(freq_labels):
            self.log_message(
                f"  {lbl}: {pareto_F[:,i].min():.3f} ~ {pareto_F[:,i].max():.3f} W")

        self.pareto_panel.update_pareto(pareto_pop, pareto_F, freq_labels)
        self._status_vals["pareto_size"].setText(str(len(pareto_F)))
        self._status_vals["max_total"].setText(f"{pareto_F[best_idx].sum():.2f}")
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage(
            f"优化完成 | Pareto={len(pareto_F)} 方案 | "
            f"最优总功率={pareto_F[best_idx].sum():.2f} W | {self.cpu_info}"
        )

    def _on_solution_selected(self, row: int):
        self.use_solution_btn.setEnabled(True)
        p, F = self.pareto_panel.get_selected_solution()
        if F is not None:
            self.log_message(
                f"已选方案[{row}] | 各频点功率: {[f'{v:.3f}W' for v in F]} | 总功率: {F.sum():.3f}W")

    def _use_selected_solution(self):
        p_SWS, F = self.pareto_panel.get_selected_solution()
        if p_SWS is None:
            QMessageBox.information(self, "提示", "请先在 Pareto 前沿表格中选择一个方案")
            return
        # 将选中方案回填到初始参数编辑器
        p_str = ",".join(f"{v:.6f}" for v in p_SWS)
        self.initial_editor.set_value("初始p_SWS值 (逗号分隔):", p_str)
        self.log_message(f"已将选中方案回填到初始参数: {p_str}")
        QMessageBox.information(self, "已回填",
            f'选中方案的 p_SWS 已回填到"初始参数"标签页。\n'
            f"各频点功率: {[f'{v:.3f}W' for v in F]}\n总功率: {F.sum():.3f}W")

    # ── 保存结果 ──────────────────────────────────────────────────
    def save_results(self):
        if self.optimization_result is None:
            QMessageBox.warning(self, "保存失败", "没有可用的优化结果")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存优化结果",
            f"TWT_MODE优化结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;所有文件 (*)")
        if not path: return
        if not path.endswith('.txt'): path += '.txt'

        try:
            result     = self.optimization_result
            pareto_F   = result['pareto_F']
            pareto_pop = result['pareto_pop']
            labels     = result['freq_labels']

            with open(path, 'w', encoding='utf-8') as f:
                f.write("===== 行波管多频点 MODE 多目标差分进化优化结果 =====\n\n")
                f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pareto 前沿大小: {len(pareto_F)}\n\n")

                f.write("─── Pareto 前沿解集 ───\n")
                header = "方案\t" + "\t".join(labels) + "\t总功率(W)\t" + \
                         "\t".join([f"p_SWS[{i}]" for i in range(pareto_pop.shape[1])])
                f.write(header + "\n")

                for k in range(len(pareto_F)):
                    row_vals = [str(k)] + \
                               [f"{pareto_F[k, j]:.4f}" for j in range(pareto_F.shape[1])] + \
                               [f"{pareto_F[k].sum():.4f}"] + \
                               [f"{v:.6f}" for v in pareto_pop[k]]
                    f.write("\t".join(row_vals) + "\n")

                # 推荐方案：总功率最大
                best_idx = pareto_F.sum(axis=1).argmax()
                f.write(f"\n─── 推荐方案（总功率最大）：方案 {best_idx} ───\n")
                f.write(f"总功率: {pareto_F[best_idx].sum():.4f} W\n")
                for j, lbl in enumerate(labels):
                    f.write(f"  {lbl}: {pareto_F[best_idx, j]:.4f} W\n")
                p_list = [f"{v:.6f}" for v in pareto_pop[best_idx]]
                f.write(f'\noptimized_p_SWS = np.array([{", ".join(p_list)}])\n')

            self.log_message(f"结果已保存至: {path}")
            QMessageBox.information(self, "保存成功", f"优化结果已保存到:\n{path}")
        except Exception as e:
            self.log_message(f"保存失败: {e}")
            QMessageBox.critical(self, "保存失败", str(e))

    # ── 配置持久化 ────────────────────────────────────────────────
    def _save_settings(self):
        try:
            settings = {
                "fixed_params":  {k: self.fixed_params_editor.get_value(k)
                                   for k in self.fixed_params_editor.widgets},
                "mode_config":   {k: self.mode_editor.get_value(k)
                                   for k in self.mode_editor.widgets},
                "initial_params":{k: self.initial_editor.get_value(k)
                                   for k in self.initial_editor.widgets},
            }
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            logger.warning(f"保存配置失败: {e}")

    def _load_settings(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                for k, v in s.get("fixed_params", {}).items():
                    self.fixed_params_editor.set_value(k, v)
                for k, v in s.get("mode_config", {}).items():
                    self.mode_editor.set_value(k, v)
                for k, v in s.get("initial_params", {}).items():
                    self.initial_editor.set_value(k, v)
        except Exception as e:
            logger.warning(f"加载配置失败: {e}")

    def log_message(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{ts}] {msg}")
        self.log_area.ensureCursorVisible()

    def reset_ui(self):
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.stop_optimization()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.use_solution_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_area.clear()
        for v in self._status_vals.values(): v.setText("—")
        self.elapsed_lbl.setText("用时: —")
        self.iter_lbl.setText("迭代: —")
        self.status_bar.showMessage(f"已重置 | {self.cpu_info}")

    def closeEvent(self, event):
        self._save_settings()
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.optimization_thread.wait()
        event.accept()


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        mp.freeze_support()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = TWTMainWindow()
    win.show()
    sys.exit(app.exec_())