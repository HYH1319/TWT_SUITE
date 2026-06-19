# -*- coding: utf-8 -*-
"""
行波管多频点 RVEA 多目标进化优化工具 (独立算法实现版)
依赖: pip install numpy PyQt5
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  第一部分: 导入与常量                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

import sys, os, json, logging, traceback, time, math
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count, Pool
from datetime import datetime
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar,
    QFileDialog, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush

from Noline_GAIN_MAINCALL_VUCORE_MIX_WITH_PVT import (
    calculate_SEGMENT_TWT_NOLINE,
)

logger = logging.getLogger(__name__)

EVAL_PENALTY = 1e4
N_FREQ = 3
MAX_WORKERS = max(1, cpu_count())
CONFIG_FILE = "./configUDF/OPT/twt_RVEA_config_MUTIL.json"

_LBL_F1KC = "频点1 Kc值 (Ω):"
_LBL_F1VPC = "频点1 Vpc值:"
_LBL_F1F   = "频点1 频率"
_LBL_F2KC = "频点2 Kc值 (Ω):"
_LBL_F2VPC = "频点2 Vpc值:"
_LBL_F2F   = "频点2 频率"
_LBL_F3KC = "频点3 Kc值 (Ω):"
_LBL_F3VPC = "频点3 Vpc值:"
_LBL_F3F   = "频点3 频率"
_LBL_DB    = "衰减量"
_FILE_FLT  = "文本文件 (*.txt);;所有文件 (*)"


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第二部分: 共享数据类 (算法与GUI之间的数据契约)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class FreqParam:
    Kc:   float
    Vpc:  float
    freq: float

@dataclass
class FixedParams:
    i:              float
    v:              float
    loss_perunit:   float
    section_seg_idx: List[int]
    n_unit:         List[int]
    w:              float
    t:              float
    Fn_K:           float
    p_in:           float
    loss_attu:      float
    vpc_coeff:      float
    kc_coeff:       float

    @property
    def n_dim(self) -> int:
        return len(self.n_unit)

@dataclass
class Coeffs:
    vpc: float
    kc:  float

@dataclass
class RVEAConfig:
    n_partitions:   int   = 12
    pop_size:       int   = 0        # 0 = 自动
    max_iter:       int   = 200
    prob_crossover: float = 0.9
    eta_crossover:  float = 15.0
    eta_mutation:   float = 20.0
    alpha:          float = 2.0      # RVEA 角度惩罚系数
    workers:        int   = 1

    @staticmethod
    def ref_dir_count(p: int, m: int) -> int:
        return math.comb(p + m - 1, m - 1)

    def n_ref_dirs_for(self, n_obj: int) -> int:
        return self.ref_dir_count(self.n_partitions, n_obj)

@dataclass
class OptimizationResult:
    pareto_pop:   np.ndarray
    pareto_F:     np.ndarray          # 已取反 → 正功率
    freq_labels:  List[str]
    best_idx:     int
    elapsed_sec:  float = 0.0

    @property
    def n_solutions(self) -> int:
        return len(self.pareto_F)

    @property
    def best_p_SWS(self) -> np.ndarray:
        return self.pareto_pop[self.best_idx]

    @property
    def best_total_power(self) -> float:
        return float(self.pareto_F[self.best_idx].sum())

# 回调类型别名
ProgressCallback  = Callable[[int, np.ndarray], None]
IterationCallback = Callable[[int, np.ndarray], None]
InitialCallback   = Callable[[np.ndarray], None]
LogCallback       = Callable[[str], None]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第三部分: 算法核心 (RVEA 纯 Python 实现, 不依赖任何第三方库)    ║
# ║  ----------------------------------------------------------------║
# ║  本部分完全独立, 不引用任何 Qt 组件。                             ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 3.1 物理模型评估工具 ──────────────────────────────────────────

def adjust_params(p_SWS, idx, Vpc, Kc, coeffs: Coeffs):
    delta = (p_SWS[idx] - p_SWS[0]) / p_SWS[0]
    return {
        "Vpc": Vpc + coeffs.vpc * delta * Vpc,
        "Kc":  Kc  + coeffs.kc  * delta * Kc,
    }


def _eval_single(args):
    """进程池最小任务：评估 (个体, 频点) 的负输出功率 (最小化)"""
    p_SWS, fixed, fp, coeffs, ind_idx, freq_idx = args
    try:
        para_func = partial(adjust_params, coeffs=coeffs)
        result = calculate_SEGMENT_TWT_NOLINE(
            I=fixed.i, V=fixed.v, Kc=fp.Kc,
            Loss_perunit=fixed.loss_perunit,
            SectionedSEGMENT_IDX=fixed.section_seg_idx,
            p_SWS=p_SWS, N_unit=fixed.n_unit,
            w=fixed.w, t=fixed.t, Fn_K=fixed.Fn_K,
            f0_GHz=fp.freq, Vpc=fp.Vpc,
            para_func=para_func, P_in=fixed.p_in,
            Loss_attu=fixed.loss_attu,
        )
        return (ind_idx, freq_idx, -result["输出功率P_out"])
    except Exception as e:
        logger.error("评估出错 ind=%d freq=%d: %s", ind_idx, freq_idx, e)
        return (ind_idx, freq_idx, EVAL_PENALTY)


def evaluate_single_point(p_SWS, fixed, fp, coeffs):
    """评估单个 (个体, 频点) 的输出功率 (W)，失败返回 NaN"""
    try:
        para_func = partial(adjust_params, coeffs=coeffs)
        result = calculate_SEGMENT_TWT_NOLINE(
            I=fixed.i, V=fixed.v, Kc=fp.Kc,
            Loss_perunit=fixed.loss_perunit,
            SectionedSEGMENT_IDX=fixed.section_seg_idx,
            p_SWS=p_SWS, N_unit=fixed.n_unit,
            w=fixed.w, t=fixed.t, Fn_K=fixed.Fn_K,
            f0_GHz=fp.freq, Vpc=fp.Vpc,
            para_func=para_func, P_in=fixed.p_in,
            Loss_attu=fixed.loss_attu,
        )
        return result["输出功率P_out"]
    except Exception as e:
        logger.error("评估出错: %s", e)
        return float("nan")


# ── 3.2 Das-Dennis 参考方向生成 ──────────────────────────────────

def das_dennis_ref_dirs(n_obj: int, n_partitions: int) -> np.ndarray:
    """
    生成 Das-Dennis 均匀参考方向。
    返回形状 (n_ref, n_obj)，每行是一个参考方向，分量之和 = 1。
    """
    if n_partitions == 0:
        return np.eye(n_obj)

    def _gen(cur, remaining, dims_left):
        if dims_left == 1:
            yield cur + [remaining]
        else:
            for k in range(remaining + 1):
                yield from _gen(cur + [k], remaining - k, dims_left - 1)

    partitions = list(_gen([], n_partitions, n_obj))
    return np.array(partitions, dtype=float) / n_partitions


# ── 3.3 快速非支配排序 ──────────────────────────────────────────

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """判断 a 是否支配 b (最小化意义)"""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    """
    快速非支配排序。
    返回按前沿分组的索引列表，fronts[0] 为 Pareto 前沿。
    """
    N = F.shape[0]
    domination_count = np.zeros(N, dtype=int)
    dominated_set = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            if _dominates(F[i], F[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(F[j], F[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    fronts = []
    current = [i for i in range(N) if domination_count[i] == 0]

    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        current = nxt

    return fronts


def non_dominated_indices(F: np.ndarray) -> np.ndarray:
    """返回第一前沿 (Pareto 前沿) 的索引数组"""
    fronts = fast_non_dominated_sort(F)
    return np.array(fronts[0], dtype=int) if fronts else np.array([], dtype=int)


# ── 3.4 SBX 交叉 ────────────────────────────────────────────────

def sbx_crossover(p1, p2, xl, xu, prob=0.9, eta=15.0):
    """模拟二进制交叉"""
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    if np.random.random() > prob:
        return c1, c2

    for i in range(n):
        if np.random.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-14:
            continue

        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = np.random.random()

        if u <= 0.5:
            bq = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            bq = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

        c1[i] = 0.5 * ((y1 + y2) - bq * (y2 - y1))
        c2[i] = 0.5 * ((y1 + y2) + bq * (y2 - y1))

        c1[i] = np.clip(c1[i], xl[i], xu[i])
        c2[i] = np.clip(c2[i], xl[i], xu[i])

        if np.random.random() > 0.5:
            c1[i], c2[i] = c2[i], c1[i]

    return c1, c2


# ── 3.5 PM 变异 ─────────────────────────────────────────────────

def pm_mutation(x, xl, xu, eta=20.0, prob=None):
    """多项式变异"""
    n = len(x)
    y = x.copy()
    if prob is None:
        prob = 1.0 / n

    for i in range(n):
        if np.random.random() > prob:
            continue
        delta = xu[i] - xl[i]
        if delta < 1e-14:
            continue

        u = np.random.random()
        if u < 0.5:
            deltaq = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        else:
            deltaq = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))

        y[i] = x[i] + deltaq * delta
        y[i] = np.clip(y[i], xl[i], xu[i])

    return y


# ── 3.6 RVEA 算法类 ─────────────────────────────────────────────

class RVEAAlgorithm:
    """
    RVEA (Reference Vector Guided Evolutionary Algorithm) 完整实现。
    包含原论文中的 Reference Vector Adaptation (参考向量自适应) 机制。
    """

    def __init__(self, ref_dirs: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                 pop_size: int,
                 prob_crossover: float = 0.9,
                 eta_crossover: float  = 15.0,
                 eta_mutation: float   = 20.0,
                 alpha: float = 2.0,
                 adapt_freq: float = 0.1):  # 新增：自适应频率 (占总代数的比例)
        self.original_ref_dirs = ref_dirs.copy() # 保存原始参考向量
        self.ref_dirs      = ref_dirs.copy()
        self.n_ref         = len(ref_dirs)
        self.n_obj         = ref_dirs.shape[1]
        self.n_var         = len(xl)
        self.xl            = np.asarray(xl, dtype=float)
        self.xu            = np.asarray(xu, dtype=float)
        self.pop_size      = max(pop_size, self.n_ref)
        self.prob_crossover = prob_crossover
        self.eta_crossover  = eta_crossover
        self.eta_mutation   = eta_mutation
        self.alpha          = alpha
        self.adapt_freq     = adapt_freq
        
        self._update_ref_dirs_unit()

    def _update_ref_dirs_unit(self):
        """重新计算单位参考向量"""
        norms = np.linalg.norm(self.ref_dirs, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        self.ref_dirs_unit = self.ref_dirs / norms

    # ── 初始化种群 ──────────────────────────────────────────────
    def initialize(self, initial_x: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.random.uniform(self.xl, self.xu, (self.pop_size, self.n_var))
        if initial_x is not None:
            X[0] = np.clip(initial_x, self.xl, self.xu)
        return X

    # ── 产生子代 (增加重复检查) ────────────────────────────────
    def create_offspring(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        offspring = np.empty((self.pop_size, self.n_var))
        for i in range(0, self.pop_size, 2):
            p1 = self._tournament_select(F)
            p2 = self._tournament_select(F)
            c1, c2 = sbx_crossover(X[p1], X[p2], self.xl, self.xu,
                                    self.prob_crossover, self.eta_crossover)
            c1 = pm_mutation(c1, self.xl, self.xu, self.eta_mutation)
            c2 = pm_mutation(c2, self.xl, self.xu, self.eta_mutation)
            offspring[i] = c1
            if i + 1 < self.pop_size:
                offspring[i + 1] = c2
        return offspring

    # ── 环境选择 ────────────────────────────────────────────────
    def select(self, X: np.ndarray, F: np.ndarray,
               offspring_X: np.ndarray, offspring_F: np.ndarray,
               gen: int, max_gen: int) -> Tuple[np.ndarray, np.ndarray]:
        combined_X = np.vstack([X, offspring_X])
        combined_F = np.vstack([F, offspring_F])
        sel_idx = self._environmental_selection(combined_F, gen, max_gen)
        return combined_X[sel_idx], combined_F[sel_idx]

    # ── ★ 新增：参考向量自适应 ★ ────────────────────────────────
    def adapt_reference_vectors(self, F: np.ndarray):
        """
        根据当前非支配前沿的截距，缩放原始参考向量。
        这是 RVEA 算法的核心机制之一，用于处理不规则 Pareto 前沿。
        """
        # 1. 获取非支配解
        pf_idx = non_dominated_indices(F)
        if len(pf_idx) < self.n_obj:
            return # 点太少无法构造超平面
        
        pf_F = F[pf_idx]
        
        # 2. 归一化 (计算截距)
        ideal = pf_F.min(axis=0)
        F_t = pf_F - ideal
        
        intercepts = np.full(self.n_obj, 1.0)
        for m in range(self.n_obj):
            w = np.full(self.n_obj, 1e-6)
            w[m] = 1.0
            asf = np.max(F_t / w, axis=1)
            extreme_idx = int(np.argmin(asf))
            
            # 简化版截距计算：直接用极值点在该目标的值
            intercepts[m] = F_t[extreme_idx, m] if F_t[extreme_idx, m] > 1e-14 else 1.0
            
        # 3. 缩放原始参考向量
        self.ref_dirs = self.original_ref_dirs * intercepts
        self._update_ref_dirs_unit()

    # ── 内部: 锦标赛选择 ────────────────────────────────────────
    def _tournament_select(self, F: np.ndarray, k: int = 2) -> int:
        candidates = np.random.randint(0, len(F), k)
        best = candidates[0]
        for c in candidates[1:]:
            if _dominates(F[c], F[best]):
                best = c
            elif not _dominates(F[best], F[c]) and np.random.random() < 0.5:
                best = c
        return int(best)

    # ── 内部: RVEA 环境选择核心 ─────────────────────────────────
    def _environmental_selection(self, F: np.ndarray,
                                  gen: int, max_gen: int) -> np.ndarray:
        N = F.shape[0]
        n_select = self.pop_size

        F_norm = self._normalize(F)

        # 使用当前（可能已自适应的）参考向量
        F_len = np.linalg.norm(F_norm, axis=1, keepdims=True)
        F_len[F_len < 1e-14] = 1.0
        F_unit = F_norm / F_len                       
        cos_theta = F_unit @ self.ref_dirs_unit.T      

        association = np.argmax(cos_theta, axis=1)      
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))   
        d = F_len.ravel()                                    

        t_ratio = gen / max_gen if max_gen > 0 else 1.0
        M = float(self.n_obj)

        selected = []
        used = set()
        for j in range(self.n_ref):
            mask = (association == j)
            if not np.any(mask):
                continue
            indices = np.where(mask)[0]
            apd = d[indices] * (1.0 + M * theta[indices, j] * (t_ratio ** self.alpha))
            best_local = indices[int(np.argmin(apd))]
            selected.append(best_local)
            used.add(best_local)

        if len(selected) < n_select:
            remaining = np.array([i for i in range(N) if i not in used])
            if len(remaining) > 0:
                rem_apd = np.empty(len(remaining))
                for k, i in enumerate(remaining):
                    j = association[i]
                    rem_apd[k] = d[i] * (1.0 + M * theta[i, j] * (t_ratio ** self.alpha))
                order = np.argsort(rem_apd)
                for idx in order:
                    selected.append(remaining[idx])
                    if len(selected) >= n_select:
                        break

        return np.array(selected[:n_select], dtype=int)

    # ── 内部: 目标归一化 (ASF 截距法，增强鲁棒性) ──────────────
    def _normalize(self, F: np.ndarray) -> np.ndarray:
        ideal = F.min(axis=0)
        F_t = F - ideal
        
        # 尝试用完整的超平面法求截距
        extreme = np.zeros(self.n_obj, dtype=int)
        for m in range(self.n_obj):
            w = np.full(self.n_obj, 1e-6)
            w[m] = 1.0
            asf = np.max(F_t / w, axis=1)
            extreme[m] = int(np.argmin(asf))

        try:
            ext_F = F_t[extreme]
            if np.linalg.matrix_rank(ext_F) >= self.n_obj:
                a = np.linalg.solve(ext_F, np.ones(self.n_obj))
                intercepts = 1.0 / a
                # 检查截距是否合理 (必须为正)
                if np.all(intercepts > 1e-14):
                    return F_t / intercepts
        except (np.linalg.LinAlgError, ValueError):
            pass

        # 增强鲁棒性：如果超平面法失败，退化为极值点法
        maxv = F_t.max(axis=0)
        maxv[maxv < 1e-14] = 1.0
        return F_t / maxv


# ── 3.7 优化器封装 ──────────────────────────────────────────────

class RVEAOptimizer:
    """
    行波管多频点 RVEA 多目标优化器。

    ★ 职责:
      - 管理 RVEAAlgorithm 实例和进程池
      - 将算法循环、回调通知、结果提取整合在一起
      - 对外暴露 run() / stop() / evaluate_initial() 接口

    ★ 与 GUI 的耦合方式:
      - 仅通过 ProgressCallback / IterationCallback / LogCallback 通知进度
      - GUI 不传入任何 Qt 对象
    """

    def __init__(self,
                 fixed:       FixedParams,
                 freq_params: List[FreqParam],
                 coeffs:      Coeffs,
                 bounds:      List[Tuple[float, float]],
                 config:      RVEAConfig,
                 on_progress:  Optional[ProgressCallback]  = None,
                 on_iteration: Optional[IterationCallback] = None,
                 on_initial:   Optional[InitialCallback]   = None,
                 on_log:       Optional[LogCallback]       = None):
        self.fixed       = fixed
        self.freq_params = freq_params
        self.coeffs      = coeffs
        self.bounds      = bounds
        self.config      = config
        self._on_progress  = on_progress
        self._on_iteration = on_iteration
        self._on_initial   = on_initial
        self._on_log       = on_log
        self._is_running   = False
        self._pool         = None

    # ── 公开接口 ────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._is_running

    def stop(self):
        self._is_running = False

    def run(self, initial_p=None) -> Optional[OptimizationResult]:
        """执行 RVEA 优化 (阻塞)，返回 OptimizationResult 或 None"""
        t0 = time.time()

        if hasattr(sys, "frozen"):
            mp.freeze_support()
            mp.set_executable(sys.executable)

        try:
            self._is_running = True
            workers = max(1, self.config.workers)
            self._log("初始化 RVEA 优化 (%d 进程)..." % workers)
            self._pool = Pool(processes=workers)

            n_obj = len(self.freq_params)
            n_var = len(self.bounds)
            xl = np.array([b[0] for b in self.bounds])
            xu = np.array([b[1] for b in self.bounds])

            # 参考方向
            n_ref   = self.config.n_ref_dirs_for(n_obj)
            ref_dirs = das_dennis_ref_dirs(n_obj, self.config.n_partitions)

            pop_size = self.config.pop_size if self.config.pop_size > 0 else n_ref
            pop_size = max(pop_size, n_ref)
            self._log("种群大小调整为 %d (匹配 %d 参考向量)" % (pop_size, n_ref))

            # 创建算法实例
            algo = RVEAAlgorithm(
                ref_dirs=ref_dirs, xl=xl, xu=xu, pop_size=pop_size,
                prob_crossover=self.config.prob_crossover,
                eta_crossover=self.config.eta_crossover,
                eta_mutation=self.config.eta_mutation,
                alpha=self.config.alpha,
            )

            self._log(
                "RVEA: %d目标 | %d变量 | %d个体 | %d参考向量 | "
                "SBX(p=%.2f,η=%.1f) + PM(η=%.1f) | α=%.1f" % (
                    n_obj, n_var, pop_size, n_ref,
                    self.config.prob_crossover, self.config.eta_crossover,
                    self.config.eta_mutation, self.config.alpha))

            # ── 初始种群 ──
            init_x = np.array(initial_p) if initial_p is not None else None
            X = algo.initialize(initial_x=init_x)
            F = self._evaluate_population(X)

            # 初始回调
            pf_idx = non_dominated_indices(F)
            pf_F   = -F[pf_idx] if len(pf_idx) > 0 else np.empty((0, n_obj))
            if self._on_initial:
                self._on_initial(pf_F)

            max_gen = self.config.max_iter

            # ── 主循环 ──
            for gen in range(1, max_gen + 1):
                if not self._is_running:
                    self._log("优化被用户停止，提取当前结果...")
                    break

                offspring_X = algo.create_offspring(X, F)
                offspring_F = self._evaluate_population(offspring_X)

                X, F = algo.select(X, F, offspring_X, offspring_F, gen, max_gen)

                # ★★★ 触发参考向量自适应 (例如每 10% 的代数触发一次) ★★★
                if max_gen > 0 and (gen % max(1, int(max_gen * 0.1)) == 0 or gen == max_gen):
                    algo.adapt_reference_vectors(F)
                # 回调
                pf_idx = non_dominated_indices(F)
                pf_F   = -F[pf_idx] if len(pf_idx) > 0 else np.empty((0, n_obj))

                progress = int(gen / max_gen * 100)
                if self._on_progress:
                    self._on_progress(progress, pf_F)
                if self._on_iteration:
                    self._on_iteration(gen, pf_F)

                if gen % 5 == 0 or gen == max_gen:
                    if self._on_log and len(pf_F) > 0:
                        self._on_log(
                            "Generation %d/%d | Pareto前沿=%d | "
                            "最大各频点功率=%s" % (
                                gen, max_gen, len(pf_idx),
                                pf_F.max(axis=0).tolist()))

            return self._extract_result(F, X, time.time() - t0)

        except Exception as e:
            self._log("RVEA 出错: %s\n%s" % (e, traceback.format_exc()))
            return None
        finally:
            self._cleanup()

    def evaluate_initial(self, p_SWS) -> List[float]:
        """评估初始参数各频点功率 (W)，返回 list"""
        p = np.array(p_SWS)
        return [evaluate_single_point(p, self.fixed, fp, self.coeffs)
                for fp in self.freq_params]

    # ── 内部方法 ────────────────────────────────────────────────

    def _evaluate_population(self, X: np.ndarray) -> np.ndarray:
        """并行评估种群, 返回目标值矩阵 F (最小化: 负功率)"""
        n_pop  = X.shape[0]
        n_freq = len(self.freq_params)
        flat_args = []
        for i in range(n_pop):
            for j in range(n_freq):
                flat_args.append((X[i], self.fixed, self.freq_params[j],
                                  self.coeffs, i, j))

        workers   = max(1, self.config.workers)
        chunksize = max(1, len(flat_args) // (4 * workers))
        raw = self._pool.map(_eval_single, flat_args, chunksize=chunksize)

        F = np.full((n_pop, n_freq), EVAL_PENALTY)
        for ind_idx, freq_idx, val in raw:
            F[ind_idx, freq_idx] = val
        return F

    def _log(self, msg: str):
        if self._on_log:
            self._on_log(msg)
        else:
            logger.info(msg)

    def _make_freq_labels(self) -> List[str]:
        return ["频点%d(%.1fGHz)" % (i + 1, fp.freq)
                for i, fp in enumerate(self.freq_params)]

    def _extract_result(self, F: np.ndarray, X: np.ndarray,
                        elapsed: float) -> OptimizationResult:
        pf_idx = non_dominated_indices(F)
        if len(pf_idx) == 0:
            return OptimizationResult(
                pareto_pop=np.empty((0, len(self.bounds))),
                pareto_F=np.empty((0, len(self.freq_params))),
                freq_labels=self._make_freq_labels(),
                best_idx=-1, elapsed_sec=elapsed)

        pareto_F   = -F[pf_idx]       # 取反 → 正功率
        pareto_pop = X[pf_idx]
        best_idx   = int(pareto_F.sum(axis=1).argmax())
        return OptimizationResult(
            pareto_pop=pareto_pop, pareto_F=pareto_F,
            freq_labels=self._make_freq_labels(),
            best_idx=best_idx, elapsed_sec=elapsed)

    def _cleanup(self):
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第四部分: GUI                                                   ║
# ║  ----------------------------------------------------------------║
# ║  本部分依赖第三部分的算法类和数据类, 但算法部分完全不依赖 GUI。    ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 4.1 工作线程 ────────────────────────────────────────────────

class RVEATask(QThread):
    """将 RVEAOptimizer.run() 移至后台线程执行"""
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


# ── 4.2 通用编辑器基类 ──────────────────────────────────────────

class BaseEditor(QWidget):
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.widgets = {}
        layout = QFormLayout()
        for field in fields:
            if len(field) == 2 and field[1] == "label":
                layout.addRow(QLabel(field[0]))
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
            if widget is not None:
                layout.addRow(label, widget)
                self.widgets[key] = widget
        self.setLayout(layout)

    def get(self, key):
        w = self.widgets.get(key)
        if w is None: return None
        if isinstance(w, (QDoubleSpinBox, QSpinBox)): return w.value()
        if isinstance(w, QLineEdit): return w.text()
        return None

    def put(self, key, val):
        w = self.widgets.get(key)
        if w is None: return
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            try: w.setValue(float(val))
            except Exception: pass
        elif isinstance(w, QLineEdit):
            w.setText(str(val))

    def collect(self):
        return {k: self.get(k) for k in self.widgets}

    def apply(self, data):
        if not isinstance(data, dict): return
        for k, v in data.items():
            self.put(k, v)


# ── 4.3 固定参数编辑器 ──────────────────────────────────────────

class FixedParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("电子枪参数", "label"),
            ("电流 I (A):",          "double", 0, 3e3, 0.3,    0.001, 4),
            ("电压 V (V):",          "double", 0, 3e6, 23000,   1,     0),
            ("慢波结构参数", "label"),
            ("每单元损耗参数:",       "double", 0, 3e3, 0.0,    0.001, 4),
            ("衰减段索引 (逗号分隔):", "text",  "1"),
            ("各段周期数 (逗号分隔):", "text",  "20,5,10,10,10,10"),
            ("束流参数", "label"),
            ("束流宽度 w (mm):",     "double", 0, 3e3, 0.2,    0.001, 4),
            ("束流厚度 t (mm):",     "double", 0, 3e3, 0.2,    0.001, 4),
            ("填充参数", "label"),
            ("填充率倒数 Fn_K:",      "double", 1, 3e3, 1.0,    0.001, 4),
            ("输入/衰减参数", "label"),
            ("输入功率 (W):",        "double", 0, 3e3, 0.1,    0.001, 4),
            (_LBL_DB,                "double", 0, 3e3, 20,      1,     1),
            ("参数调整系数", "label"),
            ("Vpc 调整系数:",        "double", 0, 3e3, 0.82,   0.001, 4),
            ("Kc 调整系数:",         "double", 0, 3e3, 1.6,    0.001, 4),
            ("频点1参数", "label"),
            (_LBL_F1KC,  "double", 0, 3e3, 2.67,   0.001, 4),
            (_LBL_F1VPC, "double", 0, 1,   0.2867, 0.001, 4),
            (_LBL_F1F,   "double", 0, 3e3, 220.0,  1,     1),
            ("频点2参数", "label"),
            (_LBL_F2KC,  "double", 0, 3e3, 2.67,   0.001, 4),
            (_LBL_F2VPC, "double", 0, 1,   0.2867, 0.001, 4),
            (_LBL_F2F,   "double", 0, 3e3, 225.0,  1,     1),
            ("频点3参数", "label"),
            (_LBL_F3KC,  "double", 0, 3e3, 2.67,   0.001, 4),
            (_LBL_F3VPC, "double", 0, 1,   0.2867, 0.001, 4),
            (_LBL_F3F,   "double", 0, 3e3, 230.0,  1,     1),
        ]
        super().__init__(fields, parent)

    def get_params(self):
        try:
            def pl(s, dt=float):
                return [dt(v.strip()) for v in s.split(",") if v.strip()]
            return FixedParams(
                i=float(self.get("电流 I (A):")),
                v=float(self.get("电压 V (V):")),
                loss_perunit=float(self.get("每单元损耗参数:")),
                section_seg_idx=pl(self.get("衰减段索引 (逗号分隔):"), int),
                n_unit=pl(self.get("各段周期数 (逗号分隔):"), int),
                w=float(self.get("束流宽度 w (mm):")),
                t=float(self.get("束流厚度 t (mm):")),
                Fn_K=float(self.get("填充率倒数 Fn_K:")),
                p_in=float(self.get("输入功率 (W):")),
                loss_attu=float(self.get(_LBL_DB)),
                vpc_coeff=float(self.get("Vpc 调整系数:")),
                kc_coeff=float(self.get("Kc 调整系数:")),
            )
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None

    def get_freqs(self):
        try:
            result = []
            for kc_k, vpc_k, f_k in [
                (_LBL_F1KC, _LBL_F1VPC, _LBL_F1F),
                (_LBL_F2KC, _LBL_F2VPC, _LBL_F2F),
                (_LBL_F3KC, _LBL_F3VPC, _LBL_F3F),
            ]:
                result.append(FreqParam(
                    Kc=float(self.get(kc_k)),
                    Vpc=float(self.get(vpc_k)),
                    freq=float(self.get(f_k)),
                ))
            return result
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


# ── 4.4 RVEA 配置编辑器 ─────────────────────────────────────────

class RVEAConfigEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("分区数 p:",              "int",    2, 100,  5),
            ("种群大小(0=自动):",       "int",    0, 10000, 0),
            ("最大迭代:",              "int",    10, 5000, 20),
            ("交叉概率:",              "double", 0.0, 1.0, 0.9, 0.01, 2),
            ("交叉eta:",               "double", 1,  100,  15,  1, 0),
            ("变异eta:",               "double", 1,  100,  20,  1, 0),
            ("APD惩罚指数α:",          "double", 0.1, 10.0, 2.0, 0.1, 1),
            ("参数下限:",              "text",   "0.499,0.490,0.49,0.49,0.49,0.49"),
            ("参数上限:",              "text",   "0.501,0.510,0.51,0.51,0.51,0.51"),
            ("并行数:",                "int",    1,  128,  min(32, MAX_WORKERS)),
        ]
        super().__init__(fields, parent)

    def get_config(self, n_dim):
        try:
            lb_s = self.get("参数下限:")
            ub_s = self.get("参数上限:")
            lb = [float(v.strip()) for v in lb_s.split(",") if v.strip()]
            ub = [float(v.strip()) for v in ub_s.split(",") if v.strip()]
            if len(lb) == 1 and n_dim > 1: lb = lb * n_dim
            if len(ub) == 1 and n_dim > 1: ub = ub * n_dim
            if len(lb) != n_dim or len(ub) != n_dim:
                raise ValueError("边界维度不匹配 (需要%d维)" % n_dim)
            return (RVEAConfig(
                n_partitions   = self.get("分区数 p:"),
                pop_size       = self.get("种群大小(0=自动):"),
                max_iter       = self.get("最大迭代:"),
                prob_crossover = self.get("交叉概率:"),
                eta_crossover  = self.get("交叉eta:"),
                eta_mutation   = self.get("变异eta:"),
                alpha          = self.get("APD惩罚指数α:"),
                workers        = min(self.get("并行数:"), MAX_WORKERS),
            ), list(zip(lb, ub)))
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None, None


# ── 4.5 初始参数编辑器 ──────────────────────────────────────────

class InitEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("初始p_SWS:", "text", "0.50,0.50,0.50,0.50,0.50,0.50"),
        ]
        super().__init__(fields, parent)

    def get_p(self, n_dim):
        try:
            txt  = self.get("初始p_SWS:")
            vals = [float(v.strip()) for v in txt.split(",") if v.strip()]
            if len(vals) == 1 and n_dim > 1: vals = vals * n_dim
            if len(vals) != n_dim:
                raise ValueError("维度不匹配 (需要%d维)" % n_dim)
            return vals
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


# ── 4.6 Pareto 前沿面板 ─────────────────────────────────────────

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

    def update_result(self, result: OptimizationResult):
        self._pop  = result.pareto_pop
        self._fval = result.pareto_F
        ns = result.n_solutions
        no = result.pareto_F.shape[1]
        nd = result.pareto_pop.shape[1]
        bi = result.best_idx

        hdr = result.freq_labels + ["总功率"] + ["p[%d]" % i for i in range(nd)]
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


# ── 4.7 主窗口 ──────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行波管 RVEA 多目标优化工具 (独立算法版)")
        self.setGeometry(100, 100, 1300, 850)
        self.task   = None
        self.result = None
        self.t0     = None
        self._init_ui()
        self._load_cfg()
        self.statusBar().showMessage("就绪 | CPU:%d核" % cpu_count())

    # ── UI 构建 ─────────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---- 左侧: 参数与控制 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(2, 2, 2, 2)

        tabs = QTabWidget()
        tabs.setFont(QFont("Arial", 10))
        self.edit_fp = FixedParamsEditor()
        self.edit_rc = RVEAConfigEditor()
        self.edit_ip = InitEditor()
        tabs.addTab(self.edit_fp, "固定参数")
        tabs.addTab(self.edit_rc, "RVEA配置")
        tabs.addTab(self.edit_ip, "初始参数")
        left_layout.addWidget(tabs)

        self.ref_lbl = QLabel("")
        self.ref_lbl.setStyleSheet(
            "color:#1565C0;font-weight:bold;padding:4px;"
            "background:#e3f2fd;border-radius:4px;")
        self._update_ref_info()
        left_layout.addWidget(self.ref_lbl)

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

        # ---- 右侧: 状态 + 日志 + Pareto ----
        right_splitter = QSplitter(Qt.Vertical)

        # 右上: 状态 & 日志
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

        # 右下: Pareto 面板
        self.pareto_panel = ParetoPanel()
        self.pareto_panel.solution_selected.connect(
            lambda _: self.btn_use.setEnabled(True))
        pareto_grp = QGroupBox("Pareto前沿")
        pg_layout = QVBoxLayout(pareto_grp)
        pg_layout.addWidget(self.pareto_panel)
        right_splitter.addWidget(pareto_grp)

        right_splitter.setSizes([350, 500])
        splitter.addWidget(right_splitter)
        splitter.setSizes([380, 900])

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
            p  = self.edit_rc.get("分区数 p:")
            nr = RVEAConfig.ref_dir_count(p, N_FREQ)
            self.ref_lbl.setText(
                "Das-Dennis(p=%d) -> %d参考向量 | 种群>=%d" % (p, nr, nr))
            if "rv" in self._status:
                self._status["rv"].setText(str(nr))
        except Exception:
            pass

    # ── 日志 ────────────────────────────────────────────────────

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append("[%s] %s" % (ts, msg))
        self.log_box.ensureCursorVisible()

    # ── 按钮动作 ────────────────────────────────────────────────

    def _start(self):
        fixed = self.edit_fp.get_params()
        freqs = self.edit_fp.get_freqs()
        if fixed is None or freqs is None:
            return
        cfg, bounds = self.edit_rc.get_config(fixed.n_dim)
        if cfg is None:
            return
        init_p = self.edit_ip.get_p(fixed.n_dim)
        if init_p is None:
            return

        coeffs = Coeffs(vpc=fixed.vpc_coeff, kc=fixed.kc_coeff)
        self._save_cfg()
        self.log_box.clear()
        self._log("===== 开始RVEA优化 =====")

        # 初始评估
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

        ps = cfg.pop_size if cfg.pop_size > 0 else nr
        self._log("维度:%d | 频点:%d | 种群:%d | 迭代:%d | 进程:%d" % (
            fixed.n_dim, len(freqs), ps, cfg.max_iter, cfg.workers))

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

    # ── 回调槽 ──────────────────────────────────────────────────

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

        self.pareto_panel.update_result(result)
        self._status["ps"].setText(str(result.n_solutions))
        self._status["mt"].setText("%.2f" % result.best_total_power)
        self.btn_save.setEnabled(True)

    def _use(self):
        p, F = self.pareto_panel.get_selected()
        if p is None:
            QMessageBox.information(self, "提示", "请先选择方案")
            return
        p_str = ",".join("%.6f" % v for v in p)
        self.edit_ip.put("初始p_SWS:", p_str)
        self._log("已回填p_SWS: %s" % p_str)

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
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("RVEA优化结果 | 时间:%s | 耗时:%.1fs | Pareto:%d\n" % (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    r.elapsed_sec, r.n_solutions))
                nd  = r.pareto_pop.shape[1]
                hdr = ["方案"] + r.freq_labels + ["总功率"]
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

    # ── 配置持久化 ──────────────────────────────────────────────

    def _save_cfg(self):
        try:
            data = {
                "fp": self.edit_fp.collect(),
                "rc": self.edit_rc.collect(),
                "ip": self.edit_ip.collect(),
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
            for key, editor in [("fp", self.edit_fp),
                                ("rc", self.edit_rc),
                                ("ip", self.edit_ip)]:
                editor.apply(data.get(key, {}))
        except Exception as e:
            logger.error("配置加载失败: %s", e)

    def closeEvent(self, event):
        self._save_cfg()
        if self.task is not None and self.task.isRunning():
            self.task.stop()
            self.task.wait()
        event.accept()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第五部分: 入口                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    mp.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
