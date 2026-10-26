# -*- coding: utf-8 -*-
"""
行波管多频点 RVEA 多目标优化 —— 计算核心 (严格按 RVEA 原论文实现)
依赖: pip install numpy

参照: R. Cheng, Y. Jin, M. Olhofer, B. Sendhoff,
      "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization",
      IEEE TEVC, vol.20, no.5, pp.773-791, 2016.

对外契约:
  数据类    : FreqParam / FixedParams / Coeffs / RVEAConfig / OptimizationResult
  评估工具  : adjust_params / evaluate_single_point
  算法主体  : RVEAAlgorithm / RVEAOptimizer

本模块不生成展示字符串、不写文件、无 __main__ 演示。
"""

import sys, logging, traceback, time, math
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER_MIXPVT_SUPER import (
    calculate_SEGMENT_TWT_NOLINE,
)

logger = logging.getLogger(__name__)

EVAL_PENALTY = 1e4


# ╔══════════════════════════════════════════════════════════════════╗
# ║  数据类                                                          ║
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
    precision:      str = "高精度"

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
    pop_size:       int   = 0        # 保留字段, 标准 RVEA 中种群严格 = n_ref, 此值被忽略
    max_iter:       int   = 200
    prob_crossover: float = 0.9
    eta_crossover:  float = 15.0
    eta_mutation:   float = 20.0
    alpha:          float = 2.0
    workers:        int   = 1

    @staticmethod
    def ref_dir_count(p: int, m: int) -> int:
        return math.comb(p + m - 1, m - 1)

    def n_ref_dirs_for(self, n_obj: int) -> int:
        return self.ref_dir_count(self.n_partitions, n_obj)


@dataclass
class OptimizationResult:
    pareto_pop:   np.ndarray
    pareto_F:     np.ndarray
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


ProgressCallback  = Callable[[int, np.ndarray], None]
IterationCallback = Callable[[int, np.ndarray], None]
InitialCallback   = Callable[[np.ndarray], None]
LogCallback       = Callable[[str], None]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  物理模型评估                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def adjust_params(p_SWS, idx, Vpc, Kc, coeffs: Coeffs):
    delta = (p_SWS[idx] - p_SWS[0]) / p_SWS[0]
    return {
        "Vpc": Vpc + coeffs.vpc * delta * Vpc,
        "Kc":  Kc  + coeffs.kc  * delta * Kc,
    }


def _eval_single(args):
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
            precision=fixed.precision,
        )
        return (ind_idx, freq_idx, -result["输出功率P_out"])
    except Exception as e:
        logger.error("评估出错 ind=%d freq=%d: %s", ind_idx, freq_idx, e)
        return (ind_idx, freq_idx, EVAL_PENALTY)


def evaluate_single_point(p_SWS, fixed, fp, coeffs):
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
            precision=fixed.precision,
        )
        return result["输出功率P_out"]
    except Exception as e:
        logger.error("评估出错: %s", e)
        return float("nan")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Das-Dennis 参考方向 (RVEA 论文 §III-A)                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def das_dennis_ref_dirs(n_obj: int, n_partitions: int) -> np.ndarray:
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║  非支配排序                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
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
    fronts = fast_non_dominated_sort(F)
    return np.array(fronts[0], dtype=int) if fronts else np.array([], dtype=int)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SBX 交叉 (带边界, Deb & Agrawal 1995)  ★RVEA正确性              ║
# ╚══════════════════════════════════════════════════════════════════╝

def sbx_crossover(p1, p2, xl, xu, prob=0.9, eta=15.0):
    """
    带边界约束的模拟二进制交叉 (Bounded SBX)。
    严格按 Deb 2001 NSGA-II 论文的边界版本, 而非"先算后 clip"简化版。
    """
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    if np.random.random() > prob:
        return c1, c2

    for i in range(n):
        if np.random.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-14:
            continue

        y1 = min(p1[i], p2[i])
        y2 = max(p1[i], p2[i])
        yl = xl[i]
        yu = xu[i]

        rand = np.random.random()

        # 子代 1: 向 yl 方向
        beta = 1.0 + 2.0 * (y1 - yl) / (y2 - y1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c1_i = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        # 子代 2: 向 yu 方向
        beta = 1.0 + 2.0 * (yu - y2) / (y2 - y1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c2_i = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        c1_i = float(np.clip(c1_i, yl, yu))
        c2_i = float(np.clip(c2_i, yl, yu))

        if np.random.random() < 0.5:
            c1[i], c2[i] = c2_i, c1_i
        else:
            c1[i], c2[i] = c1_i, c2_i

    return c1, c2


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PM 变异 (带边界, Deb & Goyal 1996)  ★RVEA正确性                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def pm_mutation(x, xl, xu, eta=20.0, prob=None):
    """
    带边界约束的多项式变异 (Bounded PM)。
    严格按 Deb & Goyal 1996 的边界版本。
    """
    n = len(x)
    y = x.copy()
    if prob is None:
        prob = 1.0 / n

    for i in range(n):
        if np.random.random() > prob:
            continue
        yl = xl[i]
        yu = xu[i]
        if (yu - yl) < 1e-14:
            continue

        delta1 = (x[i] - yl) / (yu - yl)
        delta2 = (yu - x[i]) / (yu - yl)
        rand = np.random.random()
        mut_pow = 1.0 / (eta + 1.0)

        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
            deltaq = 1.0 - val ** mut_pow

        y[i] = x[i] + deltaq * (yu - yl)
        y[i] = float(np.clip(y[i], yl, yu))

    return y


# ╔══════════════════════════════════════════════════════════════════╗
# ║  RVEA 算法 (严格按原论文 Algorithm 1 / 2 / 3)                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class RVEAAlgorithm:
    """
    RVEA (Reference Vector Guided Evolutionary Algorithm) 完整实现:
      - 归一化 (Algorithm 3): 极值点 + 超平面截距
      - APD 角度惩罚距离, 含 γ 归一化项
      - 参考向量自适应 (Algorithm 2): V_{t+1} = V_0 ∘ (z_max - z_min)
    """

    def __init__(self, ref_dirs: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                 prob_crossover: float = 0.9,
                 eta_crossover: float  = 15.0,
                 eta_mutation: float   = 20.0,
                 alpha: float = 2.0):
        self.original_ref_dirs = ref_dirs.copy()
        self.ref_dirs          = ref_dirs.copy()
        self.n_ref             = len(ref_dirs)
        self.n_obj             = ref_dirs.shape[1]
        self.n_var             = len(xl)
        self.xl                = np.asarray(xl, dtype=float)
        self.xu                = np.asarray(xu, dtype=float)
        self.prob_crossover    = prob_crossover
        self.eta_crossover     = eta_crossover
        self.eta_mutation      = eta_mutation
        self.alpha             = alpha
        self.gamma             = np.ones(self.n_ref)  # 占位, 下面 _update 会填
        self._update_ref_dirs_unit()

    # ── 参考向量单位化 + γ 计算 ★RVEA正确性 ────────────────────
    def _update_ref_dirs_unit(self):
        norms = np.linalg.norm(self.ref_dirs, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        self.ref_dirs_unit = self.ref_dirs / norms

        # γ_j = 参考向量 V_j 到最近邻参考向量的夹角 (原文 APD 中的归一化因子)
        G = self.ref_dirs_unit @ self.ref_dirs_unit.T
        np.fill_diagonal(G, -np.inf)
        cos_max = G.max(axis=1)
        gamma = np.arccos(np.clip(cos_max, -1.0, 1.0))
        self.gamma = np.clip(gamma, 1e-6, None)

    # ── 初始化 ──────────────────────────────────────────────────
    def initialize(self, initial_x: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.random.uniform(self.xl, self.xu, (self.n_ref, self.n_var))
        if initial_x is not None:
            X[0] = np.clip(initial_x, self.xl, self.xu)
        return X

    # ── 产生子代 ────────────────────────────────────────────────
    def create_offspring(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        n = len(X)
        offspring = np.empty((n, self.n_var))
        for i in range(0, n, 2):
            p1 = self._tournament_select(F)
            p2 = self._tournament_select(F)
            c1, c2 = sbx_crossover(X[p1], X[p2], self.xl, self.xu,
                                    self.prob_crossover, self.eta_crossover)
            c1 = pm_mutation(c1, self.xl, self.xu, self.eta_mutation)
            c2 = pm_mutation(c2, self.xl, self.xu, self.eta_mutation)
            offspring[i] = c1
            if i + 1 < n:
                offspring[i + 1] = c2
        return offspring

    # ── 环境选择 (一步到位) ─────────────────────────────────────
    def select(self, X: np.ndarray, F: np.ndarray,
               offspring_X: np.ndarray, offspring_F: np.ndarray,
               gen: int, max_gen: int) -> Tuple[np.ndarray, np.ndarray]:
        combined_X = np.vstack([X, offspring_X])
        combined_F = np.vstack([F, offspring_F])
        sel_idx = self._environmental_selection(combined_F, gen, max_gen)
        return combined_X[sel_idx], combined_F[sel_idx]

    # ── 参考向量自适应 (原文 Algorithm 2) ★RVEA正确性 ─────────
    def adapt_reference_vectors(self, F: np.ndarray):
        """
        V_{t+1} = V_0 ∘ (z_max - z_min)
        其中 z_max / z_min 为当前种群 (整个 P_t) 在各目标上的最大/最小值,
        而非只取 Pareto 前沿。
        """
        z_min = F.min(axis=0)
        z_max = F.max(axis=0)
        ranges = z_max - z_min
        ranges[ranges < 1e-14] = 1.0  # 防止某目标退化为常数时除零

        self.ref_dirs = self.original_ref_dirs * ranges
        self._update_ref_dirs_unit()

    # ── 锦标赛选择 ──────────────────────────────────────────────
    def _tournament_select(self, F: np.ndarray, k: int = 2) -> int:
        candidates = np.random.randint(0, len(F), k)
        best = candidates[0]
        for c in candidates[1:]:
            if _dominates(F[c], F[best]):
                best = c
            elif not _dominates(F[best], F[c]) and np.random.random() < 0.5:
                best = c
        return int(best)

    # ── RVEA 环境选择 (原文 Algorithm 1 Line 9-15) ★RVEA正确性 ─
    def _environmental_selection(self, F: np.ndarray,
                                  gen: int, max_gen: int) -> np.ndarray:
        """
        每个参考向量最多选 1 个解 (one-to-one), 无解映射则空;
        不进行"补满"操作 —— 严格遵循原文, 种群数可能 < n_ref。
        """
        F_norm = self._normalize(F)

        F_len = np.linalg.norm(F_norm, axis=1, keepdims=True)
        F_len[F_len < 1e-14] = 1.0
        F_unit = F_norm / F_len
        cos_theta = F_unit @ self.ref_dirs_unit.T

        association = np.argmax(cos_theta, axis=1)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        d = F_len.ravel()

        t_ratio = gen / max_gen if max_gen > 0 else 1.0
        M = float(self.n_obj)
        # APD_{i,j} = d_{i,j} * [1 + M * (t/t_max)^α * θ_{i,j} / γ_j]
        penalty = M * (t_ratio ** self.alpha)

        selected = []
        for j in range(self.n_ref):
            mask = (association == j)
            if not np.any(mask):
                continue
            indices = np.where(mask)[0]
            apd = d[indices] * (1.0 + penalty * theta[indices, j] / self.gamma[j])
            best_local = indices[int(np.argmin(apd))]
            selected.append(best_local)

        if not selected:
            # 极端保护: 全部映射到空, 返回全种群的 APD 最优
            apd_all = d * (1.0 + penalty * theta[np.arange(len(d)), association] /
                           self.gamma[association])
            selected = [int(np.argmin(apd_all))]

        return np.array(selected, dtype=int)

    # ── 归一化 (原文 Algorithm 3) ──────────────────────────────
    def _normalize(self, F: np.ndarray) -> np.ndarray:
        ideal = F.min(axis=0)
        F_t = F - ideal

        # 找极值点
        extreme = np.zeros(self.n_obj, dtype=int)
        for m in range(self.n_obj):
            w = np.full(self.n_obj, 1e-6)
            w[m] = 1.0
            asf = np.max(F_t / w, axis=1)
            extreme[m] = int(np.argmin(asf))

        # 解超平面求截距
        try:
            ext_F = F_t[extreme]
            if np.linalg.matrix_rank(ext_F) >= self.n_obj:
                a = np.linalg.solve(ext_F, np.ones(self.n_obj))
                intercepts = 1.0 / a
                if np.all(intercepts > 1e-14):
                    return F_t / intercepts
        except (np.linalg.LinAlgError, ValueError):
            pass

        # 退化: 用各目标的最大值 (原文退化处理)
        maxv = F_t.max(axis=0)
        maxv[maxv < 1e-14] = 1.0
        return F_t / maxv


class RVEAOptimizer:
    """
    行波管多频点 RVEA 多目标优化器 (严格 RVEA)。
    种群大小严格等于参考向量数 (n_ref)。
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

    def is_running(self) -> bool:
        return self._is_running

    def stop(self):
        self._is_running = False

    def run(self, initial_p=None) -> Optional[OptimizationResult]:
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

            n_ref    = self.config.n_ref_dirs_for(n_obj)
            ref_dirs = das_dennis_ref_dirs(n_obj, self.config.n_partitions)

            # ★RVEA正确性: 种群严格等于参考向量数, 忽略 config.pop_size
            self._log("种群大小 = %d (标准 RVEA 要求种群 = 参考向量数)" % n_ref)

            algo = RVEAAlgorithm(
                ref_dirs=ref_dirs, xl=xl, xu=xu,
                prob_crossover=self.config.prob_crossover,
                eta_crossover=self.config.eta_crossover,
                eta_mutation=self.config.eta_mutation,
                alpha=self.config.alpha,
            )

            self._log(
                "RVEA: %d目标 | %d变量 | %d个体 | %d参考向量 | "
                "SBX(p=%.2f,η=%.1f) + PM(η=%.1f) | α=%.1f | 精度=%s" % (
                    n_obj, n_var, n_ref, n_ref,
                    self.config.prob_crossover, self.config.eta_crossover,
                    self.config.eta_mutation, self.config.alpha,
                    self.fixed.precision))

            init_x = np.array(initial_p) if initial_p is not None else None
            X = algo.initialize(initial_x=init_x)
            F = self._evaluate_population(X)

            pf_idx = non_dominated_indices(F)
            pf_F   = -F[pf_idx] if len(pf_idx) > 0 else np.empty((0, n_obj))
            if self._on_initial:
                self._on_initial(pf_F)

            max_gen     = self.config.max_iter
            adapt_step  = max(1, int(max_gen * 0.1))  # f_r = 0.1

            for gen in range(1, max_gen + 1):
                if not self._is_running:
                    self._log("优化被用户停止，提取当前结果...")
                    break

                offspring_X = algo.create_offspring(X, F)
                offspring_F = self._evaluate_population(offspring_X)

                X, F = algo.select(X, F, offspring_X, offspring_F, gen, max_gen)

                if gen % adapt_step == 0 or gen == max_gen:
                    algo.adapt_reference_vectors(F)

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
                            "Generation %d/%d | 种群=%d | Pareto=%d | "
                            "最大各目标=%s" % (
                                gen, max_gen, len(X), len(pf_idx),
                                pf_F.max(axis=0).tolist()))

            return self._extract_result(F, X, time.time() - t0)

        except Exception as e:
            self._log("RVEA 出错: %s\n%s" % (e, traceback.format_exc()))
            return None
        finally:
            self._cleanup()

    def evaluate_initial(self, p_SWS) -> List[float]:
        p = np.array(p_SWS)
        return [evaluate_single_point(p, self.fixed, fp, self.coeffs)
                for fp in self.freq_params]

    def _evaluate_population(self, X: np.ndarray) -> np.ndarray:
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

    def _extract_result(self, F: np.ndarray, X: np.ndarray,
                        elapsed: float) -> OptimizationResult:
        pf_idx = non_dominated_indices(F)
        if len(pf_idx) == 0:
            return OptimizationResult(
                pareto_pop=np.empty((0, len(self.bounds))),
                pareto_F=np.empty((0, len(self.freq_params))),
                best_idx=-1, elapsed_sec=elapsed)

        pareto_F   = -F[pf_idx]
        pareto_pop = X[pf_idx]
        best_idx   = int(pareto_F.sum(axis=1).argmax())
        return OptimizationResult(
            pareto_pop=pareto_pop, pareto_F=pareto_F,
            best_idx=best_idx, elapsed_sec=elapsed)

    def _cleanup(self):
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None