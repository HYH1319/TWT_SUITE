# ═══════════════════════════════════════════════════════════════════
#  文件 1: twt_rvea_optimizer.py  (核心优化引擎，零 GUI 依赖)
# ═══════════════════════════════════════════════════════════════════
"""
行波管多频点 RVEA 多目标优化引擎
依赖: pip install numpy pymoo
"""

import logging
import traceback
import numpy as np
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from multiprocessing import Pool
import multiprocessing as mp
import sys

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from Noline_GAIN_MAINCALL_VUCORE_MIX_WITH_PVT import (
    calculate_SEGMENT_TWT_NOLINE,
)

logger = logging.getLogger(__name__)
EVAL_PENALTY = 1e4


# ── 数据类 ──────────────────────────────────────────────────────


@dataclass
class FreqParam:
    Kc: float
    Vpc: float
    freq: float


@dataclass
class FixedParams:
    i: float
    v: float
    loss_perunit: float
    section_seg_idx: List[int]
    n_unit: List[int]
    w: float
    t: float
    Fn_K: float
    p_in: float
    loss_attu: float
    vpc_coeff: float
    kc_coeff: float

    @property
    def n_dim(self) -> int:
        return len(self.n_unit)


@dataclass
class Coeffs:
    vpc: float
    kc: float


@dataclass
class RVEAConfig:
    n_partitions: int = 12
    pop_size: int = 0
    max_iter: int = 200
    prob_crossover: float = 0.9
    eta_crossover: float = 15.0
    eta_mutation: float = 20.0
    workers: int = 1

    @staticmethod
    def ref_dir_count(p: int, m: int) -> int:
        from math import comb

        return comb(p + m - 1, m - 1)

    def n_ref_dirs_for(self, n_obj: int) -> int:
        return self.ref_dir_count(self.n_partitions, n_obj)


@dataclass
class OptimizationResult:
    pareto_pop: np.ndarray
    pareto_F: np.ndarray
    freq_labels: List[str]
    best_idx: int
    elapsed_sec: float = 0.0

    @property
    def n_solutions(self) -> int:
        return len(self.pareto_F)

    @property
    def best_p_SWS(self) -> np.ndarray:
        return self.pareto_pop[self.best_idx]

    @property
    def best_total_power(self) -> float:
        return float(self.pareto_F[self.best_idx].sum())


ProgressCallback = Callable[[int, np.ndarray], None]
IterationCallback = Callable[[int, np.ndarray], None]
InitialCallback = Callable[[np.ndarray], None]
LogCallback = Callable[[str], None]


# ── 工具函数 ────────────────────────────────────────────────────


def adjust_params(p_SWS, idx, Vpc, Kc, coeffs: Coeffs):
    delta = (p_SWS[idx] - p_SWS[0]) / p_SWS[0]
    return {
        "Vpc": Vpc + coeffs.vpc * delta * Vpc,
        "Kc": Kc + coeffs.kc * delta * Kc,
    }


def _eval_single(args):
    """进程池最小任务：评估 (个体, 频点) 的负输出功率"""
    p_SWS, fixed, fp, coeffs, ind_idx, freq_idx = args
    try:
        para_func = partial(adjust_params, coeffs=coeffs)
        result = calculate_SEGMENT_TWT_NOLINE(
            I=fixed.i,
            V=fixed.v,
            Kc=fp.Kc,
            Loss_perunit=fixed.loss_perunit,
            SectionedSEGMENT_IDX=fixed.section_seg_idx,
            p_SWS=p_SWS,
            N_unit=fixed.n_unit,
            w=fixed.w,
            t=fixed.t,
            Fn_K=fixed.Fn_K,
            f0_GHz=fp.freq,
            Vpc=fp.Vpc,
            para_func=para_func,
            P_in=fixed.p_in,
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
            I=fixed.i,
            V=fixed.v,
            Kc=fp.Kc,
            Loss_perunit=fixed.loss_perunit,
            SectionedSEGMENT_IDX=fixed.section_seg_idx,
            p_SWS=p_SWS,
            N_unit=fixed.n_unit,
            w=fixed.w,
            t=fixed.t,
            Fn_K=fixed.Fn_K,
            f0_GHz=fp.freq,
            Vpc=fp.Vpc,
            para_func=para_func,
            P_in=fixed.p_in,
            Loss_attu=fixed.loss_attu,
        )
        return result["输出功率P_out"]
    except Exception as e:
        logger.error("评估出错: %s", e)
        return float("nan")


# ── pymoo 问题（手动进程池批量评估）─────────────────────────────


class TWTProblem(Problem):
    """
    重写 _evaluate 批量处理整代种群，
    展平为 pop_size × n_freq 个独立任务，一次 pool.map 全并行。
    无需 StarmapParallelization，兼容所有 pymoo 版本。
    """

    def __init__(self, fixed, freq_params, coeffs, bounds, pool, n_workers):
        n_var = len(bounds)
        n_obj = len(freq_params)
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=0, xl=xl, xu=xu)
        self.fixed = fixed
        self.freq_params = freq_params
        self.coeffs = coeffs
        self.pool = pool
        self.n_workers = max(1, n_workers)

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        n_freq = len(self.freq_params)

        flat_args = []
        for i in range(n_pop):
            for j in range(n_freq):
                flat_args.append(
                    (X[i], self.fixed, self.freq_params[j], self.coeffs, i, j)
                )

        chunksize = max(1, len(flat_args) // (4 * self.n_workers))
        raw = self.pool.map(_eval_single, flat_args, chunksize=chunksize)

        F = np.full((n_pop, n_freq), EVAL_PENALTY)
        for ind_idx, freq_idx, val in raw:
            F[ind_idx, freq_idx] = val
        out["F"] = F


# ── 自定义采样 ──────────────────────────────────────────────────


class InitialSampling(Sampling):
    def __init__(self, initial_x=None):
        super().__init__()
        self.initial_x = np.array(initial_x) if initial_x is not None else None

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.uniform(problem.xl, problem.xu, (n_samples, problem.n_var))
        if self.initial_x is not None and len(self.initial_x) == problem.n_var:
            X[0] = self.initial_x
        return X


# ── 停止异常 ────────────────────────────────────────────────────


class StopOptimization(Exception):
    pass


# ── pymoo Callback ──────────────────────────────────────────────


class _RVEACallback(Callback):
    def __init__(
        self,
        is_running_fn,
        max_iter,
        nds,
        on_progress=None,
        on_iteration=None,
        on_initial=None,
        on_log=None,
    ):
        super().__init__()
        self._is_running = is_running_fn
        self._max_iter = max_iter
        self._nds = nds
        self._on_progress = on_progress
        self._on_iteration = on_iteration
        self._on_initial = on_initial
        self._on_log = on_log
        self._initial_fired = False
        self.last_algorithm = None

    def notify(self, algorithm):
        self.last_algorithm = algorithm

        if not self._is_running():
            raise StopOptimization()

        gen = algorithm.n_gen
        F = algorithm.pop.get("F")

        fronts = self._nds.do(F)
        pf_idx = fronts[0] if len(fronts) > 0 else np.array([], dtype=int)
        pf_idx = np.asarray(pf_idx, dtype=int)

        if len(pf_idx) > 0:
            pf_F = -F[pf_idx]
        else:
            pf_F = np.empty((0, F.shape[1]))

        progress = int(gen / self._max_iter * 100)

        if not self._initial_fired and self._on_initial:
            self._on_initial(pf_F)
            self._initial_fired = True

        if self._on_progress:
            self._on_progress(progress, pf_F)
        if self._on_iteration:
            self._on_iteration(gen, pf_F)

        if gen % 5 == 0 or gen == self._max_iter:
            if self._on_log and len(pf_F) > 0:
                self._on_log(
                    "Generation %d/%d | Pareto前沿=%d | "
                    "最大各频点功率=%s"
                    % (gen, self._max_iter, len(pf_idx), pf_F.max(axis=0).tolist())
                )


# ═══════════════════════════════════════════════════════════════════
#  核心优化器
# ═══════════════════════════════════════════════════════════════════


class RVEAOptimizer:
    """
    行波管多频点 RVEA 多目标优化器

    用法:
        opt = RVEAOptimizer(fixed, freqs, coeffs, bounds, cfg)
        result = opt.run(initial_p=[0.50]*6)
    """

    def __init__(
        self,
        fixed: FixedParams,
        freq_params: List[FreqParam],
        coeffs: Coeffs,
        bounds: List[Tuple[float, float]],
        config: RVEAConfig,
        on_progress: Optional[ProgressCallback] = None,
        on_iteration: Optional[IterationCallback] = None,
        on_initial: Optional[InitialCallback] = None,
        on_log: Optional[LogCallback] = None,
    ):
        self.fixed = fixed
        self.freq_params = freq_params
        self.coeffs = coeffs
        self.bounds = bounds
        self.config = config
        self._on_progress = on_progress
        self._on_iteration = on_iteration
        self._on_initial = on_initial
        self._on_log = on_log
        self._is_running = False
        self._pool = None

    def is_running(self) -> bool:
        return self._is_running

    def stop(self):
        self._is_running = False

    def run(self, initial_p=None):
        """执行 RVEA 优化（阻塞），返回 OptimizationResult 或 None"""
        import time as _time

        t0 = _time.time()
        if hasattr(sys, "frozen"):
            mp.freeze_support()
            mp.set_executable(sys.executable)

        nds = NonDominatedSorting()

        callback = _RVEACallback(
            is_running_fn=self.is_running,
            max_iter=self.config.max_iter,
            nds=nds,
            on_progress=self._on_progress,
            on_iteration=self._on_iteration,
            on_initial=self._on_initial,
            on_log=self._on_log,
        )

        try:
            self._is_running = True

            workers = max(1, self.config.workers)
            self._log("初始化 RVEA 优化 (%d 进程)..." % workers)
            self._pool = Pool(processes=workers)

            problem = TWTProblem(
                fixed=self.fixed,
                freq_params=self.freq_params,
                coeffs=self.coeffs,
                bounds=self.bounds,
                pool=self._pool,
                n_workers=workers,
            )

            n_obj = len(self.freq_params)
            n_ref = self.config.n_ref_dirs_for(n_obj)
            ref_dirs = get_reference_directions(
                "das-dennis",
                n_obj,
                n_partitions=self.config.n_partitions,
            )

            pop_size = self.config.pop_size
            if pop_size <= 0:
                pop_size = n_ref
            elif pop_size < n_ref:
                pop_size = n_ref
                self._log("种群大小调整为 %d（匹配 %d 参考向量）" % (pop_size, n_ref))

            if initial_p is not None:
                sampling = InitialSampling(np.array(initial_p))
            else:
                sampling = FloatRandomSampling()

            algorithm = RVEA(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=sampling,
                crossover=SBX(
                    prob=self.config.prob_crossover,
                    eta=self.config.eta_crossover,
                ),
                mutation=PM(eta=self.config.eta_mutation),
                eliminate_duplicates=True,
            )

            termination = MaximumGenerationTermination(self.config.max_iter)

            self._log(
                "RVEA: %d目标 | %d变量 | %d个体 | %d参考向量 | "
                "SBX(p=%.2f,η=%.1f) + PM(η=%.1f)"
                % (
                    n_obj,
                    len(self.bounds),
                    pop_size,
                    n_ref,
                    self.config.prob_crossover,
                    self.config.eta_crossover,
                    self.config.eta_mutation,
                )
            )

            res = minimize(
                problem,
                algorithm,
                termination,
                callback=callback,
                seed=None,
                verbose=False,
            )

            return self._extract_result(res.F, res.X, nds, _time.time() - t0)

        except StopOptimization:
            self._log("优化被用户停止，提取当前最优...")
            return self._extract_partial(callback, nds, _time.time() - t0)

        except Exception as e:
            self._log("RVEA 出错: %s\n%s" % (e, traceback.format_exc()))
            return None

        finally:
            self._cleanup()

    def evaluate_initial(self, p_SWS):
        """评估初始参数各频点功率 (W)，返回 list"""
        p = np.array(p_SWS)
        return [
            evaluate_single_point(p, self.fixed, fp, self.coeffs)
            for fp in self.freq_params
        ]

    # ── 内部方法 ──────────────────────────────────────────────

    def _log(self, msg):
        if self._on_log:
            self._on_log(msg)
        else:
            logger.info(msg)

    def _make_freq_labels(self):
        return [
            "频点%d(%.1fGHz)" % (i + 1, fp.freq)
            for i, fp in enumerate(self.freq_params)
        ]

    def _extract_result(self, F, X, nds, elapsed):
        fronts = nds.do(F)
        if len(fronts) == 0 or len(fronts[0]) == 0:
            return OptimizationResult(
                pareto_pop=np.empty((0, len(self.bounds))),
                pareto_F=np.empty((0, len(self.freq_params))),
                freq_labels=self._make_freq_labels(),
                best_idx=-1,
                elapsed_sec=elapsed,
            )
        pf_idx = np.asarray(fronts[0], dtype=int)
        pareto_F = -F[pf_idx]
        pareto_pop = X[pf_idx]
        best_idx = int(pareto_F.sum(axis=1).argmax())
        return OptimizationResult(
            pareto_pop=pareto_pop,
            pareto_F=pareto_F,
            freq_labels=self._make_freq_labels(),
            best_idx=best_idx,
            elapsed_sec=elapsed,
        )

    def _extract_partial(self, callback, nds, elapsed):
        try:
            algo = callback.last_algorithm
            if algo is not None:
                F = algo.pop.get("F")
                X = algo.pop.get("X")
                return self._extract_result(F, X, nds, elapsed)
            return None
        except Exception as e:
            self._log("提取部分结果失败: %s" % e)
            return None

    def _cleanup(self):
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None


# ── 独立运行入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime

    fixed = FixedParams(
        i=0.3,
        v=23000,
        loss_perunit=0.0,
        section_seg_idx=[1],
        n_unit=[20, 5, 10, 10, 10, 10],
        w=0.5,
        t=0.1,
        Fn_K=1.0,
        p_in=0.1,
        loss_attu=20.0,
        vpc_coeff=0.82,
        kc_coeff=1.6,
    )
    freq_params = [
        FreqParam(Kc=2.67, Vpc=0.2867, freq=220.0),
        FreqParam(Kc=2.67, Vpc=0.2867, freq=225.0),
        FreqParam(Kc=2.67, Vpc=0.2867, freq=230.0),
    ]
    coeffs = Coeffs(vpc=0.82, kc=1.6)
    bounds = [
        (0.499, 0.501),
        (0.490, 0.510),
        (0.490, 0.510),
        (0.490, 0.510),
        (0.490, 0.510),
        (0.490, 0.510),
    ]
    config = RVEAConfig(
        n_partitions=5,
        pop_size=0,
        max_iter=20,
        prob_crossover=0.9,
        eta_crossover=15,
        eta_mutation=20,
        workers=32,
    )

    def _log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print("[%s] %s" % (ts, msg))

    opt = RVEAOptimizer(fixed, freq_params, coeffs, bounds, config, on_log=_log)
    init_p = [0.50] * fixed.n_dim
    init_powers = opt.evaluate_initial(init_p)
    _log("─── 初始参数各频点功率 ───")
    for j, fp in enumerate(freq_params):
        _log("  频点%d (%.1f GHz): %.3f W" % (j + 1, fp.freq, init_powers[j]))
    _log("  初始总功率: %.3f W" % sum(init_powers))

    result = opt.run(initial_p=init_p)
    if result and result.n_solutions > 0:
        _log("\n===== 优化完成 =====")
        _log("Pareto 前沿: %d 个方案" % result.n_solutions)
        _log("推荐最优解 (方案 %d):" % result.best_idx)
        for i, lbl in enumerate(result.freq_labels):
            _log("  %s: %.3f W" % (lbl, result.pareto_F[result.best_idx, i]))
        _log("  总功率: %.3f W" % result.best_total_power)
        _log("  p_SWS: %s" % result.best_p_SWS)
        _log("耗时: %.1fs" % result.elapsed_sec)
    else:
        _log("优化未产生有效结果")
