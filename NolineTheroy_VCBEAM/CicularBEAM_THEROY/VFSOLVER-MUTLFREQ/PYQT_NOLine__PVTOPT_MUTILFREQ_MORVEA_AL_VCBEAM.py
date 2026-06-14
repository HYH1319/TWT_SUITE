# -*- coding: utf-8 -*-
"""
行波管多频点 RVEA 多目标进化优化工具 (单文件完整版)
依赖: pip install numpy PyQt5 pymoo
"""
import sys, os, json, logging, traceback, time
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

from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER_MIX_WITH_PVT import (
    calculate_SEGMENT_TWT_NOLINE,
)

logger = logging.getLogger(__name__)
EVAL_PENALTY = 1e4
N_FREQ = 3
MAX_WORKERS = max(1, cpu_count())

_APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) \
    else os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_APP_DIR, "configUDF", "OPT", "twt_RVEA_config_MUTIL.json")

# 标签常量
_LBL_F1KC = "频点1 Kc值 (Ω):"
_LBL_F1VPC = "频点1 Vpc值:"
_LBL_F1F = "频点1 频率" + "(GHz):"
_LBL_F2KC = "频点2 Kc值 (Ω):"
_LBL_F2VPC = "频点2 Vpc值:"
_LBL_F2F = "频点2 频率" + "(GHz):"
_LBL_F3KC = "频点3 Kc值 (Ω):"
_LBL_F3VPC = "频点3 Vpc值:"
_LBL_F3F = "频点3 频率" + "(GHz):"
_LBL_DB = "衰减量" + "(dB):"
_FILE_FLT = "文本文件" + " (*.txt)" + ";;" + "所有文件" + " (*)"


# ═══════════════════════════════════════════════════════════════
#  以下与 CORE 版完全一致
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  GUI 部分
# ═══════════════════════════════════════════════════════════════

class RVEATask(QThread):
    finished = pyqtSignal(object)
    prog_sig = pyqtSignal(int, object)
    iter_sig = pyqtSignal(int, object)
    init_sig = pyqtSignal(object)
    log_sig = pyqtSignal(str)

    def __init__(self, opt, ip=None):
        super().__init__()
        self.opt = opt
        self.ip = ip

    def run(self):
        self.opt._on_progress = self.prog_sig.emit
        self.opt._on_iteration = self.iter_sig.emit
        self.opt._on_initial = self.init_sig.emit
        self.opt._on_log = self.log_sig.emit
        try:
            result = self.opt.run(initial_p=self.ip)
            self.finished.emit(result)
        except Exception as e:
            self.log_sig.emit("异常: %s" % e)
            self.finished.emit(None)

    def stop(self):
        self.opt.stop()


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
            label = QLabel(key)
            widget = None
            if wtype == "double":
                widget = QDoubleSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
                if len(args) > 3:
                    widget.setSingleStep(args[3])
                if len(args) > 4:
                    widget.setDecimals(args[4])
            elif wtype == "int":
                widget = QSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
            elif wtype == "text":
                widget = QLineEdit()
                if args:
                    widget.setText(args[0])
            if widget is not None:
                layout.addRow(label, widget)
                self.widgets[key] = widget
        self.setLayout(layout)

    def get(self, key):
        w = self.widgets.get(key)
        if w is None:
            return None
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            return w.value()
        if isinstance(w, QLineEdit):
            return w.text()
        return None

    def put(self, key, val):
        w = self.widgets.get(key)
        if w is None:
            return
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            try:
                w.setValue(float(val))
            except Exception:
                pass
        elif isinstance(w, QLineEdit):
            w.setText(str(val))

    def collect(self):
        return {k: self.get(k) for k in self.widgets}

    def apply(self, data):
        if not isinstance(data, dict):
            return
        for k, v in data.items():
            self.put(k, v)


class FixedParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("电子枪参数", "label"),
            ("电流 I (A):", "double", 0, 3e3, 0.3, 0.001, 4),
            ("电压 V (V):", "double", 0, 3e6, 23000, 1, 0),
            ("慢波结构参数", "label"),
            ("每单元损耗参数:", "double", 0, 3e3, 0.0, 0.001, 4),
            ("衰减段索引 (逗号分隔):", "text", "1"),
            ("各段周期数 (逗号分隔):", "text", "20,5,10,10,10,10"),
            ("束流参数", "label"),
            ("束流宽度 w (mm):", "double", 0, 3e3, 0.2, 0.001, 4),
            ("束流厚度 t (mm):", "double", 0, 3e3, 0.2, 0.001, 4),
            ("填充参数", "label"),
            ("填充率倒数 Fn_K:", "double", 1, 3e3, 1.0, 0.001, 4),
            ("输入/衰减参数", "label"),
            ("输入功率 (W):", "double", 0, 3e3, 0.1, 0.001, 4),
            (_LBL_DB, "double", 0, 3e3, 20, 1, 1),
            ("参数调整系数", "label"),
            ("Vpc 调整系数:", "double", 0, 3e3, 0.82, 0.001, 4),
            ("Kc 调整系数:", "double", 0, 3e3, 1.6, 0.001, 4),
            ("频点1参数", "label"),
            (_LBL_F1KC, "double", 0, 3e3, 2.67, 0.001, 4),
            (_LBL_F1VPC, "double", 0, 1, 0.2867, 0.001, 4),
            (_LBL_F1F, "double", 0, 3e3, 220.0, 1, 1),
            ("频点2参数", "label"),
            (_LBL_F2KC, "double", 0, 3e3, 2.67, 0.001, 4),
            (_LBL_F2VPC, "double", 0, 1, 0.2867, 0.001, 4),
            (_LBL_F2F, "double", 0, 3e3, 225.0, 1, 1),
            ("频点3参数", "label"),
            (_LBL_F3KC, "double", 0, 3e3, 2.67, 0.001, 4),
            (_LBL_F3VPC, "double", 0, 1, 0.2867, 0.001, 4),
            (_LBL_F3F, "double", 0, 3e3, 230.0, 1, 1),
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


class RVEAConfigEditor(BaseEditor):
    def __init__(self, parent=None):
        # ★★★ 默认值与 CORE standalone __main__ 完全一致 ★★★
        fields = [
            ("分区数 p:", "int", 2, 100, 5),           # CORE: 5
            ("种群大小(0=自动):", "int", 0, 10000, 0),  # CORE: 0
            ("最大迭代:", "int", 10, 5000, 20),          # CORE: 20
            ("交叉概率:", "double", 0.0, 1.0, 0.9, 0.01, 2),
            ("交叉eta:", "double", 1, 100, 15, 1, 0),
            ("变异eta:", "double", 1, 100, 20, 1, 0),
            ("参数下限:", "text", "0.499,0.490,0.49,0.49,0.49,0.49"),
            ("参数上限:", "text", "0.501,0.510,0.51,0.51,0.51,0.51"),
            ("并行数:", "int", 1, 128, min(32, MAX_WORKERS)),  # CORE: 32
        ]
        super().__init__(fields, parent)

    def get_config(self, n_dim):
        try:
            lb_s = self.get("参数下限:")
            ub_s = self.get("参数上限:")
            lb = [float(v.strip()) for v in lb_s.split(",") if v.strip()]
            ub = [float(v.strip()) for v in ub_s.split(",") if v.strip()]
            if len(lb) == 1 and n_dim > 1:
                lb = lb * n_dim
            if len(ub) == 1 and n_dim > 1:
                ub = ub * n_dim
            if len(lb) != n_dim or len(ub) != n_dim:
                raise ValueError("边界维度不匹配")
            return RVEAConfig(
                n_partitions=self.get("分区数 p:"),
                pop_size=self.get("种群大小(0=自动):"),
                max_iter=self.get("最大迭代:"),
                prob_crossover=self.get("交叉概率:"),
                eta_crossover=self.get("交叉eta:"),
                eta_mutation=self.get("变异eta:"),
                workers=min(self.get("并行数:"), MAX_WORKERS),
            ), list(zip(lb, ub))
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None, None


class InitEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("初始p_SWS:", "text", "0.50,0.50,0.50,0.50,0.50,0.50"),
        ]
        super().__init__(fields, parent)

    def get_p(self, n_dim):
        try:
            txt = self.get("初始p_SWS:")
            vals = [float(v.strip()) for v in txt.split(",") if v.strip()]
            if len(vals) == 1 and n_dim > 1:
                vals = vals * n_dim
            if len(vals) != n_dim:
                raise ValueError("维度不匹配")
            return vals
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


class ParetoPanel(QWidget):
    solution_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pop = None
        self._fval = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.info_lbl = QLabel("等待优化结果...")
        layout.addWidget(self.info_lbl)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setStyleSheet(
            "QTableWidget{font-family:Consolas,monospace;font-size:10pt;}"
            "QHeaderView::section{background:#e8e8e8;font-weight:bold;padding:4px;}")
        self.table.selectionModel().selectionChanged.connect(self._on_sel)
        layout.addWidget(self.table, 1)

    def update_result(self, result):
        self._pop = result.pareto_pop
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行波管 RVEA 多目标优化工具")
        self.setGeometry(100, 100, 1300, 850)
        self.task = None
        self.result = None
        self.t0 = None
        self._init_ui()
        self._load_cfg()
        self.statusBar().showMessage("就绪 | CPU:%d核" % cpu_count())

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
        self.btn_run = self._make_btn("开始优化", "#4CAF50", self._start)
        self.btn_stop = self._make_btn("停止", "#f44336", self._stop)
        self.btn_stop.setEnabled(False)
        self.btn_save = self._make_btn("保存结果", "#2196F3", self._save)
        self.btn_save.setEnabled(False)
        self.btn_use = self._make_btn("使用选中方案", "#2196F3", self._use)
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
            ("ps", "Pareto大小", "#e0e0e0"),
            ("mt", "最大总功率(W)", "#c8e6c9"),
            ("rv", "参考向量数", "#e3f2fd"),
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
            p = self.edit_rc.get("分区数 p:")
            nr = RVEAConfig.ref_dir_count(p, N_FREQ)
            self.ref_lbl.setText(
                "Das-Dennis(p=%d) -> %d参考向量 | 种群>=%d" % (p, nr, nr))
            if "rv" in self._status:
                self._status["rv"].setText(str(nr))
        except Exception:
            pass

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append("[%s] %s" % (ts, msg))
        self.log_box.ensureCursorVisible()

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
        self._log("推荐方案%d: 总功率%.3fW" % (result.best_idx, result.best_total_power))
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
                nd = r.pareto_pop.shape[1]
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


if __name__ == "__main__":
    mp.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
