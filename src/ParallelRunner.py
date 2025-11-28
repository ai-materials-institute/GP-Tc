#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ParallelRunner + ProgressBar (auto-progress by default)
======================================================

- Auto-selects ThreadPool in notebooks; prefers ProcessPool in scripts when safe.
- Shows a ProgressBar by default; pass silent=True for quiet mode.
- Accepts a user-provided ProgressBar (we won't call finish() on yours).

Author
------
Aaditya Panigrahi
"""

from __future__ import annotations
import os, sys, signal, pickle, inspect, time, random, math, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, Tuple

# --- import your ProgressBar (must provide: .total, .global_offset, .update(abs_done, force=True)) ---
sys.path.append("../src")
from ProgressBar import ProgressBar

# --- Keep BLAS tame in workers ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_USE_SIGALRM = hasattr(signal, "SIGALRM")

class _Timeout(Exception): ...
def _alarm_timeout(*_): raise _Timeout("per-item timeout")

def _normalize_timeout(t) -> int:
    if t is None: return 0
    try: t = float(t)
    except Exception: return 0
    return 0 if t <= 0 or t == float("inf") else int(t)

def _ensure_picklable(x) -> bool:
    try:
        pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:
        return False

def _in_interactive_main() -> bool:
    main = sys.modules.get("__main__")
    return not hasattr(main, "__file__")

def _is_bound_method(fn: Any) -> bool:
    return inspect.ismethod(fn) and (fn.__self__ is not None)

# ---------------- Worker entry points (must be top-level & importable) ----------------
# NOTE: `use_alarm` ensures we never touch signals in threads (only in processes).

def _worker_call_func(idx_item: Tuple[int, Any],
                      func: Callable,
                      broadcast_kwargs: dict,
                      per_item_timeout_secs: Optional[float],
                      use_alarm: bool) -> Tuple[int, Any]:
    idx, item = idx_item
    t = _normalize_timeout(per_item_timeout_secs)
    if use_alarm and _USE_SIGALRM and t > 0:
        signal.signal(signal.SIGALRM, _alarm_timeout)
        signal.alarm(t)
    try:
        out = func(item, **(broadcast_kwargs or {}))
        if use_alarm and _USE_SIGALRM and t > 0:
            signal.alarm(0)
        return idx, out
    except Exception:
        if use_alarm and _USE_SIGALRM and t > 0:
            signal.alarm(0)
        raise

def _worker_call_bound(idx_item: Tuple[int, Any],
                       instance: Any,
                       method_name: str,
                       broadcast_kwargs: dict,
                       per_item_timeout_secs: Optional[float],
                       use_alarm: bool) -> Tuple[int, Any]:
    idx, item = idx_item
    t = _normalize_timeout(per_item_timeout_secs)
    if use_alarm and _USE_SIGALRM and t > 0:
        signal.signal(signal.SIGALRM, _alarm_timeout)
        signal.alarm(t)
    try:
        method = getattr(instance, method_name)
        out = method(item, **(broadcast_kwargs or {}))
        if use_alarm and _USE_SIGALRM and t > 0:
            signal.alarm(0)
        return idx, out
    except Exception:
        if use_alarm and _USE_SIGALRM and t > 0:
            signal.alarm(0)
        raise

# ----------------------------------- ParallelRunner -----------------------------------

class ParallelRunner:
    """
    Minimal parallel map with progress reporting.

    By default, shows a ProgressBar. Pass `silent=True` for a quiet run.
    If you pass your own `ProgressBar` via `progress=...`, we use it and do NOT call finish().

    Parameters
    ----------
    max_workers : int, optional
        Max number of workers. Defaults to SLURM_CPUS_ON_NODE or os.cpu_count() or 1.
    executor : {'auto','process','thread'}, default 'auto'
        Backend selection.
    per_item_timeout_secs : float or None, optional
        POSIX-only per-item timeout (SIGALRM). None/<=0 disables.
    progress_refresh_sec : float, default 0.1
        Minimum seconds between progress redraws.
    """

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        executor: str = "auto",
        per_item_timeout_secs: Optional[float] = None,
        progress_refresh_sec: float = 0.1,
    ):
        self.max_workers = max_workers
        self.executor = executor
        self.per_item_timeout_secs = per_item_timeout_secs
        self.progress_refresh_sec = max(0.02, float(progress_refresh_sec))

    def run(
        self,
        func: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        broadcast_kwargs: Optional[dict] = None,
        progress: Optional[ProgressBar] = None,
        silent: bool = False,
    ) -> List[Any]:
        """
        Apply a callable to each item, in parallel, with optional progress updates.

        Parameters
        ----------
        func : callable
            Function or bound method: func(item, **broadcast_kwargs).
        items : iterable
            Items to process (converted to list).
        broadcast_kwargs : dict, optional
            Extra kwargs applied to every call.
        progress : ProgressBar, optional
            If provided, we'll use it and NOT call finish().
        silent : bool, default False
            If True, no progress bar is shown (overrides auto-creation).

        Returns
        -------
        list
            Results aligned to input order.
        """
        items = list(items)
        if not items:
            return []

        workers = (
            self.max_workers
            or int(os.environ.get("SLURM_CPUS_ON_NODE", "0") or 0)
            or (os.cpu_count() or 1)
        )

        # Strategy
        call_kind, call_payload, use_threads = self._decide_call_strategy(func, broadcast_kwargs)

        # Ensure spawn (Linux safety)
        if not use_threads:
            try:
                mp.set_start_method("spawn", force=False)
            except RuntimeError:
                pass

        results = [None] * len(items)
        idx_items = list(enumerate(items))

        # --- Progress bar: auto-create by default unless silent=True or user provided one ---
        auto_progress = False
        if progress is None and not silent:
            progress = ProgressBar(total=len(items), global_offset=0, history_elapsed=0.0)
            auto_progress = True

        # Progress setup
        completed = 0
        last_draw = 0.0
        global_offset = getattr(progress, "global_offset", 0) if progress else 0

        if progress:
            try:
                want_total = global_offset + len(items)
                if hasattr(progress, "total") and getattr(progress, "total") != want_total:
                    progress.total = want_total
                progress.update(global_offset, force=True)
                last_draw = time.time()
            except Exception:
                pass

        def _maybe_redraw():
            nonlocal last_draw
            if not progress:
                return
            now = time.time()
            if (now - last_draw) >= self.progress_refresh_sec or completed == len(items):
                try:
                    progress.update(global_offset + completed, force=True)
                except Exception:
                    pass
                last_draw = now

        # Submit jobs
        executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        executor_args = {"max_workers": workers}
        if not use_threads:
            executor_args["mp_context"] = mp.get_context("spawn")
        use_alarm = (not use_threads)  # only processes may use SIGALRM

        with executor_cls(**executor_args) as ex:
            if call_kind == "func":
                futs = [
                    ex.submit(_worker_call_func, p, call_payload["func"],
                              broadcast_kwargs or {}, self.per_item_timeout_secs, use_alarm)
                    for p in idx_items
                ]
            else:
                futs = [
                    ex.submit(_worker_call_bound, p, call_payload["instance"], call_payload["method_name"],
                              broadcast_kwargs or {}, self.per_item_timeout_secs, use_alarm)
                    for p in idx_items
                ]

            for fut in as_completed(futs):
                idx, res = fut.result()
                results[idx] = res
                completed += 1
                _maybe_redraw()

        if progress:
            try:
                progress.update(global_offset + completed, force=True)
            except Exception:
                pass
            # If we created the bar automatically, we own finishing it.
            if auto_progress:
                try:
                    progress.finish()
                except Exception:
                    pass

        return results

    # ---------- call-strategy helper ----------
    def _decide_call_strategy(self, func, broadcast_kwargs):
        if self.executor == "thread":
            if _is_bound_method(func):
                return ("bound", {"instance": func.__self__, "method_name": func.__name__}, True)
            return ("func", {"func": func}, True)

        if self.executor == "process":
            if _is_bound_method(func):
                inst = func.__self__
                if _ensure_picklable(inst):
                    return ("bound", {"instance": inst, "method_name": func.__name__}, False)
                return ("bound", {"instance": inst, "method_name": func.__name__}, True)
            return ("func", {"func": func},
                    not _ensure_picklable(func) or not _ensure_picklable(broadcast_kwargs or {}))

        # Auto mode
        if _in_interactive_main():
            # Notebooks prefer threads
            if _is_bound_method(func):
                return ("bound", {"instance": func.__self__, "method_name": func.__name__}, True)
            return ("func", {"func": func}, True)

        # Scripts prefer processes when safe
        if _is_bound_method(func):
            inst = func.__self__
            if _ensure_picklable(inst) and _ensure_picklable(broadcast_kwargs or {}):
                return ("bound", {"instance": inst, "method_name": func.__name__}, False)
            return ("bound", {"instance": inst, "method_name": func.__name__}, True)
        if _ensure_picklable(func) and _ensure_picklable(broadcast_kwargs or {}):
            return ("func", {"func": func}, False)
        return ("func", {"func": func}, True)

'''
# ===============================
# DEMO 1 — default (auto progress) plain function
# ===============================
def f(x, offset=0):
    time.sleep(random.uniform(0.1, 0.2))
    return math.sqrt(x + offset)

if __name__ == "__main__":
    items = list(range(30))
    runner = ParallelRunner(executor="auto", per_item_timeout_secs=None)
    # No progress passed → auto ProgressBar is created and finished for you
    results1 = runner.run(f, items, broadcast_kwargs={"offset": 1})
    print("Demo 1 (auto progress): first 10 results:", results1[:10])

    # ===============================
    # DEMO 2 — instance method with your own ProgressBar (we won't finish it)
    # ===============================
    class Worker:
        def __init__(self, offset=2):
            self.offset = offset
        def compute(self, x):
            time.sleep(random.uniform(0.1, 0.2))
            return math.sqrt(x + self.offset)

    w = Worker(offset=3)
    items2 = list(range(20))

    pb2 = ProgressBar(total=len(items2), global_offset=0, history_elapsed=0.0)
    pb2.update(0, force=True)
    results2 = runner.run(w.compute, items2, progress=pb2)  # uses your bar
    pb2.finish()
    print("Demo 2 (custom progress): first 10 results:", results2[:10])

    # ===============================
    # DEMO 3 — silent mode
    # ===============================
    results3 = runner.run(f, items2, broadcast_kwargs={"offset": 4}, silent=True)
    print("Demo 3 (silent): first 5 results:", results3[:5])
    '''