#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
progress_bar.py
---------------
Terminal progress bar with ETA and elapsed time.
No file I/O, no context manager — pure .step() / .finish() interface.

Features
--------
- Live in-terminal bar (overwrites one line)
- ETA includes prior progress (via history_elapsed + global_offset)
- Optional ASCII-only mode
- No external dependencies

Author
------
Aaditya Panigrahi
"""

from __future__ import annotations
import sys
import time
from typing import Optional


class ProgressBar:
    """
    Simple terminal progress bar with ETA and elapsed time.

    Parameters
    ----------
    total : int
        Total number of items in the process.
    width : int, optional
        Width of the progress bar (default: 40).
    global_offset : int, optional
        Items already completed before this run (default: 0).
    history_elapsed : float, optional
        Seconds already spent before this run (used in ETA).
    ascii_only : bool, optional
        Use '#' and '.' instead of Unicode blocks.
    stream : file-like, optional
        Output stream (default: sys.stdout).
    """

    def __init__(
        self,
        total: int,
        *,
        width: int = 40,
        global_offset: int = 0,
        history_elapsed: float = 0.0,
        ascii_only: bool = False,
        stream=None,
    ):
        if total <= 0:
            raise ValueError("total must be > 0")

        self.total = int(total)
        self.width = int(width)
        self.global_offset = int(global_offset)
        self.history_elapsed = float(history_elapsed)
        self.ascii_only = ascii_only
        self.stream = stream or sys.stdout

        self.start_time = time.time()
        self._processed_local = 0
        self._last_print_time = 0.0

    # ---------- Helpers ----------
    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 0 or seconds == float("inf"):
            seconds = 0.0
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"

    def _bar_chars(self):
        return ("#", ".") if self.ascii_only else ("█", "·")

    def _make_line(self, done: int) -> str:
        pct = int(100 * done / self.total)
        filled_len = int(self.width * pct / 100)
        full, empty = self._bar_chars()
        bar = full * filled_len + empty * (self.width - filled_len)

        # elapsed time (current + prior)
        elapsed_session = time.time() - self.start_time
        elapsed_total = self.history_elapsed + elapsed_session

        # work completed (current + prior)
        completed_total = max(1, self.global_offset + self._processed_local)

        # compute ETA from historical + current progress
        per_item = elapsed_total / completed_total
        remaining = per_item * max(0, self.total - done)

        return (
            f"[{bar}] {pct:3d}% | "
            f"Elapsed {self._format_time(elapsed_total)} | "
            f"ETA {self._format_time(remaining)}"
        )

    def _print_line(self, line: str):
        self.stream.write("\r" + line)
        self.stream.flush()

    # ---------- Public API ----------
    def update(self, global_done: int, force: bool = False):
        """Manually set absolute progress (0 ≤ global_done ≤ total)."""
        now = time.time()
        if not force and now - self._last_print_time < 0.1:
            return  # throttle updates
        self._last_print_time = now

        global_done = max(0, min(int(global_done), self.total))
        self._print_line(self._make_line(global_done))

    def step(self, n: int = 1):
        """Increment local progress by `n` items and refresh display."""
        self._processed_local += int(n)
        done = self.global_offset + self._processed_local
        self.update(done)

    def finish(self):
        """Mark progress complete."""
        self.update(self.total, force=True)
        self.stream.write("\n")
        self.stream.flush()

'''
# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import random
    pb2 = ProgressBar(total=total_items,
                      global_offset=already_done,
                      history_elapsed=prior_time,
                      ascii_only=True)
    pb2.update(already_done, force=True)
    for _ in range(150):
        time.sleep(random.uniform(0.01, 0.02))
        pb2.step()
    pb2.finish()
'''