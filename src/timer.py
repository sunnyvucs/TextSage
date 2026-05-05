"""
timer.py
Context manager that logs the start time, end time, and duration of any pipeline step.
Usage:
    with StepTimer("Split PDF"):
        split_pdf(...)
"""

import logging
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

_SEP = "─" * 60


@contextmanager
def StepTimer(step_name: str, pdf_name: str = ""):
    label = f"[{pdf_name}] {step_name}" if pdf_name else step_name
    start = time.perf_counter()
    start_ts = time.strftime("%H:%M:%S")

    log.info("%s", _SEP)
    log.info("▶ START  %s  (at %s)", label, start_ts)

    try:
        yield
        elapsed = time.perf_counter() - start
        end_ts = time.strftime("%H:%M:%S")
        log.info("✔ END    %s  (at %s)  duration=%.2fs", label, end_ts, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        end_ts = time.strftime("%H:%M:%S")
        log.error("✘ FAILED %s  (at %s)  duration=%.2fs", label, end_ts, elapsed)
        log.error("  Error: %s", exc, exc_info=True)
        raise
