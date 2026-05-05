"""
logger.py
Configures logging for the entire pipeline.
- INFO and above → terminal (stdout)
- WARNING and above → logs/errors.log (file, rotated at 5 MB)
- Full DEBUG trace → logs/debug.log (file, rotated at 10 MB)
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(r"D:\AI\AI Projects\PDFParserAI\logs")
_initialized = False


def setup_logging(debug: bool = False) -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_terminal = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  [%(filename)s:%(lineno)d]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Terminal handler ──────────────────────────────────────────────────────
    import sys
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt_terminal)
    # Force flush after every record so output appears immediately
    ch.flush = lambda: sys.stdout.flush()
    root.addHandler(ch)

    # Disable buffering on stdout
    sys.stdout.reconfigure(line_buffering=True)

    # ── Error log file (warnings + errors only) ───────────────────────────────
    err_handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "errors.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(fmt_file)
    root.addHandler(err_handler)

    # ── Full debug log file ───────────────────────────────────────────────────
    dbg_handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "debug.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    dbg_handler.setLevel(logging.DEBUG)
    dbg_handler.setFormatter(fmt_file)
    root.addHandler(dbg_handler)
