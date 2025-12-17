"""
tb_writer.py

Project-wide TensorBoard helper. Drop this into your thesis_multiagent/src/ and
import it from training scripts:

    from tb_writer import log_scalar, log_image, log_figure, log_text, flush, close

It creates a timestamped run directory under src/tb_logs by default.
"""
from __future__ import annotations

import os
import datetime
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    raise ImportError("torch and torch.utils.tensorboard are required for tb_writer.py") from e

# Root folder for tensorboard logs (folder is next to this file)
TB_LOG_ROOT = os.path.join(os.path.dirname(__file__), "tb_logs")
os.makedirs(TB_LOG_ROOT, exist_ok=True)

# Unique run folder: 2025-11-24_12-34-56 style
_run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TB_LOG_DIR = os.path.join(TB_LOG_ROOT, _run_name)
os.makedirs(TB_LOG_DIR, exist_ok=True)

# Create a SummaryWriter once at import time
writer = SummaryWriter(log_dir=TB_LOG_DIR)

# ---- Convenience API ----------------------------------------------------

def log_scalar(name: str, value: float, step: int) -> None:
    """Log a scalar metric."""
    writer.add_scalar(name, value, step)


def log_image(name: str, img: Any, step: int) -> None:
    """Log an image.

    img must be a CHW tensor or HxWxC numpy array. SummaryWriter supports both.
    """
    writer.add_image(name, img, step, dataformats="HWC")


def log_figure(name: str, fig, step: int) -> None:
    """Log a matplotlib figure object."""
    writer.add_figure(name, fig, step)


def log_text(name: str, text: str, step: int) -> None:
    """Log arbitrary text."""
    writer.add_text(name, text, step)


def flush() -> None:
    """Flush pending events to disk."""
    writer.flush()


def close() -> None:
    """Close the writer (call at process exit)."""
    writer.close()


# Optional helper to expose the active TB directory
def get_logdir() -> str:
    """Return the path to the active TensorBoard log directory for this run."""
    return TB_LOG_DIR


# If module is run directly, print where logs go.
if __name__ == "__main__":
    print("TensorBoard logs directory:", TB_LOG_DIR)

