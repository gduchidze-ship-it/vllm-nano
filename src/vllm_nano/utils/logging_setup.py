"""Central logging setup: Loguru core, Rich console, JSON file, stdlib bridge.

Library modules just use `from loguru import logger`. The application entry
point calls `setup_logging()` exactly once; importing this module has no side
effects.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.logging import RichHandler

if TYPE_CHECKING:
    from loguru import Record

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SECRETS = re.compile(
    r"(Bearer\s+)[A-Za-z0-9\-_.]{20,}"  # Authorization headers
    r"|\bhf_[A-Za-z0-9]{20,}"  # Hugging Face tokens
)

LOG_FILE_NAME = "vllm_nano.jsonl"


def redact(text: str) -> str:
    """Mask bearer tokens and Hugging Face tokens in `text`."""
    return _SECRETS.sub(lambda m: (m.group(1) or "") + "<redacted>", text)


@contextmanager
def request_context(rid: str) -> Iterator[None]:
    """Tag every log line emitted inside the block with `rid`, then restore."""
    token = request_id.set(rid)
    try:
        yield
    finally:
        request_id.reset(token)


def _patch(record: Record) -> None:
    """Attach the request id and scrub secrets before any sink sees the record."""
    record["extra"]["request_id"] = request_id.get()
    record["message"] = redact(record["message"])
    for key, value in list(record["extra"].items()):
        if isinstance(value, str):
            record["extra"][key] = redact(value)


def _console_format(record: Record) -> str:
    return "{extra[request_id]} | {message}"


class _InterceptHandler(logging.Handler):
    """Route stdlib `logging` records (transformers, torch, ...) into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _route_third_party_logs() -> None:
    """Make transformers log through the root logger instead of its own handler."""
    from transformers.utils import logging as hf_logging

    hf_logging.disable_default_handler()
    hf_logging.enable_propagation()


def setup_logging(
    level: str = "INFO",
    log_dir: Path | str | None = "logs",
    console: bool = True,
) -> None:
    """Configure logging for the process. Safe to call more than once.

    Args:
        level: Minimum level for all sinks (e.g. "DEBUG", "INFO").
        log_dir: Directory for the rotating JSON-lines file, or None to disable it.
        console: Whether to log to the terminal through Rich.
    """
    logger.remove()
    logger.configure(patcher=_patch)

    if console:
        logger.add(
            RichHandler(rich_tracebacks=True, markup=False),
            level=level,
            format=_console_format,
        )
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.add(
            path / LOG_FILE_NAME,
            level=level,
            serialize=True,
            rotation="1 day",
            retention="14 days",
            compression="zip",
            enqueue=True,
        )
    if not console and log_dir is None:
        logger.add(sys.stderr, level=level)

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    _route_third_party_logs()
