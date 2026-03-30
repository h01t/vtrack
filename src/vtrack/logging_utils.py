"""Logging helpers for consistent runtime diagnostics."""

from __future__ import annotations

import json
import logging
import os
from typing import TypedDict


class LogContext(TypedDict, total=False):
    command: str
    source: str
    source_kind: str
    model: str
    tracker: str
    device: str
    frame: int
    elapsed_ms: float
    event: str
    error_type: str


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "vtrack") -> logging.Logger:
    """Return a configured logger honoring VTRACK_LOG_LEVEL/FORMAT."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.getenv("VTRACK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv("VTRACK_LOG_FORMAT", "text").lower()

    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def build_log_context(**kwargs: object) -> LogContext:
    """Build a compact context object without null values."""
    return {k: v for k, v in kwargs.items() if v is not None}  # type: ignore[return-value]


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    context: LogContext | None = None,
) -> None:
    """Emit a structured log entry."""
    logger.log(level, message, extra={"context": context or {}})
