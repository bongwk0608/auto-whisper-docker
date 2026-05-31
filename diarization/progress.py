from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_progress_enabled(value: str | None, verbose: bool = False) -> bool:
    if value is None or value == "":
        return verbose
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ProgressContext:
    file_index: int | None = None
    file_total: int | None = None


class DiarizationProgressReporter:
    def __init__(
        self,
        context: ProgressContext | None = None,
        min_percent_delta: float = 1.0,
        min_interval_seconds: float = 10.0,
        clock: Any = time.monotonic,
    ) -> None:
        self.context = context or ProgressContext()
        self.min_percent_delta = min_percent_delta
        self.min_interval_seconds = min_interval_seconds
        self.clock = clock
        self.started_at = float(self.clock())
        self.last_stage: str | None = None
        self.last_percent: float | None = None
        self.last_printed_at: float | None = None

    def update(
        self,
        stage: str,
        current: float | None = None,
        total: float | None = None,
        detail: str | None = None,
        force: bool = False,
    ) -> bool:
        now = float(self.clock())
        percent = self.percent(current, total)
        if not force and not self.should_print(stage, percent, now):
            return False

        elapsed = now - self.started_at
        eta = self.eta(elapsed, current, total)
        parts = [self.file_prefix(), stage]
        if percent is not None:
            parts.append(f"{percent:.1f}%")
        if detail:
            parts.append(detail)
        parts.append(f"elapsed={format_duration(elapsed)}")
        parts.append(f"eta={format_duration(eta)}")
        print(" ".join(part for part in parts if part), flush=True)

        self.last_stage = stage
        self.last_percent = percent
        self.last_printed_at = now
        return True

    def message(self, message: str, force: bool = True) -> bool:
        return self.update(message, force=force)

    def file_prefix(self) -> str:
        if self.context.file_index is None or self.context.file_total is None:
            return "[file ?/?]"
        return f"[file {self.context.file_index}/{self.context.file_total}]"

    def should_print(self, stage: str, percent: float | None, now: float) -> bool:
        if stage != self.last_stage:
            return True
        if self.last_printed_at is None:
            return True
        if now - self.last_printed_at >= self.min_interval_seconds:
            return True
        if percent is None or self.last_percent is None:
            return False
        return percent - self.last_percent >= self.min_percent_delta

    @staticmethod
    def percent(current: float | None, total: float | None) -> float | None:
        if current is None or total is None or total <= 0:
            return None
        return max(0.0, min(100.0, (current / total) * 100.0))

    @staticmethod
    def eta(elapsed: float, current: float | None, total: float | None) -> float | None:
        if current is None or total is None or current <= 0 or total <= 0:
            return None
        remaining = max(0.0, total - current)
        return remaining * (elapsed / current)


class ProjectProgressHook:
    def __init__(self, reporter: DiarizationProgressReporter, upstream_hook: Any | None = None) -> None:
        self.reporter = reporter
        self.upstream_hook = upstream_hook

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = None
        if self.upstream_hook is not None:
            result = self.upstream_hook(*args, **kwargs)

        stage = self.stage_from(args, kwargs)
        current = self.number_from(kwargs, "completed", "current", "done", "n", "value")
        total = self.number_from(kwargs, "total", "size", "maximum")
        if current is None or total is None:
            inferred_current, inferred_total = self.progress_from_args(args)
            current = current if current is not None else inferred_current
            total = total if total is not None else inferred_total
        self.reporter.update(f"pyannote {stage}", current=current, total=total)
        return result

    @staticmethod
    def stage_from(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        for key in ("step_name", "stage", "task", "name"):
            value = kwargs.get(key)
            if value:
                return str(value).replace("_", " ")
        for arg in args:
            if isinstance(arg, str) and arg:
                return arg.replace("_", " ")
        return "inference"

    @staticmethod
    def number_from(kwargs: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = kwargs.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def progress_from_args(args: tuple[Any, ...]) -> tuple[float | None, float | None]:
        numbers = [float(arg) for arg in args if isinstance(arg, (int, float))]
        if len(numbers) >= 2:
            return numbers[-2], numbers[-1]
        return None, None
