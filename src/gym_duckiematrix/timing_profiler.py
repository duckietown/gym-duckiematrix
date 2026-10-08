"""Lightweight timing profiler."""

import logging
import time
from collections import defaultdict
from types import TracebackType

TABLE_SEPARATOR = "+-{0}-+-------------+-------------+-------------+\n"
TABLE_HEADER_FMT = "| {0} |   # Calls   |  Time (ms)  |  Freq (Hz)  |\n"
TABLE_ROW_FMT = "| {key} | {count} | {duration} | {frequency} |\n"


class TimingProfiler:
    """Track average durations for named code paths."""

    _buffer: dict[str, float]
    _count: defaultdict[str, int]
    _data: defaultdict[str, float]
    _enabled: bool
    _title: str

    class ProfilingContext:
        """Profiling context manager."""

        _key: str
        _profiler: "TimingProfiler"

        def __init__(self, profiler: "TimingProfiler", key: str) -> None:
            """Initialize the profiling context."""
            self._profiler = profiler
            self._key = key

        def __enter__(self) -> None:
            """Start timing when entering the context."""
            self._profiler.tick(self._key)

        def __exit__(
            self,
            _: type[BaseException] | None,
            __: BaseException | None,
            ___: TracebackType | None,
        ) -> None:
            """Stop timing when exiting the context."""
            self._profiler.tock(self._key)

    def __init__(self, title: str) -> None:
        """Initialize the profiler with a log title."""
        self._title = title
        self._enabled = False
        self._buffer = {}
        self._count = defaultdict(lambda: 0)
        self._data = defaultdict(lambda: 0.0)

    @property
    def enabled(self) -> bool:
        """Return whether profiling is enabled."""
        return self._enabled

    def enable(self, *, status: bool = True) -> None:
        """Enable or disable profiling."""
        self._enabled = status

    def profile(self, key: str) -> ProfilingContext:
        """Return a profiling context."""
        return self.ProfilingContext(self, key)

    def observe(self, key: str, duration_ms: float) -> None:
        """Record an externally measured duration."""
        if not self._enabled:
            return
        self._data[key] += duration_ms
        self._count[key] += 1

    def tick(self, key: str) -> None:
        """Start measuring *key*."""
        if not self._enabled:
            return
        self._buffer[key] = time.perf_counter() * 1000

    def tock(self, key: str) -> None:
        """Stop measuring *key*."""
        if not self._enabled:
            return
        tick = self._buffer.pop(key, None)
        if tick is None:
            return
        self.observe(key, time.perf_counter() * 1000 - tick)

    def log(self, logger: logging.Logger) -> None:
        """Log profiling information."""
        if not self._enabled:
            return
        if not self._count:
            logger.info("\n%s:\n(no samples)\n", self._title)
            return
        column_size = max(len(key) for key in self._count)
        table = ""
        table += TABLE_SEPARATOR.format("-" * column_size)
        table += TABLE_HEADER_FMT.format("Key".ljust(column_size, " "))
        table += TABLE_SEPARATOR.format("-" * column_size)
        for key in sorted(self._count.keys()):
            cumulative_duration = self._data[key]
            count = self._count[key]
            average_duration = cumulative_duration / float(count)
            average_frequency = (
                "inf"
                if average_duration <= 0
                else round(1000 / average_duration, 1)
            )
            table += TABLE_ROW_FMT.format(
                key=key.ljust(column_size, " "),
                count=str(count).rjust(11, " "),
                duration=str(round(average_duration, 2)).rjust(11, " "),
                frequency=str(average_frequency).rjust(11, " "),
            )
        table += TABLE_SEPARATOR.format("-" * column_size)
        logger.info("\n%s:\n%s\n", self._title, table)
