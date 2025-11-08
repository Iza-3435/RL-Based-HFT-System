"""Production logging configuration."""

import logging
import sys
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    QUIET = "QUIET"
    NORMAL = "NORMAL"
    VERBOSE = "VERBOSE"
    DEBUG = "DEBUG"


class ProductionLogger:
    """Production-grade structured logger."""

    _instance: Optional["ProductionLogger"] = None
    _level: LogLevel = LogLevel.NORMAL

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._setup_logging()

    def _setup_logging(self):
        """Configure logging system."""
        self.logger = logging.getLogger("hft")

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self._update_level()

    def set_level(self, level: LogLevel):
        """Set logging verbosity."""
        self._level = level
        self._update_level()

    def _update_level(self):
        """Update logger level based on verbosity."""
        level_map = {
            LogLevel.QUIET: logging.WARNING,  # Only warnings and errors
            LogLevel.NORMAL: logging.INFO,     # Info and above
            LogLevel.VERBOSE: logging.DEBUG,   # Debug and above
            LogLevel.DEBUG: logging.DEBUG,     # All logging
        }
        self.logger.setLevel(level_map[self._level])

        # Also set root logger to prevent other loggers from bypassing
        logging.getLogger().setLevel(level_map[self._level])

    def should_log_verbose(self) -> bool:
        """Check if verbose logging enabled."""
        return self._level in (LogLevel.VERBOSE, LogLevel.DEBUG)

    def should_log_debug(self) -> bool:
        """Check if debug logging enabled."""
        return self._level == LogLevel.DEBUG

    def info(self, msg: str, **kwargs):
        """Log info message."""
        # Suppress info logs in QUIET mode
        if self._level == LogLevel.QUIET:
            return
        if kwargs:
            msg = f"{msg} | {self._format_kwargs(kwargs)}"
        self.logger.info(msg)

    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        if kwargs:
            msg = f"{msg} | {self._format_kwargs(kwargs)}"
        self.logger.debug(msg)

    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        if kwargs:
            msg = f"{msg} | {self._format_kwargs(kwargs)}"
        self.logger.warning(msg)

    def error(self, msg: str, **kwargs):
        """Log error message."""
        if kwargs:
            msg = f"{msg} | {self._format_kwargs(kwargs)}"
        self.logger.error(msg)

    def verbose(self, msg: str, **kwargs):
        """Log verbose message (only in verbose/debug mode)."""
        if self.should_log_verbose():
            self.info(msg, **kwargs)

    @staticmethod
    def _format_kwargs(kwargs: dict) -> str:
        """Format kwargs for logging."""
        return " ".join(f"{k}={v}" for k, v in kwargs.items())


def get_logger() -> ProductionLogger:
    """Get singleton logger instance."""
    return ProductionLogger()
