"""Tests for logging configuration."""

import pytest
import logging

from src.core.logging_config import ProductionLogger, LogLevel, get_logger


class TestProductionLogger:
    """Test production logger."""

    def test_singleton_pattern(self):
        """Test logger is singleton."""
        logger1 = get_logger()
        logger2 = get_logger()

        assert logger1 is logger2

    def test_log_level_setting(self):
        """Test setting log levels."""
        logger = get_logger()

        logger.set_level(LogLevel.QUIET)
        assert logger.logger.level == logging.ERROR

        logger.set_level(LogLevel.NORMAL)
        assert logger.logger.level == logging.INFO

        logger.set_level(LogLevel.VERBOSE)
        assert logger.logger.level == logging.INFO

        logger.set_level(LogLevel.DEBUG)
        assert logger.logger.level == logging.DEBUG

    def test_verbose_logging_control(self):
        """Test verbose logging control."""
        logger = get_logger()

        logger.set_level(LogLevel.QUIET)
        assert not logger.should_log_verbose()

        logger.set_level(LogLevel.NORMAL)
        assert not logger.should_log_verbose()

        logger.set_level(LogLevel.VERBOSE)
        assert logger.should_log_verbose()

        logger.set_level(LogLevel.DEBUG)
        assert logger.should_log_verbose()

    def test_kwargs_formatting(self):
        """Test kwargs formatting in log messages."""
        logger = get_logger()

        formatted = logger._format_kwargs({'a': 1, 'b': 'test', 'c': 3.14})
        assert 'a=1' in formatted
        assert 'b=test' in formatted
        assert 'c=3.14' in formatted

    def test_info_logging(self, caplog):
        """Test info level logging."""
        logger = get_logger()
        logger.set_level(LogLevel.NORMAL)

        with caplog.at_level(logging.INFO):
            logger.info("Test message", key="value")

        assert "Test message" in caplog.text
        assert "key=value" in caplog.text

    def test_debug_logging(self, caplog):
        """Test debug level logging."""
        logger = get_logger()
        logger.set_level(LogLevel.DEBUG)

        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message", test=123)

        assert "Debug message" in caplog.text
        assert "test=123" in caplog.text

    def test_verbose_suppression(self, caplog):
        """Test verbose messages suppressed in normal mode."""
        logger = get_logger()
        logger.set_level(LogLevel.NORMAL)

        caplog.clear()
        logger.verbose("Verbose message")

        assert "Verbose message" in caplog.text

        logger.set_level(LogLevel.QUIET)
        caplog.clear()
        logger.verbose("Should not appear")

        assert "Should not appear" not in caplog.text
