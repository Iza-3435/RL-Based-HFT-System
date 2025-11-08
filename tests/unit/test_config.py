"""Tests for configuration management."""

import pytest
from pathlib import Path
import tempfile
import yaml

from src.core.config import ConfigManager, TradingConfig
from src.core.types import TradingMode, VenueConfig
from src.core.logging_config import LogLevel


class TestConfigManager:
    """Test configuration management."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = ConfigManager(Path("nonexistent.yaml"))

        assert config.trading is not None
        assert len(config.trading.symbols) > 0
        assert config.network is not None
        assert len(config.network.venues) > 0
        assert config.ml is not None
        assert config.system is not None

    def test_trading_config_defaults(self):
        """Test trading configuration defaults."""
        config = ConfigManager()

        assert config.trading.max_position_size == 10000
        assert config.trading.max_portfolio_value == 1_000_000
        assert config.trading.risk_limit_pct == 0.02
        assert 'AAPL' in config.trading.symbols

    def test_network_config_venues(self):
        """Test network venue configuration."""
        config = ConfigManager()

        assert 'NYSE' in config.network.venues
        assert 'NASDAQ' in config.network.venues
        assert config.network.timeout_ms == 1000
        assert config.network.retry_attempts == 3

        nyse = config.network.venues['NYSE']
        assert isinstance(nyse, VenueConfig)
        assert nyse.name == 'NYSE'
        assert nyse.base_latency_us > 0

    def test_ml_config(self):
        """Test ML configuration."""
        config = ConfigManager()

        assert config.ml.enable_routing_optimizer is True
        assert config.ml.enable_latency_prediction is True
        assert config.ml.model_update_interval > 0
        assert config.ml.training_samples > 0

    def test_system_config(self):
        """Test system configuration."""
        config = ConfigManager()

        assert config.system.mode == TradingMode.PRODUCTION
        assert config.system.log_level == LogLevel.NORMAL
        assert config.system.data_dir is not None
        assert config.system.log_dir is not None

    def test_yaml_config_loading(self):
        """Test loading from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'trading': {
                    'symbols': ['AAPL', 'MSFT'],
                    'max_position_size': 5000,
                },
                'system': {
                    'mode': 'balanced',
                    'log_level': 'verbose',
                }
            }
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = ConfigManager(config_path)

            assert config.trading.symbols == ['AAPL', 'MSFT']
            assert config.trading.max_position_size == 5000
            assert config.system.mode == TradingMode.BALANCED
            assert config.system.log_level == LogLevel.VERBOSE

        finally:
            config_path.unlink()

    def test_environment_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv('TRADING_SYMBOLS', 'TSLA,NVDA')
        monkeypatch.setenv('MAX_POSITION_SIZE', '20000')
        monkeypatch.setenv('TRADING_MODE', 'fast')

        config = ConfigManager()

        assert config.trading.symbols == ['TSLA', 'NVDA']
        assert config.trading.max_position_size == 20000
        assert config.system.mode == TradingMode.FAST
