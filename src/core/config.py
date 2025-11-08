"""Configuration management."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import yaml

from src.core.types import VenueConfig, TradingMode
from src.core.logging_config import LogLevel


@dataclass
class TradingConfig:
    """Trading configuration."""

    symbols: List[str]
    max_position_size: int = 10000
    max_portfolio_value: float = 1_000_000
    risk_limit_pct: float = 0.02
    enable_short_selling: bool = True


@dataclass
class NetworkConfig:
    """Network configuration."""

    venues: Dict[str, VenueConfig]
    timeout_ms: int = 1000
    retry_attempts: int = 3


@dataclass
class MLConfig:
    """ML model configuration."""

    enable_routing_optimizer: bool = True
    enable_latency_prediction: bool = True
    model_update_interval: int = 3600
    training_samples: int = 10000


@dataclass
class SystemConfig:
    """System configuration."""

    mode: TradingMode = TradingMode.PRODUCTION
    log_level: LogLevel = LogLevel.NORMAL
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    enable_backtesting: bool = True


class ConfigManager:
    """Configuration manager with environment overrides."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/production.yaml")
        self._load_config()

    def _load_config(self):
        """Load configuration from file and environment."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {}

        self.trading = self._load_trading_config(config_data.get("trading", {}))
        self.network = self._load_network_config(config_data.get("network", {}))
        self.ml = self._load_ml_config(config_data.get("ml", {}))
        self.system = self._load_system_config(config_data.get("system", {}))

    def _load_trading_config(self, data: dict) -> TradingConfig:
        """Load trading configuration."""
        symbols_env = os.getenv("TRADING_SYMBOLS")
        symbols = (
            symbols_env.split(",") if symbols_env else data.get("symbols", self._default_symbols())
        )

        return TradingConfig(
            symbols=symbols,
            max_position_size=int(
                os.getenv("MAX_POSITION_SIZE", data.get("max_position_size", 10000))
            ),
            max_portfolio_value=float(
                os.getenv("MAX_PORTFOLIO_VALUE", data.get("max_portfolio_value", 1_000_000))
            ),
            risk_limit_pct=float(os.getenv("RISK_LIMIT_PCT", data.get("risk_limit_pct", 0.02))),
        )

    def _load_network_config(self, data: dict) -> NetworkConfig:
        """Load network configuration."""
        venues = {
            "NYSE": VenueConfig("NYSE", 850, (50, 200), 0.001, 1.5),
            "NASDAQ": VenueConfig("NASDAQ", 920, (60, 180), 0.0008, 1.3),
            "CBOE": VenueConfig("CBOE", 1100, (80, 250), 0.0015, 1.8),
            "IEX": VenueConfig("IEX", 870, (55, 190), 0.0012, 1.6),
            "ARCA": VenueConfig("ARCA", 880, (60, 210), 0.0009, 1.4),
        }

        return NetworkConfig(
            venues=venues,
            timeout_ms=int(os.getenv("NETWORK_TIMEOUT_MS", data.get("timeout_ms", 1000))),
            retry_attempts=int(os.getenv("NETWORK_RETRY_ATTEMPTS", data.get("retry_attempts", 3))),
        )

    def _load_ml_config(self, data: dict) -> MLConfig:
        """Load ML configuration."""
        return MLConfig(
            enable_routing_optimizer=data.get("enable_routing_optimizer", True),
            enable_latency_prediction=data.get("enable_latency_prediction", True),
            model_update_interval=int(data.get("model_update_interval", 3600)),
            training_samples=int(data.get("training_samples", 10000)),
        )

    def _load_system_config(self, data: dict) -> SystemConfig:
        """Load system configuration."""
        mode_str = os.getenv("TRADING_MODE", data.get("mode", "production"))
        mode = TradingMode(mode_str)

        log_level_str = os.getenv("LOG_LEVEL", data.get("log_level", "normal")).upper()
        log_level = LogLevel[log_level_str]

        return SystemConfig(
            mode=mode,
            log_level=log_level,
            data_dir=Path(os.getenv("DATA_DIR", data.get("data_dir", "data"))),
            log_dir=Path(os.getenv("LOG_DIR", data.get("log_dir", "logs"))),
            enable_backtesting=data.get("enable_backtesting", True),
        )

    @staticmethod
    def _default_symbols() -> List[str]:
        """Default trading symbols."""
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "TSLA",
            "NVDA",
            "META",
            "AMZN",
            "NFLX",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "C",
            "JNJ",
            "PFE",
            "UNH",
            "ABBV",
            "PG",
            "KO",
            "XOM",
            "CVX",
            "DIS",
            "SPY",
            "QQQ",
            "IWM",
            "GLD",
            "TLT",
        ]
