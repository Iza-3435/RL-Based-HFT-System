"""Professional trade execution logger."""

from datetime import datetime
from typing import Optional
from src.core.terminal_formatter import TerminalFormatter
from src.core.logging_config import get_logger, LogLevel


class TradeLogger:
    """Logger for trade executions with professional formatting."""

    def __init__(self):
        self.logger = get_logger()
        self.formatter = TerminalFormatter(use_colors=True)
        self.header_printed = False
        self.total_pnl = 0.0
        self.trade_count = 0
        self.wins = 0
        self.losses = 0

    def print_header(self):
        """Print trade table header once."""
        if not self.header_printed and self.logger._level != LogLevel.QUIET:
            print(self.formatter.trade_header())
            self.header_printed = True

    def print_footer(self):
        """Print trade table footer."""
        if self.header_printed and self.logger._level != LogLevel.QUIET:
            print(self.formatter.trade_footer())

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        pnl: float,
        venue: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Log a trade execution."""
        self.trade_count += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1

        # Only print trades in normal/verbose/debug modes
        if self.logger._level != LogLevel.QUIET:
            self.print_header()
            trade_line = self.formatter.trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                pnl=pnl,
                total_pnl=self.total_pnl,
                venue=venue,
                timestamp=timestamp,
            )
            print(trade_line)

    def get_summary_stats(self) -> dict:
        """Get summary statistics."""
        win_rate = (self.wins / self.trade_count * 100) if self.trade_count > 0 else 0
        return {
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
        }


# Global trade logger instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get singleton trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger
