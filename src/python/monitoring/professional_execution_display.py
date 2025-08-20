import time
import sys
import os
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import statistics
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'

    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'

    RESET = '\033[0m'
    CLEAR_LINE = '\033[K'
    CLEAR_SCREEN = '\033[2J'
    MOVE_UP = '\033[1A'
    SAVE_CURSOR = '\033[s'
    RESTORE_CURSOR = '\033[u'
class TradeStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
@dataclass
class LiveTrade:

    trade_id: str
    timestamp: float
    symbol: str
    venue: str
    side: str
    quantity: int
    entry_price: float
    current_price: float
    executed_price: Optional[float] = None
    status: TradeStatus = TradeStatus.PENDING
    latency_us: Optional[float] = None
    slippage_bps: Optional[float] = None
    pnl: float = 0.0
    fees: float = 0.0
    strategy: str = ""
    ml_confidence: Optional[float] = None

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def unrealized_pnl(self) -> float:
        if self.executed_price is None:
            return 0.0
        if self.side.upper() == 'BUY':
            return (self.current_price - self.executed_price) * self.quantity
        else:
            return (self.executed_price - self.current_price) * self.quantity
@dataclass
class PerformanceStats:

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_volume: float = 0.0
    avg_latency_us: float = 0.0
    avg_slippage_bps: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    pnl_per_trade: float = 0.0

    def update(self, trade: LiveTrade):
        self.total_trades += 1
        if trade.is_profitable:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.total_pnl += trade.pnl
        self.total_volume += abs(trade.quantity * (trade.executed_price or trade.entry_price))

        if trade.latency_us:
            self.avg_latency_us = ((self.avg_latency_us * (self.total_trades - 1)) + trade.latency_us) / self.total_trades

        if trade.slippage_bps:
            self.avg_slippage_bps = ((self.avg_slippage_bps * (self.total_trades - 1)) + trade.slippage_bps) / self.total_trades

        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        self.pnl_per_trade = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
class ProfessionalExecutionDisplay:


    def __init__(self, max_displayed_trades: int = 20):
        self.max_displayed_trades = max_displayed_trades
        self.active_trades: Dict[str, LiveTrade] = {}
        self.completed_trades: Deque[LiveTrade] = deque(maxlen=1000)
        self.performance_stats = PerformanceStats()
        self.venue_stats: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)

        self.running = False
        self.display_thread = None
        self.blink_state = False
        self.last_update = time.time()

        self._setup_terminal()

    def _setup_terminal(self):

        print(f"{Colors.CLEAR_SCREEN}\033[?25l", end='', flush=True)

    def start_display(self):

        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

    def stop_display(self):

        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        print("\033[?25h", end='', flush=True)

    def add_trade(self, trade: LiveTrade):

        self.active_trades[trade.trade_id] = trade

    def update_trade(self, trade_id: str, **updates):

        if trade_id in self.active_trades:
            trade = self.active_trades[trade_id]
            for key, value in updates.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)

            if trade.status == TradeStatus.FILLED:
                self.completed_trades.append(trade)
                self.performance_stats.update(trade)
                self.venue_stats[trade.venue].update(trade)
                del self.active_trades[trade_id]

    def _display_loop(self):

        while self.running:
            try:
                self._render_display()
                time.sleep(0.1)
                self.blink_state = not self.blink_state
            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(1.0)

    def _render_display(self):

        print("\033[H", end='')

        self._render_header()

        self._render_performance_dashboard()

        self._render_active_trades()

        self._render_recent_trades()

        self._render_venue_stats()

        self._render_footer()

        sys.stdout.flush()

    def _render_header(self):

        now = datetime.now()
        uptime = time.time() - self.last_update

        header = f
        print(header, end='')

    def _render_performance_dashboard(self):

        stats = self.performance_stats

        pnl_color = Colors.GREEN if stats.total_pnl >= 0 else Colors.RED
        pnl_indicator = "📈" if stats.total_pnl >= 0 else "📉"

        blink = Colors.BLINK if abs(stats.total_pnl) > 1000 and self.blink_state else ""

        win_rate_color = Colors.GREEN if stats.win_rate >= 60 else Colors.YELLOW if stats.win_rate >= 40 else Colors.RED
        latency_color = Colors.GREEN if stats.avg_latency_us < 500 else Colors.YELLOW if stats.avg_latency_us < 1000 else Colors.RED

        dashboard = f
        print(dashboard, end='')

    def _render_active_trades(self):

        print("┌── Live Trading Display ──────────────────────────────────────┐", end='')

        active_trades = list(self.active_trades.values())[-self.max_displayed_trades:]

        for trade in active_trades:
            side_color = Colors.GREEN if trade.side.upper() == 'BUY' else Colors.RED
            side_symbol = "🟢 BUY " if trade.side.upper() == 'BUY' else "🔴 SELL"

            unrealized = trade.unrealized_pnl
            pnl_color = Colors.GREEN if unrealized >= 0 else Colors.RED
            pnl_blink = Colors.BLINK if abs(unrealized) > 100 and self.blink_state else ""

            age = trade.age_seconds
            age_color = Colors.RED if age > 5 else Colors.YELLOW if age > 2 else Colors.GRAY

            venue_color = Colors.CYAN

            print("┌── Live Trading Display ──────────────────────────────────────┐", end='')

        for _ in range(max(0, 10 - len(active_trades))):
            print("│                                                              │", end='')

        print("└──────────────────────────────────────────────────────────────┘", end='')

    def _render_recent_trades(self):

        print("┌── Recent Trades ──────────────────────────────────────────────┐", end='')

        recent_trades = list(self.completed_trades)[-8:]

        for trade in recent_trades:
            trade_time = datetime.fromtimestamp(trade.timestamp).strftime('%H:%M:%S')

            pnl_color = Colors.GREEN if trade.is_profitable else Colors.RED
            pnl_symbol = "✅" if trade.is_profitable else "❌"
            pnl_blink = Colors.BLINK if abs(trade.pnl) > 50 and self.blink_state else ""

            side_color = Colors.GREEN if trade.side.upper() == 'BUY' else Colors.RED
            side_display = f"{side_color}{trade.side.upper():<4}{Colors.RESET}"

            latency_us = trade.latency_us or 0
            latency_color = Colors.GREEN if latency_us < 500 else Colors.YELLOW if latency_us < 1000 else Colors.RED

            slippage = trade.slippage_bps or 0
            slippage_color = Colors.GREEN if abs(slippage) < 1 else Colors.YELLOW if abs(slippage) < 3 else Colors.RED

            print("┌── Live Trading Display ──────────────────────────────────────┐", end='')

        for _ in range(max(0, 8 - len(recent_trades))):
            print("┌── Live Trading Display ──────────────────────────────────────┐", end='')

        print("└──────────────────────────────────────────────────────────────┘", end='')

    def _render_venue_stats(self):

        print("┌── Live Trading Display ──────────────────────────────────────┐", end='')

        sorted_venues = sorted(self.venue_stats.items(), key=lambda x: x[1].total_pnl, reverse=True)

        for venue, stats in sorted_venues[:6]:
            if stats.total_trades > 0:
                recent_trades = [t for t in self.completed_trades if t.venue == venue][-10:]
                if recent_trades:
                    avg_recent_latency = statistics.mean([t.latency_us or 1000 for t in recent_trades])
                    status_color = Colors.GREEN if avg_recent_latency < 800 else Colors.YELLOW if avg_recent_latency < 1500 else Colors.RED
                    status = f"{status_color}● ACTIVE{Colors.RESET}"
                else:
                    status = f"{Colors.GRAY}○ IDLE{Colors.RESET}"
            else:
                status = f"{Colors.GRAY}○ NO DATA{Colors.RESET}"

            pnl_color = Colors.GREEN if stats.total_pnl >= 0 else Colors.RED
            win_rate_color = Colors.GREEN if stats.win_rate >= 60 else Colors.YELLOW if stats.win_rate >= 40 else Colors.RED
            latency_color = Colors.GREEN if stats.avg_latency_us < 500 else Colors.YELLOW if stats.avg_latency_us < 1000 else Colors.RED

            print("┌── Live Trading Display ──────────────────────────────────────┐", end='')

        print("└──────────────────────────────────────────────────────────────┘", end='')

    def _render_footer(self):

        footer = f
        print(footer, end='')
def demo_execution_display():

    display = ProfessionalExecutionDisplay()

    display.start_display()

    venues = ['NYSE', 'NASDAQ', 'ARCA', 'IEX', 'CBOE']
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    try:
        for i in range(50):
            trade = LiveTrade(
                trade_id=f"T{i:04d}",
                timestamp=time.time(),
                symbol=np.random.choice(symbols),
                venue=np.random.choice(venues),
                side=np.random.choice(['BUY', 'SELL']),
                quantity=np.random.randint(100, 10000),
                entry_price=100 + np.random.randn() * 5,
                current_price=100 + np.random.randn() * 5,
                strategy="ML_MOMENTUM"
            )

            display.add_trade(trade)

            time.sleep(0.5)

            if np.random.random() < 0.3 and trade.trade_id in display.active_trades:
                executed_price = trade.entry_price + np.random.randn() * 0.05
                latency = max(200, np.random.lognormal(np.log(800), 0.5))
                slippage = (executed_price - trade.entry_price) / trade.entry_price * 10000
                pnl = (executed_price - trade.entry_price) * trade.quantity * (1 if trade.side == 'BUY' else -1)

                display.update_trade(
                    trade.trade_id,
                    status=TradeStatus.FILLED,
                    executed_price=executed_price,
                    latency_us=latency,
                    slippage_bps=slippage,
                    pnl=pnl
                )

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo stopped by user{Colors.RESET}")
    finally:
        display.stop_display()
        print(f"\n{Colors.GREEN}Professional execution display demo completed!{Colors.RESET}\n")
if __name__ == "__main__":
    demo_execution_display()