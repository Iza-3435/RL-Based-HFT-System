"""Professional trading terminal output formatter - Bloomberg style."""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class Color:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Backgrounds
    BG_BLACK = "\033[40m"
    BG_GRAY = "\033[100m"
    BG_DARK_GRAY = "\033[48;5;236m"  # Darker grey for alternating rows
    BG_DARKER_GRAY = "\033[48;5;233m"
    BG_HEADER = "\033[48;5;240m"  # Lighter grey for headers
    BG_DARK_GREEN = "\033[48;5;28m"  # Darker green for profits
    BG_DARK_RED = "\033[48;5;88m"  # Darker red for losses


class TerminalFormatter:
    """Format trading output for professional terminal display."""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.row_alternate = False

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors enabled."""
        if not self.use_colors:
            return text
        return f"{color}{text}{Color.RESET}"

    def banner(self, mode: str, duration: int, symbols: int) -> str:
        """Generate startup banner."""
        lines = [
            "",
            "=" * 80,
            "  HFT NETWORK OPTIMIZER - PRODUCTION TRADING SYSTEM",
            "=" * 80,
            f"  Mode: {mode.upper()}  |  Duration: {duration}s  |  Symbols: {symbols}",
            "=" * 80,
            ""
        ]
        return "\n".join(lines)

    def phase_start(self, phase: str, description: str) -> str:
        """Format phase initialization message."""
        return f"[*] {phase}: {description}"

    def phase_complete(self, phase: str, duration_ms: Optional[float] = None) -> str:
        """Format phase completion message."""
        time_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        return f"[+] {phase} ready{time_str}"

    def system_ready(self) -> str:
        """Format system ready message."""
        return (
            f"\n{'=' * 80}\n"
            f"  SYSTEM READY | Trading Active\n"
            f"{'=' * 80}\n"
        )

    def trade_header(self) -> str:
        """Format trade table header."""
        separator = "─" * 120
        header = (
            f"{'Time':<14}"
            f"{'Symbol':<10}"
            f"{'Side':<8}"
            f"{'Qty':>8}"
            f"{'Price':>12}"
            f"{'P&L':>18}"
            f"{'Total P&L':>18}"
            f"{'Venue':<10}"
        )
        # Full-width header with background
        header_line = f"{Color.BG_HEADER}{header:<120}{Color.RESET}"
        return f"\n{separator}\n{header_line}\n{separator}"

    def trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        pnl: float,
        total_pnl: float,
        venue: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Format a single trade execution."""
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        elif pnl < 0:
            self.loss_count += 1

        # Toggle alternating row background
        self.row_alternate = not self.row_alternate
        bg = Color.BG_DARK_GRAY if self.row_alternate else ""

        # Format timestamp
        if timestamp is None:
            timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]

        # Format P&L with colored block background (only on amount)
        if pnl > 0:
            pnl_display = f" {Color.BG_DARK_GREEN}{Color.WHITE}+${pnl:>10,.2f}{Color.RESET} "
        elif pnl < 0:
            pnl_display = f" {Color.BG_DARK_RED}{Color.WHITE}-${abs(pnl):>10,.2f}{Color.RESET} "
        else:
            pnl_display = f"  ${pnl:>10,.2f} "

        # Format total P&L with colored block background (only on amount)
        if total_pnl > 0:
            total_display = f" {Color.BG_DARK_GREEN}{Color.WHITE}${total_pnl:>12,.2f}{Color.RESET} "
        elif total_pnl < 0:
            total_display = f" {Color.BG_DARK_RED}{Color.WHITE}${total_pnl:>12,.2f}{Color.RESET} "
        else:
            total_display = f" ${total_pnl:>12,.2f} "

        venue_str = (venue or "N/A")[:8]

        # Build row content
        row_content = (
            f"{time_str:<14}"
            f"{symbol:<10}"
            f"{side.upper():<8}"
            f"{quantity:>8}"
            f"${price:>11,.2f}"
            f"{pnl_display}"
            f"{total_display}"
            f"{venue_str:<10}"
        )

        # Apply full-width background and pad to 120 chars
        if bg:
            row = f"{bg}{row_content:<120}{Color.RESET}"
        else:
            row = f"{row_content:<120}"

        return row

    def trade_footer(self) -> str:
        """Format trade table footer."""
        separator = "─" * 120
        return f"{separator}\n"

    def training_progress(self, current: int, target: int, elapsed: float) -> str:
        """Format training progress."""
        pct = (current / target * 100) if target > 0 else 0
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        return (
            f"\rTraining: {bar} {pct:>5.1f}% ({current:,}/{target:,} ticks) {elapsed:.0f}s"
        )

    def animated_progress_bar(
        self,
        current: int,
        target: int,
        elapsed: float,
        label: str = "Progress",
        show_rate: bool = True,
    ) -> str:
        """
        Create an animated progress bar with spinning animation.

        Args:
            current: Current progress value
            target: Target value
            elapsed: Elapsed time in seconds
            label: Label to display
            show_rate: Whether to show rate per second

        Returns:
            Formatted progress bar string with carriage return
        """
        # Calculate percentage
        pct = (current / target * 100) if target > 0 else 0

        # Animated spinner - themed for HFT system
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = spinner_frames[int(elapsed * 10) % len(spinner_frames)]

        # Progress bar
        bar_width = 30
        filled = int(bar_width * pct / 100)

        # Create themed progress bar (green for filled, dark gray for empty)
        if filled > 0:
            bar = f"{Color.GREEN}{'█' * filled}{Color.RESET}"
        else:
            bar = ""

        if filled < bar_width:
            bar += f"{Color.DIM}{'░' * (bar_width - filled)}{Color.RESET}"

        # Calculate rate
        rate = current / elapsed if elapsed > 0 else 0

        # Build the progress string with system theme
        progress_str = f"\r{Color.WHITE}{spinner}{Color.RESET} {label}: {bar} "
        progress_str += f"{Color.BOLD}{Color.WHITE}{pct:>5.1f}%{Color.RESET} "
        progress_str += f"({current:,}/{target:,}) "

        if show_rate:
            progress_str += f"| {Color.GREEN if rate > 0 else Color.WHITE}{rate:.0f}/s{Color.RESET} "

        progress_str += f"| {Color.DIM}{elapsed:.1f}s{Color.RESET}"

        return progress_str

    def tick_generation_bar(
        self, tick_count: int, elapsed: float, phase: str = "Generating Market Data"
    ) -> str:
        """
        Create an animated progress bar specifically for tick generation.

        Args:
            tick_count: Number of ticks generated
            elapsed: Elapsed time in seconds
            phase: Current phase description

        Returns:
            Formatted progress string
        """
        # Animated spinner - HFT themed
        spinner_frames = ["◐", "◓", "◑", "◒"]
        spinner = spinner_frames[int(elapsed * 4) % len(spinner_frames)]

        # Calculate rate
        rate = tick_count / elapsed if elapsed > 0 else 0

        # Create pulsing dots
        dots_count = int(elapsed * 2) % 4
        dots = "." * dots_count + " " * (3 - dots_count)

        # Build string with system theme (white, black, green, red, dark grey only)
        progress_str = f"\r{Color.WHITE}{spinner}{Color.RESET} "
        progress_str += f"{Color.BOLD}{phase}{dots}{Color.RESET} "
        progress_str += f"| {Color.GREEN}{tick_count:,}{Color.RESET} ticks "
        progress_str += f"| {Color.WHITE}{rate:.0f}/s{Color.RESET} "
        progress_str += f"| {Color.DIM}{elapsed:.1f}s{Color.RESET}"

        return progress_str

    def summary(
        self,
        duration: float,
        trades: int,
        total_pnl: float,
        win_rate: float,
        sharpe: Optional[float] = None,
        max_dd: Optional[float] = None,
    ) -> str:
        """Format final trading summary."""

        # Format P&L with color
        if total_pnl > 0:
            pnl_display = self._color(f"+${total_pnl:,.2f}", Color.GREEN + Color.BOLD)
        elif total_pnl < 0:
            pnl_display = self._color(f"-${abs(total_pnl):,.2f}", Color.RED + Color.BOLD)
        else:
            pnl_display = f"${total_pnl:,.2f}"

        # Format win rate with color
        if win_rate >= 60:
            wr_display = self._color(f"{win_rate:.1f}%", Color.GREEN)
        elif win_rate >= 50:
            wr_display = self._color(f"{win_rate:.1f}%", Color.YELLOW)
        else:
            wr_display = self._color(f"{win_rate:.1f}%", Color.RED)

        lines = [
            "",
            "=" * 80,
            "  TRADING SUMMARY",
            "=" * 80,
            f"  Duration:        {duration:.0f}s",
            f"  Total Trades:    {trades} ({self._color(f'{self.win_count}W', Color.GREEN)} / {self._color(f'{self.loss_count}L', Color.RED)})",
            f"  Win Rate:        {wr_display}",
            f"  Total P&L:       {pnl_display}",
        ]

        if sharpe is not None:
            sharpe_color = Color.GREEN if sharpe > 1.0 else Color.YELLOW if sharpe > 0 else Color.RED
            lines.append(f"  Sharpe Ratio:    {self._color(f'{sharpe:.2f}', sharpe_color)}")

        if max_dd is not None:
            lines.append(f"  Max Drawdown:    {self._color(f'{max_dd:.2f}%', Color.RED)}")

        lines.extend([
            "=" * 80,
            ""
        ])

        return "\n".join(lines)

    def error(self, message: str) -> str:
        """Format error message."""
        return f"{self._color('[-]', Color.RED)} {message}"

    def warning(self, message: str) -> str:
        """Format warning message."""
        return f"{self._color('[!]', Color.YELLOW)} {message}"

    def info(self, message: str) -> str:
        """Format info message."""
        return f"[i] {message}"

    def metric(self, label: str, value: str, good: bool = True) -> str:
        """Format a metric display."""
        color = Color.GREEN if good else Color.RED
        return f"  {label:<20} {self._color(value, color)}"
