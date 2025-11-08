"""Backtest report generation."""

from typing import Dict, Optional
from datetime import datetime

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestResult

logger = get_logger()


class ReportGenerator:
    """Generate comprehensive backtest reports."""

    def generate_html_report(
        self,
        result: BacktestResult,
        comparison: Optional[Dict] = None
    ) -> str:
        """Generate HTML backtest report."""
        logger.verbose("Generating HTML report")

        html = self._generate_header()
        html += self._generate_summary(result)
        html += self._generate_performance_section(result)
        html += self._generate_risk_section(result)

        if result.stress_test_summary:
            html += self._generate_stress_section(result)

        if result.monte_carlo_analysis:
            html += self._generate_monte_carlo_section(result)

        if comparison:
            html += self._generate_comparison_section(comparison)

        html += self._generate_footer()

        logger.verbose("HTML report generated")
        return html

    def _generate_header(self) -> str:
        """Generate HTML header."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>HFT Backtest Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .positive { color: green; }
        .negative { color: red; }
        .summary-box { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
"""

    def _generate_summary(self, result: BacktestResult) -> str:
        """Generate executive summary."""
        return_pct = (result.total_pnl / result.config.initial_capital) * 100
        return_class = 'positive' if return_pct > 0 else 'negative'

        return f"""
<h1>HFT Backtest Report</h1>
<div class="summary-box">
    <h2>Executive Summary</h2>
    <p><strong>Period:</strong> {result.start_time.strftime('%Y-%m-%d')} to {result.end_time.strftime('%Y-%m-%d')}</p>
    <p><strong>Initial Capital:</strong> ${result.config.initial_capital:,.0f}</p>
    <p><strong>Final Capital:</strong> ${result.final_capital:,.0f}</p>
    <p><strong>Total P&L:</strong> <span class="{return_class}">${result.total_pnl:,.2f} ({return_pct:.2f}%)</span></p>
    <p><strong>Sharpe Ratio:</strong> {result.sharpe_ratio:.2f}</p>
    <p><strong>Max Drawdown:</strong> <span class="negative">{result.max_drawdown:.2f}%</span></p>
    <p><strong>Total Trades:</strong> {result.total_trades:,}</p>
    <p><strong>Win Rate:</strong> {result.win_rate:.2f}%</p>
</div>
"""

    def _generate_performance_section(self, result: BacktestResult) -> str:
        """Generate performance metrics section."""
        return f"""
<h2>Performance Metrics</h2>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total P&L</td><td>${result.total_pnl:,.2f}</td></tr>
    <tr><td>Sharpe Ratio</td><td>{result.sharpe_ratio:.2f}</td></tr>
    <tr><td>Sortino Ratio</td><td>{result.sortino_ratio:.2f}</td></tr>
    <tr><td>Calmar Ratio</td><td>{result.calmar_ratio:.2f}</td></tr>
    <tr><td>Max Drawdown</td><td>{result.max_drawdown:.2f}%</td></tr>
    <tr><td>Profit Factor</td><td>{result.profit_factor:.2f}</td></tr>
</table>
"""

    def _generate_risk_section(self, result: BacktestResult) -> str:
        """Generate risk analysis section."""
        return f"""
<h2>Risk Analysis</h2>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Winning Trades</td><td>{result.winning_trades}</td></tr>
    <tr><td>Losing Trades</td><td>{result.losing_trades}</td></tr>
    <tr><td>Win Rate</td><td>{result.win_rate:.2f}%</td></tr>
    <tr><td>Avg Trade P&L</td><td>${result.avg_trade_pnl:.2f}</td></tr>
    <tr><td>Total Commission</td><td>${result.total_commission:.2f}</td></tr>
    <tr><td>Total Slippage</td><td>${result.total_slippage:.2f}</td></tr>
</table>
"""

    def _generate_stress_section(self, result: BacktestResult) -> str:
        """Generate stress test section."""
        stress = result.stress_test_summary
        worst = stress.get('worst_case', {})

        html = """
<h2>Stress Test Results</h2>
<div class="summary-box">
"""
        html += f"""
    <p><strong>Worst Scenario:</strong> {worst.get('scenario', 'N/A')}</p>
    <p><strong>Worst Return:</strong> <span class="negative">{worst.get('total_return', 0):.2%}</span></p>
    <p><strong>Worst Drawdown:</strong> {worst.get('max_drawdown', 0):.2%}</p>
"""

        html += "<table><tr><th>Scenario</th><th>Return</th><th>Drawdown</th><th>Survived</th></tr>"

        for scenario, data in stress.get('scenario_results', {}).items():
            survived = '✓' if data.get('survived') else '✗'
            html += f"""
    <tr>
        <td>{scenario}</td>
        <td>{data.get('total_return', 0):.2%}</td>
        <td>{data.get('max_drawdown', 0):.2%}</td>
        <td>{survived}</td>
    </tr>
"""

        html += "</table></div>"
        return html

    def _generate_monte_carlo_section(self, result: BacktestResult) -> str:
        """Generate Monte Carlo section."""
        mc = result.monte_carlo_analysis
        ret_dist = mc.get('return_distribution', {})
        percentiles = ret_dist.get('percentiles', {})

        return f"""
<h2>Monte Carlo Analysis</h2>
<div class="summary-box">
    <p><strong>Simulations:</strong> {mc.get('simulation_count', 0):,}</p>
    <p><strong>Mean Return:</strong> {ret_dist.get('mean', 0):.2%}</p>
    <p><strong>Std Dev:</strong> {ret_dist.get('std', 0):.2%}</p>
    <p><strong>5th Percentile:</strong> {percentiles.get('5th', 0):.2%}</p>
    <p><strong>Median:</strong> {percentiles.get('median', 0):.2%}</p>
    <p><strong>95th Percentile:</strong> {percentiles.get('95th', 0):.2%}</p>
    <p><strong>Positive Scenarios:</strong> {mc.get('positive_scenarios', 0):.1%}</p>
    <p><strong>VaR (95%):</strong> {mc.get('value_at_risk_95', 0):.2%}</p>
</div>
"""

    def _generate_comparison_section(self, comparison: Dict) -> str:
        """Generate comparison section."""
        html = "<h2>Strategy Comparison</h2>"

        for approach, metrics in comparison.get('performance_summary', {}).items():
            html += f"""
<h3>{approach.replace('_', ' ').title()}</h3>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Return</td><td>{metrics.get('total_return', 0):.2%}</td></tr>
    <tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
    <tr><td>Max Drawdown</td><td>{metrics.get('max_drawdown', 0):.2%}</td></tr>
    <tr><td>Total Trades</td><td>{metrics.get('total_trades', 0):,}</td></tr>
</table>
"""

        return html

    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return """
</body>
</html>
"""
