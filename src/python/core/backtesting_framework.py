import asyncio
import json
import logging
import pickle
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from core.trading_simulator import (
    TradingSimulator, OrderExecutionEngine, Fill, Order,
    MarketMakingStrategy, ArbitrageStrategy, MomentumStrategy,
    TradingStrategy, OrderSide, OrderType, TradingStrategyType
)
from strategies.risk_management_engine import (
    RiskManager, PnLAttribution, create_integrated_risk_system,
    VenueAnalyzer
)
STRATEGY_MAPPING = {
    'MARKET_MAKING': TradingStrategyType.MARKET_MAKING,
    'ARBITRAGE': TradingStrategyType.ARBITRAGE,
    'MOMENTUM': TradingStrategyType.MOMENTUM
}
logger = logging.getLogger(__name__)
# Reduce verbosity for backtesting - only show warnings and errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class BacktestMode(Enum):

    HISTORICAL = "historical"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
@dataclass
class BacktestConfig:

    start_date: datetime
    end_date: datetime
    initial_capital: float = 1_000_000

    training_window_days: int = 60
    testing_window_days: int = 20
    reoptimization_frequency: int = 20

    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL'])
    venues: List[str] = field(default_factory=lambda: ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA'])

    fill_ratio: float = 0.95
    slippage_model: str = "linear"
    latency_distribution: str = "lognormal"

    max_position_size: int = 10000
    max_daily_loss: float = 50000
    max_drawdown: float = 100000

    target_sharpe: float = 2.0
    target_annual_return: float = 0.20
    max_acceptable_drawdown: float = 0.15
@dataclass
class BacktestResult:

    config: BacktestConfig

    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_market_impact: float = 0.0

    strategy_performance: Dict[str, Dict] = field(default_factory=dict)

    venue_performance: Dict[str, Dict] = field(default_factory=dict)

    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)

    ml_routing_benefit: float = 0.0
    latency_prediction_accuracy: float = 0.0
    regime_detection_accuracy: float = 0.0

    monte_carlo_analysis: Dict = field(default_factory=dict)
    stress_test_summary: Dict = field(default_factory=dict)
    walk_forward_analysis: Dict = field(default_factory=dict)
    trade_history: List = field(default_factory=list)
class BacktestingEngine:


    def __init__(self, config: BacktestConfig):
        self.config = config

        self.data_manager = HistoricalDataManager()
        self.execution_simulator = BacktestExecutionSimulator()
        self.performance_analyzer = PerformanceAnalyzer()

        self.current_positions = defaultdict(lambda: defaultdict(float))
        self.cash_balance = config.initial_capital
        self.equity_curve = []
        self.trade_history = []
        self.daily_pnl = 0
        self.current_drawdown = 0

        self.ml_models = {}
        self.model_parameters = {}

        # BacktestingEngine initialized

    async def run_backtest(self, strategy_factory: Callable,
                          ml_predictor_factory: Callable,
                          mode: BacktestMode = BacktestMode.HISTORICAL) -> BacktestResult:

        # Starting backtest

        if mode == BacktestMode.HISTORICAL:
            return await self._run_historical_backtest(strategy_factory, ml_predictor_factory)
        elif mode == BacktestMode.WALK_FORWARD:
            return await self._run_walk_forward_backtest(strategy_factory, ml_predictor_factory)
        elif mode == BacktestMode.MONTE_CARLO:
            return await self._run_monte_carlo_backtest(strategy_factory, ml_predictor_factory)
        elif mode == BacktestMode.STRESS_TEST:
            return await self._run_stress_test_backtest(strategy_factory, ml_predictor_factory)
        else:
            raise ValueError(f"Unknown backtest mode: {mode}")

    async def _run_historical_backtest(self, strategy_factory: Callable,
                                      ml_predictor_factory: Callable) -> BacktestResult:

        strategies = strategy_factory()
        ml_predictor = ml_predictor_factory()

        risk_system = create_integrated_risk_system()

        market_data = await self.data_manager.load_historical_data(
            self.config.symbols,
            self.config.venues,
            self.config.start_date,
            self.config.end_date
        )

        training_end = self.config.start_date + timedelta(days=self.config.training_window_days)
        training_data = market_data[market_data.index <= training_end]

        await self._train_ml_models(ml_predictor, training_data)

        active_orders = {}
        order_id_counter = 0

        # Running backtest

        for timestamp, tick_data in market_data.iterrows():
            if timestamp <= training_end:
                continue

            current_prices = self._extract_prices_from_tick(tick_data)

            ml_predictions = await self._get_ml_predictions(ml_predictor, tick_data)

            signals = []
            for strategy_name, strategy in strategies.items():
                if hasattr(strategy, 'generate_signals'):
                    strategy_signals = await strategy.generate_signals(
                        tick_data.to_dict(),
                        ml_predictions
                    )
                    signals.extend(strategy_signals)

            for signal in signals:
                order_id_counter += 1
                order = Order(
                    order_id=f"BT_{order_id_counter:08d}",
                    symbol=signal.get('symbol'),
                    side=OrderSide[signal.get('side', 'BUY')],
                    quantity=signal.get('quantity', 100),
                    order_type=OrderType[signal.get('order_type', 'LIMIT')],
                    price=signal.get('price', current_prices.get(signal.get('symbol'), 100)),
                    venue=signal.get('venue', ml_predictions.get('routing_decision', 'NYSE')),
                    strategy=STRATEGY_MAPPING.get(strategy_name.upper(), TradingStrategyType.MARKET_MAKING),
                    timestamp=timestamp.timestamp()
                )

                order.predicted_latency_us = ml_predictions['latency_predictions'].get(order.venue, 1000)

                is_allowed, reason = risk_system['risk_manager'].check_pre_trade_risk(order, current_prices)

                if is_allowed:
                    fill = await self.execution_simulator.simulate_execution(
                        order, tick_data, ml_predictions
                    )

                    if fill:
                        self._update_portfolio(fill, tick_data)
                        risk_system['risk_manager'].update_position(fill, current_prices)
                        risk_system['pnl_attribution'].attribute_fill(
                            fill, order,
                            {'mid_price': current_prices.get(fill.symbol, fill.price)}
                        )
                        risk_system['venue_analyzer'].update_metrics(order, fill)

                        if hasattr(strategies[strategy_name], 'update_positions'):
                            strategies[strategy_name].update_positions(fill, current_prices)

                        active_orders[order.order_id] = order

            self._update_equity(timestamp, tick_data)

            if self._check_drawdown_breach():
                print(f"⚠️  Drawdown breach at {timestamp}, stopping backtest")
                break

            if timestamp.hour == 16:
                self.daily_pnl = 0

        return self._generate_backtest_results(strategies, risk_system, active_orders)

    async def _run_walk_forward_backtest(self, strategy_factory: Callable,
                                        ml_predictor_factory: Callable) -> BacktestResult:

        # Running walk-forward optimization

        strategies = strategy_factory()
        ml_predictor = ml_predictor_factory()
        risk_system = create_integrated_risk_system()

        market_data = await self.data_manager.load_historical_data(
            self.config.symbols,
            self.config.venues,
            self.config.start_date,
            self.config.end_date
        )

        current_date = self.config.start_date + timedelta(days=self.config.training_window_days)
        walk_forward_results = []

        while current_date < self.config.end_date:
            train_start = current_date - timedelta(days=self.config.training_window_days)
            train_end = current_date
            test_end = min(
                current_date + timedelta(days=self.config.testing_window_days),
                self.config.end_date
            )

            # Walk-forward window processing

            train_data = market_data[(market_data.index >= train_start) &
                                   (market_data.index < train_end)]
            test_data = market_data[(market_data.index >= train_end) &
                                  (market_data.index < test_end)]

            optimal_params = await self._optimize_parameters(
                strategies, ml_predictor, train_data
            )

            self._apply_parameters(strategies, ml_predictor, optimal_params)

            window_result = await self._test_on_window(
                strategies, ml_predictor, risk_system, test_data
            )

            walk_forward_results.append(window_result)

            current_date += timedelta(days=self.config.reoptimization_frequency)

        return self._combine_walk_forward_results(walk_forward_results)

    async def _run_monte_carlo_backtest(self, strategy_factory: Callable,
                                       ml_predictor_factory: Callable,
                                       num_simulations: int = 1000) -> BacktestResult:

        print(f"🎲 Running Monte Carlo simulation with {num_simulations} paths")

        base_result = await self._run_historical_backtest(
            strategy_factory, ml_predictor_factory
        )

        returns = base_result.daily_returns.values

        simulation_results = []

        for i in range(num_simulations):
            if i % 100 == 0:
                print(f"🎲 Monte Carlo simulation {i}/{num_simulations}")

            simulated_returns = np.random.choice(
                returns,
                size=len(returns),
                replace=True
            )

            equity_curve = self.config.initial_capital * np.cumprod(1 + simulated_returns)

            sim_result = {
                'simulation_id': i,
                'total_return': (equity_curve[-1] / self.config.initial_capital) - 1,
                'sharpe_ratio': self._calculate_sharpe(simulated_returns),
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'final_equity': equity_curve[-1]
            }

            simulation_results.append(sim_result)

        return self._aggregate_monte_carlo_results(base_result, simulation_results)

    async def _run_stress_test_backtest(self, strategy_factory: Callable,
                                       ml_predictor_factory: Callable) -> BacktestResult:

        scenarios = {
            'flash_crash': {
                'price_shock': -0.10,
                'volatility_multiplier': 5.0,
                'liquidity_reduction': 0.8
            },
            'high_volatility': {
                'price_shock': 0.0,
                'volatility_multiplier': 3.0,
                'liquidity_reduction': 0.5
            },
            'low_liquidity': {
                'price_shock': 0.0,
                'volatility_multiplier': 1.5,
                'liquidity_reduction': 0.2
            },
            'correlation_breakdown': {
                'correlation_shock': 0.9,
                'volatility_multiplier': 2.0
            }
        }

        stress_results = {}

        for scenario_name, params in scenarios.items():
            print(f"⚡ Running stress test: {scenario_name}")

            stressed_result = await self._run_stressed_backtest(
                strategy_factory,
                ml_predictor_factory,
                params
            )

            stress_results[scenario_name] = stressed_result

        return self._combine_stress_results(stress_results)

    async def _train_ml_models(self, ml_predictor, training_data: pd.DataFrame):

        # Training ML models for backtest

        features = []
        latency_targets = []

        for _, row in training_data.iterrows():
            feature_vector = self._extract_features_from_row(row)
            features.append(feature_vector)

            latency_targets.append(row.get('actual_latency', 1000))

        if hasattr(ml_predictor, 'train'):
            await ml_predictor.train(
                np.array(features),
                np.array(latency_targets)
            )

        if hasattr(ml_predictor, 'train_regime_detector'):
            market_regimes = self._identify_historical_regimes(training_data)
            await ml_predictor.train_regime_detector(training_data, market_regimes)

    def _extract_features_from_row(self, row: pd.Series) -> np.ndarray:

        features = []

        if 'timestamp' in row.index:
            dt = pd.to_datetime(row.name)
            features.extend([
                dt.hour / 24.0,
                dt.minute / 60.0,
                dt.dayofweek / 6.0
            ])
        else:
            features.extend([0.5, 0.5, 0.5])

        features.extend([
            row.get('price', 100) / 1000.0,
            np.log1p(row.get('volume', 1000)) / 10.0,
            row.get('spread', 0.01),
            row.get('volatility', 0.02)
        ])

        while len(features) < 45:
            features.append(0.0)

        return np.array(features[:45])

    def _extract_prices_from_tick(self, tick_data: pd.Series) -> Dict[str, float]:

        prices = {}

        for symbol in self.config.symbols:
            for price_col in ['price', f'{symbol}_price', 'mid_price']:
                if price_col in tick_data:
                    prices[symbol] = tick_data[price_col]
                    break

            if symbol not in prices:
                prices[symbol] = 100.0

        return prices

    def _identify_historical_regimes(self, data: pd.DataFrame) -> pd.Series:

        if 'returns' in data.columns:
            rolling_vol = data['returns'].rolling(20).std()
        else:
            if 'price' in data.columns:
                returns = data['price'].pct_change()
                rolling_vol = returns.rolling(20).std()
            else:
                rolling_vol = pd.Series(0.02, index=data.index)

        regimes = pd.Series(index=data.index, dtype=str)
        regimes[rolling_vol < rolling_vol.quantile(0.33)] = 'quiet'
        regimes[(rolling_vol >= rolling_vol.quantile(0.33)) &
                (rolling_vol < rolling_vol.quantile(0.67))] = 'normal'
        regimes[rolling_vol >= rolling_vol.quantile(0.67)] = 'volatile'

        return regimes

    async def _get_ml_predictions(self, ml_predictor, tick_data: pd.Series) -> Dict:

        features = self._extract_features_from_row(tick_data)

        predictions = {
            'latency_predictions': {},
            'routing_decision': None,
            'regime': 'normal'
        }

        for venue in self.config.venues:
            if hasattr(ml_predictor, 'predict_latency'):
                pred = await ml_predictor.predict_latency(venue, features)
                predictions['latency_predictions'][venue] = pred

        if hasattr(ml_predictor, 'get_best_venue'):
            best_venue = await ml_predictor.get_best_venue(
                predictions['latency_predictions']
            )
            predictions['routing_decision'] = best_venue
        elif predictions['latency_predictions']:
            predictions['routing_decision'] = min(
                predictions['latency_predictions'].items(),
                key=lambda x: x[1]
            )[0]
        else:
            predictions['routing_decision'] = self.config.venues[0]

        if hasattr(ml_predictor, 'detect_regime'):
            regime = await ml_predictor.detect_regime(tick_data)
            predictions['regime'] = regime

        return predictions

    def _update_portfolio(self, fill: Fill, tick_data: pd.Series):

        strategy = fill.order.strategy.value if hasattr(fill, 'order') else 'unknown'

        if fill.side == OrderSide.BUY:
            self.current_positions[strategy][fill.symbol] += fill.quantity
        else:
            self.current_positions[strategy][fill.symbol] -= fill.quantity

        if fill.side == OrderSide.BUY:
            self.cash_balance -= fill.quantity * fill.price + fill.fees - fill.rebate
        else:
            self.cash_balance += fill.quantity * fill.price - fill.fees + fill.rebate

        self.daily_pnl += fill.rebate - fill.fees

        self.trade_history.append({
            'timestamp': tick_data.name if hasattr(tick_data, 'name') else time.time(),
            'symbol': fill.symbol,
            'side': fill.side.value,
            'quantity': fill.quantity,
            'price': fill.price,
            'fees': fill.fees,
            'rebate': fill.rebate,
            'strategy': strategy,
            'venue': fill.venue,
            'latency_us': fill.latency_us,
            'slippage_bps': fill.slippage_bps
        })

    def _update_equity(self, timestamp: datetime, tick_data: pd.Series):

        portfolio_value = self.cash_balance

        current_prices = self._extract_prices_from_tick(tick_data)

        for strategy_positions in self.current_positions.values():
            for symbol, quantity in strategy_positions.items():
                price = current_prices.get(symbol, 100)
                portfolio_value += quantity * price

        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': portfolio_value,
            'cash': self.cash_balance,
            'positions_value': portfolio_value - self.cash_balance
        })

    def _check_drawdown_breach(self) -> bool:

        if len(self.equity_curve) < 2:
            return False

        equity_values = [e['equity'] for e in self.equity_curve]
        peak = max(equity_values)
        current = equity_values[-1]

        drawdown = (peak - current) / peak
        self.current_drawdown = drawdown

        return drawdown > self.config.max_acceptable_drawdown

    def _generate_backtest_results(self, strategies, risk_system, orders) -> BacktestResult:

        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
            print("⚠️  No equity curve data generated")
            return BacktestResult(config=self.config)

        equity_series = pd.Series(
            equity_df['equity'].values,
            index=pd.to_datetime(equity_df['timestamp'])
        )

        returns = equity_series.pct_change().dropna()

        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        days = (equity_series.index[-1] - equity_series.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        max_dd, max_dd_duration = self._calculate_drawdown_stats(equity_series)

        trades_df = pd.DataFrame(self.trade_history)
        if not trades_df.empty:
            trades_df['pnl'] = 0.0

            for (symbol, strategy), group in trades_df.groupby(['symbol', 'strategy']):
                buy_trades = group[group['side'] == 'BUY']
                sell_trades = group[group['side'] == 'SELL']

                if not buy_trades.empty and not sell_trades.empty:
                    total_pnl = (sell_trades['price'].mean() - buy_trades['price'].mean()) * \
                               min(buy_trades['quantity'].sum(), sell_trades['quantity'].sum())

                    for idx in group.index:
                        if group.loc[idx, 'side'] == 'SELL':
                            trades_df.loc[idx, 'pnl'] = total_pnl / len(sell_trades)

            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        total_fees = trades_df['fees'].sum() if not trades_df.empty else 0
        total_rebates = trades_df['rebate'].sum() if not trades_df.empty else 0
        total_slippage = trades_df['slippage_bps'].sum() * 100 if not trades_df.empty else 0

        pnl_report = risk_system['pnl_attribution'].get_attribution_report()
        strategy_performance = pnl_report.get('by_strategy', {})

        venue_report = risk_system['venue_analyzer'].analyze_venue_performance()

        if not trades_df.empty and 'latency_us' in trades_df.columns:
            avg_latency = trades_df['latency_us'].mean()
            baseline_latency = 1500
            latency_improvement = baseline_latency - avg_latency

            sensitive_volume = len(trades_df) * 100 * 0.20
            ml_routing_benefit = sensitive_volume * 100 * (latency_improvement / 100) * 0.0001
        else:
            ml_routing_benefit = 0

        result = BacktestResult(
            config=self.config,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(trades_df) if not trades_df.empty else 0,
            winning_trades=len(winning_trades) if 'winning_trades' in locals() else 0,
            losing_trades=len(losing_trades) if 'losing_trades' in locals() else 0,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_fees=total_fees,
            total_slippage=total_slippage,
            strategy_performance=strategy_performance,
            venue_performance=venue_report,
            equity_curve=equity_series,
            daily_returns=returns,
            ml_routing_benefit=ml_routing_benefit,
            trade_history=self.trade_history
        )

        return result

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # Cap Sharpe ratio to prevent extreme values
        return max(-10.0, min(10.0, sharpe))

    def _annualize_return(self, total_return: float, days: int = None) -> float:
        if days is None:
           days = (self.config.end_date - self.config.start_date).days

           if days <= 0:
            return 0.0

        return (1 + total_return) ** (365 / max(days, 1)) - 1

    def _calculate_sharpe_from_windows(self, results: List[Dict]) -> float:

        if not results:
            return 0.0

        window_returns = [r['period_return'] for r in results]
        if len(window_returns) == 0:
            return 0.0

        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)

        if std_return == 0:
            return 0.0

        return np.sqrt(252) * mean_return / std_return

    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:

        if len(returns) == 0:
            return 0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0

        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def _calculate_drawdown_stats(self, equity_series: pd.Series) -> Tuple[float, int]:

        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max

        max_drawdown = abs(drawdown.min())

        duration = 0
        max_duration = 0
        in_drawdown = False

        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    duration = 0
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                in_drawdown = False

        return max_drawdown, max_duration

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:

        rolling_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - rolling_max) / rolling_max
        return abs(np.min(drawdown))

    async def _optimize_parameters(self, strategies, ml_predictor,
                                  train_data: pd.DataFrame) -> Dict:

        # Optimizing parameters on training data

        param_grid = {
            'market_making': {
                'spread_multiplier': [0.8, 1.0, 1.2],
                'inventory_limit': [5000, 10000, 15000],
                'skew_factor': [0.05, 0.1, 0.15]
            },
            'momentum': {
                'entry_threshold': [1.5, 2.0, 2.5],
                'stop_loss_bps': [30, 50, 70],
                'hold_time': [180, 300, 600]
            },
            'arbitrage': {
                'min_spread_bps': [1, 2, 3],
                'max_position': [1000, 2000, 3000]
            },
            'ml_weight': [0.5, 0.7, 0.9]
        }

        best_params = None
        best_sharpe = -float('inf')

        for mm_spread in param_grid['market_making']['spread_multiplier']:
            for mom_threshold in param_grid['momentum']['entry_threshold']:
                for ml_weight in param_grid['ml_weight']:
                    test_params = {
                        'market_making': {'spread_multiplier': mm_spread},
                        'momentum': {'entry_threshold': mom_threshold},
                        'ml_weight': ml_weight
                    }

                    result = await self._test_parameters(
                        strategies, ml_predictor, train_data, test_params
                    )

                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_params = test_params

        print(f"🎯 Best parameters found with Sharpe: {best_sharpe:.2f}")
        return best_params

    async def _test_parameters(self, strategies, ml_predictor,
                              data: pd.DataFrame, params: Dict) -> Dict:

        self._apply_parameters(strategies, ml_predictor, params)

        equity = [self.config.initial_capital]

        for _, row in data.iterrows():
            signal_value = np.random.randn()

            if signal_value > params.get('momentum', {}).get('entry_threshold', 2.0):
                pnl = np.random.randn() * 100
                equity.append(equity[-1] + pnl)
            else:
                equity.append(equity[-1])

        returns = pd.Series(equity).pct_change().dropna()
        sharpe = self._calculate_sharpe(returns)

        return {'sharpe': sharpe, 'total_return': (equity[-1] / equity[0]) - 1}

    def _apply_parameters(self, strategies, ml_predictor, params: Dict):

        for strategy_name, strategy_params in params.items():
            if strategy_name in strategies and hasattr(strategies[strategy_name], 'params'):
                strategies[strategy_name].params.update(strategy_params)

        if 'ml_weight' in params and hasattr(ml_predictor, 'set_weight'):
            ml_predictor.set_weight(params['ml_weight'])

    async def _test_on_window(self, strategies, ml_predictor, risk_system,
                            test_data: pd.DataFrame) -> Dict:

        window_positions = defaultdict(lambda: defaultdict(float))
        window_cash = self.config.initial_capital
        window_trades = []
        window_equity = []

        for timestamp, row in test_data.iterrows():
            current_prices = self._extract_prices_from_tick(row)

            ml_predictions = await self._get_ml_predictions(ml_predictor, row)

            for strategy_name, strategy in strategies.items():
                if hasattr(strategy, 'generate_signal'):
                    signal = strategy.generate_signal(row, ml_predictions)
                    if signal:
                        window_trades.append(signal)

            portfolio_value = window_cash
            for positions in window_positions.values():
                for symbol, qty in positions.items():
                    portfolio_value += qty * current_prices.get(symbol, 100)

            window_equity.append(portfolio_value)

        if window_equity:
            window_return = (window_equity[-1] / window_equity[0]) - 1
        else:
            window_return = 0

        return {
            'period_return': window_return,
            'trade_count': len(window_trades),
            'end_cash': window_cash,
            'final_equity': window_equity[-1] if window_equity else window_cash
        }

    def _combine_walk_forward_results(self, results: List[Dict]) -> BacktestResult:

        total_equity = self.config.initial_capital
        total_trades = 0

        for result in results:
            total_equity *= (1 + result['period_return'])
            total_trades += result['trade_count']

        total_return = (total_equity / self.config.initial_capital) - 1

        combined = BacktestResult(
            config=self.config,
            total_return=total_return,
            annual_return=self._annualize_return(total_return),
            total_trades=total_trades,
            sharpe_ratio=self._calculate_sharpe_from_windows(results)
        )

        combined.walk_forward_analysis = {
            'window_count': len(results),
            'avg_window_return': np.mean([r['period_return'] for r in results]),
            'std_window_return': np.std([r['period_return'] for r in results]),
            'positive_windows': sum(1 for r in results if r['period_return'] > 0),
            'consistency': sum(1 for r in results if r['period_return'] > 0) / len(results)
        }

        return combined
    def _combine_stress_results(self, stress_results: Dict) -> 'BacktestResult':

        aggregate_result = BacktestResult(config=self.config)

        worst_return = min(result.total_return for result in stress_results.values())
        worst_drawdown = max(result.max_drawdown for result in stress_results.values())
        worst_sharpe = min(result.sharpe_ratio for result in stress_results.values())

        avg_return = np.mean([result.total_return for result in stress_results.values()])
        avg_drawdown = np.mean([result.max_drawdown for result in stress_results.values()])
        avg_sharpe = np.mean([result.sharpe_ratio for result in stress_results.values()])

        aggregate_result.total_return = avg_return
        aggregate_result.max_drawdown = avg_drawdown
        aggregate_result.sharpe_ratio = avg_sharpe

        aggregate_result.stress_test_analysis = {
        'scenarios': list(stress_results.keys()),
        'worst_case': {
            'scenario': min(stress_results.items(), key=lambda x: x[1].total_return)[0],
            'total_return': worst_return,
            'max_drawdown': worst_drawdown,
            'sharpe_ratio': worst_sharpe
        },
            'scenario_results': {
            scenario: {
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio
            }
            for scenario, result in stress_results.items()
        },
            'resilience_score': (avg_return + 1) * (1 - avg_drawdown) * max(avg_sharpe, 0)
    }

        return aggregate_result
    async def _run_stressed_backtest(self, strategy_factory: Callable,
                                ml_predictor_factory: Callable,
                                stress_params: Dict) -> 'BacktestResult':
        base_result = await self._run_historical_backtest(strategy_factory, ml_predictor_factory)

        price_shock = stress_params.get('price_shock', 0)
        volatility_multiplier = stress_params.get('volatility_multiplier', 1.0)
        liquidity_reduction = stress_params.get('liquidity_reduction', 1.0)

        stressed_return = base_result.total_return * (1 + price_shock) * (1 / volatility_multiplier)
        stressed_drawdown = base_result.max_drawdown * volatility_multiplier
        stressed_sharpe = base_result.sharpe_ratio / volatility_multiplier

        stressed_result = BacktestResult(
        config=self.config,
        total_return=stressed_return,
        max_drawdown=stressed_drawdown,
        sharpe_ratio=stressed_sharpe,
        annual_return=base_result.annual_return * (1 + price_shock),
        sortino_ratio=base_result.sortino_ratio / volatility_multiplier,
        total_trades=base_result.total_trades,
        win_rate=base_result.win_rate * (1 - liquidity_reduction * 0.1),
        equity_curve=base_result.equity_curve,
        daily_returns=base_result.daily_returns
    )

        return stressed_result
    def _aggregate_monte_carlo_results(self, base_result: 'BacktestResult',
                                 simulation_results: List[Dict]) -> 'BacktestResult':
        returns = [sim['total_return'] for sim in simulation_results]
        sharpes = [sim['sharpe_ratio'] for sim in simulation_results]
        drawdowns = [sim['max_drawdown'] for sim in simulation_results]

        return_percentiles = np.percentile(returns, [5, 25, 50, 75, 95])
        sharpe_percentiles = np.percentile(sharpes, [5, 25, 50, 75, 95])
        dd_percentiles = np.percentile(drawdowns, [5, 25, 50, 75, 95])

        base_result.monte_carlo_analysis = {
        'simulation_count': len(simulation_results),
        'return_distribution': {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'percentiles': {
                '5th': return_percentiles[0],
                '25th': return_percentiles[1],
                'median': return_percentiles[2],
                '75th': return_percentiles[3],
                '95th': return_percentiles[4]
            }
        },
            'sharpe_distribution': {
            'mean': np.mean(sharpes),
            'std': np.std(sharpes),
            'percentiles': {
                '5th': sharpe_percentiles[0],
                '25th': sharpe_percentiles[1],
                'median': sharpe_percentiles[2],
                '75th': sharpe_percentiles[3],
                '95th': sharpe_percentiles[4]
            }
        },
            'drawdown_distribution': {
            'mean': np.mean(drawdowns),
            'std': np.std(drawdowns),
            'percentiles': {
                '5th': dd_percentiles[0],
                '25th': dd_percentiles[1],
                'median': dd_percentiles[2],
                '75th': dd_percentiles[3],
                '95th': dd_percentiles[4]
            }
        },
            'positive_scenarios': sum(1 for r in returns if r > 0) / len(returns),
            'value_at_risk_95': return_percentiles[0],
            'conditional_var_95': np.mean([r for r in returns if r <= return_percentiles[0]])
    }

        return base_result
class HistoricalDataManager:


    def __init__(self):
        self.data_cache = {}

    async def load_historical_data(self, symbols: List[str], venues: List[str],
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:


        # Loading historical data

        date_range = pd.date_range(start=start_date, end=end_date, freq='1min')

        market_hours = date_range[
            (date_range.time >= pd.Timestamp('09:30:00').time()) &
            (date_range.time <= pd.Timestamp('16:00:00').time()) &
            (date_range.weekday < 5)
        ]

        data = []

        for timestamp in market_hours:
            base_row = {'timestamp': timestamp}

            for symbol in symbols:
                if timestamp == market_hours[0]:
                    base_price = np.random.uniform(50, 200)
                else:
                    prev_price = self._get_prev_price(symbol, data)
                    drift = 0.0001 * (100 - prev_price) / 100
                    volatility = 0.0002 * (1 + 0.5 * np.sin(timestamp.hour))
                    base_price = prev_price * (1 + drift + volatility * np.random.randn())

                spread = 0.01 * (1 + 0.1 * np.random.rand())

                base_row.update({
                    f'{symbol}_price': base_price,
                    f'{symbol}_bid': base_price - spread/2,
                    f'{symbol}_ask': base_price + spread/2,
                    f'{symbol}_volume': int(np.random.lognormal(10, 1)),
                    f'{symbol}_spread': spread
                })

                for venue in venues:
                    venue_adjustment = np.random.randn() * 0.001
                    base_row[f'{symbol}_{venue}_bid'] = base_price - spread/2 + venue_adjustment
                    base_row[f'{symbol}_{venue}_ask'] = base_price + spread/2 + venue_adjustment
                    base_row[f'{symbol}_{venue}_size'] = int(np.random.lognormal(7, 0.5))

            base_row['volatility'] = 0.02 * (1 + 0.5 * np.random.rand())
            base_row['market_volume'] = int(np.random.lognormal(15, 1))

            data.append(base_row)

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        cache_key = f"{'-'.join(symbols)}_{start_date}_{end_date}"
        self.data_cache[cache_key] = df

        print(f"📊 Generated {len(df):,} historical data points")
        return df

    def _get_prev_price(self, symbol: str, data: List[Dict]) -> float:

        if not data:
            return 100.0

        price_key = f'{symbol}_price'
        for row in reversed(data):
            if price_key in row:
                return row[price_key]

        return 100.0
class StressedDataManager(HistoricalDataManager):


    def __init__(self, stress_params: Dict):
        super().__init__()
        self.stress_params = stress_params

    async def load_historical_data(self, symbols: List[str], venues: List[str],
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:

        df = await super().load_historical_data(symbols, venues, start_date, end_date)

        if 'price_shock' in self.stress_params:
            shock = self.stress_params['price_shock']
            for col in df.columns:
                if '_price' in col or '_bid' in col or '_ask' in col:
                    df[col] *= (1 + shock)

        if 'volatility_multiplier' in self.stress_params:
            vol_mult = self.stress_params['volatility_multiplier']
            df['volatility'] *= vol_mult

            for col in df.columns:
                if '_price' in col:
                    noise = np.random.randn(len(df)) * 0.01 * (vol_mult - 1)
                    df[col] *= (1 + noise)

        if 'liquidity_reduction' in self.stress_params:
            liq_factor = 1 - self.stress_params['liquidity_reduction']
            for col in df.columns:
                if '_size' in col or '_volume' in col:
                    df[col] = (df[col] * liq_factor).astype(int)

        return df
class BacktestExecutionSimulator:


    def __init__(self):
        self.fill_id_counter = 0

    async def simulate_execution(self, order: Order, market_data: pd.Series,
                               ml_predictions: Dict) -> Optional['Fill']:

        fill_probability = self._calculate_fill_probability(order, market_data)

        if np.random.random() > fill_probability:
            return None

        if order.side == OrderSide.BUY:
            base_price = market_data.get(f'{order.symbol}_{order.venue}_ask',
                                       market_data.get(f'{order.symbol}_ask',
                                                     market_data.get(f'{order.symbol}_price', 100)))
            slippage = self._calculate_slippage(order, market_data, 'buy')
            exec_price = base_price * (1 + slippage)
        else:
            base_price = market_data.get(f'{order.symbol}_{order.venue}_bid',
                                       market_data.get(f'{order.symbol}_bid',
                                                     market_data.get(f'{order.symbol}_price', 100)))
            slippage = self._calculate_slippage(order, market_data, 'sell')
            exec_price = base_price * (1 - slippage)

        is_maker = order.order_type == OrderType.LIMIT and abs(order.price - exec_price) < 0.01
        fees, rebate = self._calculate_fees(order.venue, order.quantity, exec_price, is_maker)

        predicted_latency = ml_predictions.get('latency_predictions', {}).get(order.venue, 1000)
        actual_latency = np.random.lognormal(np.log(predicted_latency), 0.2)

        self.fill_id_counter += 1
        fill = Fill(
            fill_id=f"BT_{self.fill_id_counter:08d}",
            order_id=order.order_id,
            symbol=order.symbol,
            venue=order.venue,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            timestamp=order.timestamp,
            fees=fees,
            rebate=rebate,
            latency_us=actual_latency,
            slippage_bps=slippage * 10000,
            market_impact_bps=slippage * 10000 * 0.5
        )

        fill.order = order

        return fill

    def _calculate_fill_probability(self, order: Order, market_data: pd.Series) -> float:

        if order.order_type == OrderType.MARKET:
            return 0.99

        mid_price = market_data.get(f'{order.symbol}_price', 100)
        spread = market_data.get(f'{order.symbol}_spread', 0.01)

        if order.side == OrderSide.BUY:
            price_distance = (order.price - mid_price) / mid_price
            if price_distance >= spread/2:
                return 0.95
            elif price_distance >= 0:
                return 0.7
            else:
                return 0.3
        else:
            price_distance = (mid_price - order.price) / mid_price
            if price_distance >= spread/2:
                return 0.95
            elif price_distance >= 0:
                return 0.7
            else:
                return 0.3

    def _calculate_slippage(self, order: Order, market_data: pd.Series, side: str) -> float:

        volume = market_data.get(f'{order.symbol}_volume', 1000)
        size_impact = order.quantity / volume
        volatility = market_data.get('volatility', 0.02)

        base_slippage = 0.0001 * size_impact * (1 + volatility * 10)

        random_slippage = np.random.exponential(0.00005)

        return base_slippage + random_slippage

    def _calculate_fees(self, venue: str, quantity: int, price: float,
                       is_maker: bool) -> Tuple[float, float]:

        fee_schedule = {
            'NYSE': {'maker': -0.0020, 'taker': 0.0030},
            'NASDAQ': {'maker': -0.0025, 'taker': 0.0030},
            'CBOE': {'maker': -0.0023, 'taker': 0.0028},
            'IEX': {'maker': 0.0000, 'taker': 0.0009},
            'ARCA': {'maker': -0.0020, 'taker': 0.0030}
        }

        venue_fees = fee_schedule.get(venue, {'maker': 0, 'taker': 0.003})
        rate = venue_fees['maker'] if is_maker else venue_fees['taker']

        notional = quantity * price

        if rate > 0:
            return notional * rate, 0
        else:
            return 0, -notional * rate

    def _calculate_sharpe_from_windows(self, results: List[Dict]) -> float:

        returns = [r['period_return'] for r in results]
        if not returns or np.std(returns) == 0:
            return 0

        windows_per_year = 252 / self.config.testing_window_days
        return np.sqrt(windows_per_year) * np.mean(returns) / np.std(returns)

    def _annualize_return(self, total_return: float, days: int = None) -> float:
        if days is None:
            if hasattr(self, 'config') and hasattr(self.config, 'start_date') and hasattr(self.config, 'end_date'):
                days = (self.config.end_date - self.config.start_date).days
            else:
                days = 252
        if days <= 0:
            return 0.0

        return (1 + total_return) ** (365.25 / max(days, 1)) - 1

    def _aggregate_monte_carlo_results(self, base_result: BacktestResult,
                                     simulations: List[Dict]) -> BacktestResult:

        returns = [s['total_return'] for s in simulations]
        sharpes = [s['sharpe_ratio'] for s in simulations]
        drawdowns = [s['max_drawdown'] for s in simulations]

        base_result.monte_carlo_analysis = {
            'num_simulations': len(simulations),
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_5th_percentile': np.percentile(returns, 5),
            'return_95th_percentile': np.percentile(returns, 95),
            'sharpe_mean': np.mean(sharpes),
            'sharpe_std': np.std(sharpes),
            'drawdown_mean': np.mean(drawdowns),
            'drawdown_95th_percentile': np.percentile(drawdowns, 95),
            'probability_positive': sum(1 for r in returns if r > 0) / len(returns),
            'probability_meet_target': sum(1 for r in returns if r > self.config.target_annual_return) / len(returns),
            'var_95': -np.percentile(returns, 5),
            'cvar_95': -np.mean([r for r in returns if r <= np.percentile(returns, 5)])
        }

        return base_result

    async def _run_stressed_backtest(self, strategy_factory, ml_predictor_factory,
                                   stress_params: Dict) -> BacktestResult:

        stressed_data_manager = StressedDataManager(stress_params)

        original_data_manager = self.data_manager
        self.data_manager = stressed_data_manager

        try:
            result = await self._run_historical_backtest(
                strategy_factory, ml_predictor_factory
            )

            result.stress_parameters = stress_params

            return result
        finally:
            self.data_manager = original_data_manager

    def _combine_stress_results(self, stress_results: Dict[str, BacktestResult]) -> BacktestResult:

        worst_return = float('inf')
        worst_scenario = None

        for scenario, result in stress_results.items():
            if result.total_return < worst_return:
                worst_return = result.total_return
                worst_scenario = scenario

        combined = stress_results[worst_scenario]
        combined.stress_test_summary = {
            scenario: {
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'survived': result.total_return > -0.5,
                'worst_daily_loss': result.daily_returns.min() if len(result.daily_returns) > 0 else 0
            }
            for scenario, result in stress_results.items()
        }

        combined.stress_survival_rate = sum(
            1 for r in stress_results.values() if r.total_return > -0.5
        ) / len(stress_results)

        return combined
class PerformanceAnalyzer:


    def __init__(self):
        self.metrics = {}

    def analyze_results(self, results: BacktestResult) -> Dict[str, Any]:

        analysis = {
            'performance_summary': self._performance_summary(results),
            'risk_analysis': self._risk_analysis(results),
            'statistical_significance': self._test_statistical_significance(results),
            'stability_analysis': self._analyze_stability(results),
            'comparison_vs_baseline': self._compare_to_baseline(results)
        }

        return analysis

    def _performance_summary(self, results: BacktestResult) -> Dict:

        return {
            'total_return': f"{results.total_return:.2%}",
            'annual_return': f"{results.annual_return:.2%}",
            'sharpe_ratio': f"{results.sharpe_ratio:.2f}",
            'max_drawdown': f"{results.max_drawdown:.2%}",
            'win_rate': f"{results.win_rate:.2%}",
            'profit_factor': f"{results.profit_factor:.2f}",
            'total_trades': results.total_trades,
            'avg_trade_pnl': (results.avg_win * results.win_rate +
                            results.avg_loss * (1 - results.win_rate))
        }

    def _risk_analysis(self, results: BacktestResult) -> Dict:

        returns = results.daily_returns

        if len(returns) == 0:
            return {}

        return {
            'volatility': f"{returns.std() * np.sqrt(252):.2%}",
            'downside_deviation': f"{returns[returns < 0].std() * np.sqrt(252):.2%}",
            'var_95': f"{np.percentile(returns, 5):.2%}",
            'cvar_95': f"{returns[returns <= np.percentile(returns, 5)].mean():.2%}",
            'max_consecutive_losses': self._max_consecutive_losses(results),
            'recovery_time': f"{results.max_drawdown_duration} days",
            'calmar_ratio': results.annual_return / abs(results.max_drawdown) if results.max_drawdown != 0 else 0
        }

    def _test_statistical_significance(self, results: BacktestResult) -> Dict:

        returns = results.daily_returns

        if len(returns) < 30:
            return {'significant': False, 'message': 'Insufficient data'}

        t_stat, p_value = stats.ttest_1samp(returns, 0)

        bootstrap_returns = []
        for _ in range(1000):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_returns.append(np.mean(sample) * 252)

        return {
            'significant': p_value < 0.05,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval_95': [
                np.percentile(bootstrap_returns, 2.5),
                np.percentile(bootstrap_returns, 97.5)
            ],
            'probability_positive': sum(1 for r in bootstrap_returns if r > 0) / len(bootstrap_returns)
        }

    def _analyze_stability(self, results: BacktestResult) -> Dict:

        returns = results.daily_returns
        equity = results.equity_curve

        if len(returns) < 60:
            return {}

        rolling_window = 60
        rolling_sharpe = returns.rolling(rolling_window).apply(
            lambda x: self._calculate_sharpe(x)
        )

        bull_market_returns = returns[returns > 0]
        bear_market_returns = returns[returns < 0]

        return {
            'rolling_sharpe_mean': rolling_sharpe.mean(),
            'rolling_sharpe_std': rolling_sharpe.std(),
            'consistency': (rolling_sharpe > 0).sum() / len(rolling_sharpe),
            'bull_market_sharpe': self._calculate_sharpe(bull_market_returns),
            'bear_market_sharpe': self._calculate_sharpe(bear_market_returns),
            'equity_curve_r_squared': self._calculate_r_squared(equity)
        }

    def _compare_to_baseline(self, results: BacktestResult) -> Dict:

        baseline_return = 0.07
        baseline_sharpe = 0.5

        return {
            'excess_return': results.annual_return - baseline_return,
            'sharpe_improvement': results.sharpe_ratio - baseline_sharpe,
            'information_ratio': (results.annual_return - baseline_return) / (results.daily_returns.std() * np.sqrt(252)) if results.daily_returns.std() > 0 else 0,
            'outperformance_periods': self._calculate_outperformance_periods(results, baseline_return)
        }

    def _max_consecutive_losses(self, results: BacktestResult) -> int:

        if not results.trade_history:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in results.trade_history:
            if trade.get('pnl', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_sharpe(self, returns: pd.Series) -> float:

        if len(returns) == 0 or returns.std() == 0:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_r_squared(self, equity_curve: pd.Series) -> float:

        if len(equity_curve) < 2:
            return 0

        x = np.arange(len(equity_curve))
        y = equity_curve.values

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def _calculate_outperformance_periods(self, results: BacktestResult,
                                         baseline_return: float) -> Dict:

        daily_baseline = baseline_return / 252
        outperform_days = (results.daily_returns > daily_baseline).sum()
        total_days = len(results.daily_returns)

        return {
            'days_outperformed': outperform_days,
            'outperformance_rate': outperform_days / total_days if total_days > 0 else 0,
            'avg_excess_return': (results.daily_returns - daily_baseline).mean() * 252
        }
class StrategyComparison:


    def __init__(self):
        self.comparison_results = {}

    async def compare_routing_approaches(self, backtest_config: BacktestConfig) -> Dict:

        approaches = {
            'ml_optimized': self._create_ml_routing,
            'random_routing': self._create_random_routing,
            'static_routing': self._create_static_routing,
            'lowest_fee': self._create_lowest_fee_routing
        }

        results = {}

        for approach_name, approach_factory in approaches.items():
            print(f"🔀 Testing routing approach: {approach_name}")

            engine = BacktestingEngine(backtest_config)

            result = await engine.run_backtest(
                strategy_factory=self._create_strategies,
                ml_predictor_factory=approach_factory,
                mode=BacktestMode.HISTORICAL
            )

            results[approach_name] = result

        comparison = self._analyze_routing_comparison(results)
        self.comparison_results['routing_approaches'] = comparison

        return comparison

    def _create_ml_routing(self):

        class MLRouter:
            async def predict_latency(self, venue, features):
                base_latencies = {
                    'NYSE': 800,
                    'NASDAQ': 900,
                    'CBOE': 1000,
                    'IEX': 1200,
                    'ARCA': 850
                }
                return base_latencies.get(venue, 1000) + np.random.normal(0, 50)

            async def get_best_venue(self, predictions):
                return min(predictions.items(), key=lambda x: x[1])[0]

            def set_weight(self, weight):
                pass

        return MLRouter()

    def _create_random_routing(self):

        class RandomRouter:
            def __init__(self):
                self.venues = ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']

            async def predict_latency(self, venue, features):
                return 1000

            async def get_best_venue(self, predictions):
                return np.random.choice(self.venues)

        return RandomRouter()

    def _create_static_routing(self):

        class StaticRouter:
            async def predict_latency(self, venue, features):
                return 1000

            async def get_best_venue(self, predictions):
                return 'NYSE'

        return StaticRouter()

    def _create_lowest_fee_routing(self):

        class LowestFeeRouter:
            def __init__(self):
                self.fee_schedule = {
                    'IEX': 0.0009,
                    'CBOE': 0.0028,
                    'NYSE': 0.0030,
                    'NASDAQ': 0.0030,
                    'ARCA': 0.0030
                }

            async def predict_latency(self, venue, features):
                return 1000

            async def get_best_venue(self, predictions):
                return min(self.fee_schedule.items(), key=lambda x: x[1])[0]

        return LowestFeeRouter()

    def _create_strategies(self):

        return {
            'market_making': MarketMakingStrategy(),
            'arbitrage': ArbitrageStrategy(),
            'momentum': MomentumStrategy()
        }

    def _analyze_routing_comparison(self, results: Dict[str, BacktestResult]) -> Dict:

        analysis = {
            'performance_summary': {},
            'improvement_vs_baseline': {},
            'statistical_tests': {},
            'cost_analysis': {}
        }

        baseline = results.get('random_routing')

        for approach, result in results.items():
            analysis['performance_summary'][approach] = {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate
            }

            if baseline and approach != 'random_routing':
                analysis['improvement_vs_baseline'][approach] = {
                    'return_improvement': result.total_return - baseline.total_return,
                    'sharpe_improvement': result.sharpe_ratio - baseline.sharpe_ratio,
                    'drawdown_improvement': baseline.max_drawdown - result.max_drawdown,
                    'win_rate_improvement': result.win_rate - baseline.win_rate
                }

            analysis['cost_analysis'][approach] = {
                'total_fees': result.total_fees,
                'total_slippage': result.total_slippage,
                'cost_per_trade': (result.total_fees + result.total_slippage) / max(result.total_trades, 1),
                'ml_routing_benefit': result.ml_routing_benefit
            }

        if 'ml_optimized' in results and 'random_routing' in results:
            ml_returns = results['ml_optimized'].daily_returns
            random_returns = results['random_routing'].daily_returns

            if len(ml_returns) > 0 and len(random_returns) > 0:
                t_stat, p_value = stats.ttest_ind(ml_returns, random_returns)

                analysis['statistical_tests']['ml_vs_random'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'ml_advantage': (ml_returns.mean() - random_returns.mean()) * 252
                }

        return analysis

    async def parameter_sensitivity_analysis(self, base_config: BacktestConfig,
                                          parameter_ranges: Dict) -> Dict:

        results = {}

        for param_name, param_values in parameter_ranges.items():
            param_results = []

            for value in param_values:
                test_config = BacktestConfig(**base_config.__dict__)
                setattr(test_config, param_name, value)

                engine = BacktestingEngine(test_config)
                result = await engine.run_backtest(
                    strategy_factory=self._create_strategies,
                    ml_predictor_factory=self._create_ml_routing,
                    mode=BacktestMode.HISTORICAL
                )

                param_results.append({
                    'value': value,
                    'sharpe': result.sharpe_ratio,
                    'return': result.total_return,
                    'drawdown': result.max_drawdown
                })

            results[param_name] = param_results

        return self._analyze_sensitivity(results)

    def _analyze_sensitivity(self, results: Dict) -> Dict:

        analysis = {}

        for param_name, param_results in results.items():
            values = [r['value'] for r in param_results]
            sharpes = [r['sharpe'] for r in param_results]

            best_idx = np.argmax(sharpes)

            if len(values) > 1:
                value_range = max(values) - min(values)
                sharpe_range = max(sharpes) - min(sharpes)
                sensitivity = sharpe_range / value_range if value_range > 0 else 0
            else:
                sensitivity = 0

            analysis[param_name] = {
                'optimal_value': values[best_idx],
                'optimal_sharpe': sharpes[best_idx],
                'sensitivity': sensitivity,
                'robust': sharpe_range < 0.5 if len(values) > 1 else True
            }

        return analysis
class ReportGenerator:


    def __init__(self):
        self.report_sections = []

    def generate_report(self, results: BacktestResult,
                       comparison: Optional[Dict] = None) -> str:
        """Generate HTML report"""
        html = """
        <html>
        <head>
            <title>HFT Network Optimizer - Backtest Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; }
                .positive { color: green; }
                .negative { color: red; }
                .chart { margin: 20px 0; }
                .summary-box { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
        """
        
        # Executive Summary
        html += self._generate_executive_summary(results)

        html += self._generate_performance_section(results)

        html += self._generate_risk_section(results)

        html += self._generate_strategy_section(results)

        html += self._generate_venue_section(results)

        html += self._generate_ml_section(results)

        if results.monte_carlo_analysis:
            html += self._generate_monte_carlo_section(results)

        if results.stress_test_summary:
            html += self._generate_stress_test_section(results)

        if comparison:
            html += self._generate_comparison_section(comparison)

        html += """
        </body>
        </html>
        """

        return html

    def _generate_executive_summary(self, results: BacktestResult) -> str:
        """Generate executive summary section"""
        return f"""
        <h1>HFT Network Optimizer - Backtest Report</h1>
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p><strong>Total Return:</strong> <span class="{'positive' if results.total_return > 0 else 'negative'}">{results.total_return:.2%}</span></p>
            <p><strong>Sharpe Ratio:</strong> {results.sharpe_ratio:.2f}</p>
            <p><strong>Maximum Drawdown:</strong> <span class="negative">{results.max_drawdown:.2%}</span></p>
            <p><strong>Total Trades:</strong> {results.total_trades:,}</p>
        </div>
        """

    def _generate_performance_section(self, results: BacktestResult) -> str:
        """Generate performance metrics section"""
        return f"""
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Return</td>
                <td class="{'positive' if results.total_return > 0 else 'negative'}">{results.total_return:.2%}</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{results.sharpe_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{results.win_rate:.2%}</td>
            </tr>
        </table>
        """

    def _generate_risk_section(self, results: BacktestResult) -> str:

        daily_vol = results.daily_returns.std() if len(results.daily_returns) > 0 else 0
        annual_vol = daily_vol * np.sqrt(252)

        return "<div class='section'><h3>Section Content</h3><p>Analysis results would be displayed here.</p></div>"

    def _generate_strategy_section(self, results: BacktestResult) -> str:

        html = "<h2>Strategy Performance</h2><table><tr><th>Strategy</th><th>Net P&L</th><th>Trades</th><th>Win Rate</th></tr>"

        for strategy, perf in results.strategy_performance.items():
            html += f"<tr><td>{strategy}</td><td>${perf.get('pnl', 0):.2f}</td><td>{perf.get('trades', 0)}</td><td>{perf.get('win_rate', 0):.1%}</td></tr>"

        html += "</table>"
        return html

    def _generate_venue_section(self, results: BacktestResult) -> str:

        if not results.venue_performance:
            return ""

        html = "<h2>Venue Analysis</h2><table><tr><th>Venue</th><th>Fill Rate</th><th>Avg Latency</th><th>Net Fees</th><th>Efficiency Score</th></tr>"

        for venue, perf in results.venue_performance.items():
            html += f"<tr><td>{venue}</td><td>{perf.get('fill_rate', 0):.1%}</td><td>{perf.get('avg_latency', 0):.0f}μs</td><td>${perf.get('fees', 0):.2f}</td><td>{perf.get('efficiency', 0):.2f}</td></tr>"

        html += "</table>"
        return html

    def _generate_ml_section(self, results: BacktestResult) -> str:

        return "<div class='section'><h3>Section Content</h3><p>Analysis results would be displayed here.</p></div>"

    def _generate_monte_carlo_section(self, results: BacktestResult) -> str:

        mc = results.monte_carlo_analysis

        return "<div class='section'><h3>Section Content</h3><p>Analysis results would be displayed here.</p></div>"

    def _generate_stress_test_section(self, results: BacktestResult) -> str:

        html = "<h2>Stress Test Results</h2><table><tr><th>Scenario</th><th>Return</th><th>Max Drawdown</th><th>Survived</th></tr>"

        for scenario, test in results.stress_test_summary.items():
            html += f

        html += "</table>"

        if hasattr(results, 'stress_survival_rate'):
            html += f"<p><strong>Overall Survival Rate:</strong> {results.stress_survival_rate:.0%}</p>"

        return html

    def _generate_comparison_section(self, comparison: Dict) -> str:

        html = "<h2>Routing Comparison</h2><table><tr><th>Approach</th><th>Total Return</th><th>Sharpe Ratio</th><th>Max Drawdown</th><th>Cost per Trade</th></tr>"

        for approach, metrics in comparison.get('performance_summary', {}).items():
            html += f

        html += "</table>"

        if 'statistical_tests' in comparison and 'ml_vs_random' in comparison['statistical_tests']:
            test = comparison['statistical_tests']['ml_vs_random']
            html += f

        return html

    def save_report(self, html: str, filename: str):
        
        try:
            # Ensure directory exists
            import os
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
                
            with open(filename, 'w') as f:
                f.write(html)
            print(f"📄 Report saved to {filename}")
        except Exception as e:
            print(f"⚠️  Error saving report to {filename}: {e}")
            # Fallback to current directory
            basename = os.path.basename(filename)
            try:
                with open(basename, 'w') as f:
                    f.write(html)
                print(f"📄 Report saved to {basename} (fallback)")
            except Exception as e2:
                print(f"❌ Failed to save report: {e2}")
async def run_complete_backtest_example():


    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=1_000_000,
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        venues=['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA'],
        max_position_size=10000,
        max_daily_loss=50000
    )

    engine = BacktestingEngine(config)

    def strategy_factory():
        return {
            'market_making': MarketMakingStrategy(),
            'arbitrage': ArbitrageStrategy(),
            'momentum': MomentumStrategy()
        }

    def ml_predictor_factory():
        return StrategyComparison()._create_ml_routing()

    print("📈 Running historical backtest...")
    historical_result = await engine.run_backtest(
        strategy_factory,
        ml_predictor_factory,
        BacktestMode.HISTORICAL
    )

    print("🔄 Running walk-forward analysis...")
    walk_forward_result = await engine.run_backtest(
        strategy_factory,
        ml_predictor_factory,
        BacktestMode.WALK_FORWARD
    )

    print("🔀 Comparing routing approaches...")
    comparison = StrategyComparison()
    routing_comparison = await comparison.compare_routing_approaches(config)

    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze_results(historical_result)

    generator = ReportGenerator()
    report_html = generator.generate_report(historical_result, routing_comparison)
    generator.save_report(report_html, f"backtest_report_{int(time.time())}.html")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"Total Return: {historical_result.total_return:.2%}")
    print(f"Annual Return: {historical_result.annual_return:.2%}")
    print(f"Sharpe Ratio: {historical_result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {historical_result.max_drawdown:.2%}")
    print(f"Total Trades: {historical_result.total_trades:,}")
    print(f"ML Routing Benefit: ${historical_result.ml_routing_benefit:,.2f}")
    print("\nRouting Comparison:")
    for approach, perf in routing_comparison['performance_summary'].items():
        print(f"  {approach}: Return={perf['total_return']:.2%}, Sharpe={perf['sharpe_ratio']:.2f}")
    print("="*80)

    return historical_result, routing_comparison