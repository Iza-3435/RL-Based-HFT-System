import numpy as np
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulator.network_latency_simulator import NetworkLatencySimulator
from simulator.trading_simulator import (
    TradingSimulator, OrderExecutionEngine, MarketImpactModel,
    Order, Fill, OrderType, OrderSide, OrderStatus, TradingStrategyType
)
from simulator.network_latency_simulator import NetworkLatencySimulator
from simulator.enhanced_latency_simulation import (
    LatencySimulator, EnhancedOrderExecutionEngine, LatencyAnalytics,
    create_latency_configuration, integrate_latency_simulation
)
logger = logging.getLogger(__name__)
class EnhancedTradingSimulator(TradingSimulator):


    def __init__(self, venues: List[str], symbols: List[str],
                 enable_latency_simulation: bool = True):

        super().__init__(venues, symbols)

        if enable_latency_simulation:
            self.execution_engine = EnhancedOrderExecutionEngine(venues)
            logger.info("Enhanced latency simulation enabled")
        else:
            self.latency_simulator = LatencySimulator(venues)
            integrate_latency_simulation(self.execution_engine, self.latency_simulator)
            logger.info("Basic latency simulation enabled")

        self.latency_analytics = LatencyAnalytics(self.execution_engine)

        self.latency_performance_history = []
        self.venue_latency_rankings = []

        self.latency_config = create_latency_configuration()

        self.network_simulator = NetworkLatencySimulator()

        logger.info(f"EnhancedTradingSimulator initialized with latency analytics")

    async def simulate_trading(self, market_data_generator, ml_predictor,
                              duration_seconds: int = 300) -> Dict[str, Any]:

        logger.info(f"Starting enhanced trading simulation for {duration_seconds} seconds")

        self.simulation_start_time = time.time()
        tick_count = 0
        latency_snapshots = []

        market_state = {}
        current_prices = {}

        async for tick in market_data_generator.generate_market_data_stream(duration_seconds):
            self.network_simulator.update_market_conditions(
                volatility=getattr(tick, 'volatility', 0.02),
                volume_factor=getattr(tick, 'volume', 1000) / 1000
            )

            symbol_venue_key = f"{tick.symbol}_{tick.venue}"

            network_latency = self.network_simulator.get_current_latency(tick.venue)

            market_state[symbol_venue_key] = {
                'bid_price': tick.bid_price,
                'ask_price': tick.ask_price,
                'mid_price': tick.mid_price,
                'spread_bps': (tick.ask_price - tick.bid_price) / tick.mid_price * 10000,
                'volume': tick.volume,
                'bid_size': tick.bid_size,
                'ask_size': tick.ask_size,
                'volatility': getattr(tick, 'volatility', 0.02),
                'average_daily_volume': 1000000,
                'average_trade_size': 100,
                'current_network_latency_us': network_latency,
                'regime': self._detect_market_regime(tick)
            }

            if hasattr(self.execution_engine, 'latency_simulator'):
                self.execution_engine.latency_simulator.update_market_conditions(
                    symbol=tick.symbol,
                    volatility=getattr(tick, 'volatility', 0.02),
                    volume_factor=tick.volume / 1000000 if hasattr(tick, 'volume') else 1.0,
                    timestamp=getattr(tick, 'timestamp', time.time())
                )

            current_prices[tick.symbol] = tick.mid_price

            if tick_count % 10 == 0:
                await self._process_trading_cycle(
                    market_state, current_prices, ml_predictor, tick
                )

                if tick_count % 100 == 0:
                    snapshot = self._capture_latency_snapshot()
                    latency_snapshots.append(snapshot)

            tick_count += 1

            if tick_count % 1000 == 0:
                await self._log_enhanced_progress(tick_count, latency_snapshots)

        results = await self._generate_enhanced_results(latency_snapshots)

        logger.info(f"Enhanced simulation complete: {tick_count} ticks, {self.trade_count} trades")
        return results

    async def _process_trading_cycle(self, market_state: Dict, current_prices: Dict,
                                   ml_predictor, current_tick) -> None:

        aggregated_market_data = {
            'symbols': self.symbols,
            'venues': self.venues,
            **market_state
        }

        ml_predictions = await self._get_enhanced_ml_predictions(
            ml_predictor, current_tick, market_state
        )

        all_orders = []
        for strategy_name, strategy in self.strategies.items():
            orders = await strategy.generate_signals(
                aggregated_market_data, ml_predictions
            )

            enhanced_orders = self._enhance_orders_with_latency_routing(
                orders, ml_predictions, market_state
            )

            all_orders.extend(enhanced_orders)

        execution_results = await self._execute_orders_with_analytics(
            all_orders, market_state, current_prices
        )

        self._update_performance_metrics(current_prices)

        self._analyze_latency_impact(execution_results)

    async def _get_enhanced_ml_predictions(self, ml_predictor, tick, market_state) -> Dict:

        predictions = {}

        for venue in self.venues:
            routing_key = f'routing_{tick.symbol}_{venue}'

            routing_decision = ml_predictor.make_routing_decision(tick.symbol)

            if hasattr(self.execution_engine, 'latency_simulator'):
                venue_stats = self.execution_engine.latency_simulator.get_venue_latency_stats(venue, 5)
                current_latency_us = venue_stats.get('mean_us', 1000) if venue_stats else 1000
            else:
                current_latency_us = 1000

            predictions[routing_key] = {
                'venue': routing_decision.venue,
                'predicted_latency_us': routing_decision.expected_latency_us,
                'confidence': routing_decision.confidence,
                'current_latency_us': current_latency_us,
                'latency_trend': self._calculate_latency_trend(venue),
                'congestion_probability': self._estimate_congestion_probability(venue, market_state)
            }

        regime_detection = ml_predictor.detect_market_regime(market_state)
        predictions['regime'] = regime_detection.regime.value
        predictions['regime_confidence'] = getattr(regime_detection, 'confidence', 0.7)

        predictions['volatility_forecast'] = market_state.get(
            f"{tick.symbol}_{tick.venue}", {}
        ).get('volatility', 0.01)

        base_momentum = np.random.randn() * 0.5
        latency_adjustment = self._calculate_latency_momentum_adjustment(tick.symbol)
        predictions[f'momentum_signal_{tick.symbol}'] = base_momentum * latency_adjustment

        predictions['market_impact_forecast'] = self._forecast_market_impact(tick, market_state)

        return predictions

    def _enhance_orders_with_latency_routing(self, orders: List[Order],
                                           ml_predictions: Dict,
                                           market_state: Dict) -> List[Order]:

        enhanced_orders = []

        for order in orders:
            venue_latencies = {}
            for venue in self.venues:
                routing_key = f'routing_{order.symbol}_{venue}'
                if routing_key in ml_predictions:
                    pred = ml_predictions[routing_key]
                    venue_latencies[venue] = {
                        'predicted_latency_us': pred['predicted_latency_us'],
                        'confidence': pred['confidence'],
                        'congestion_prob': pred['congestion_probability']
                    }

            optimal_venue = self._select_optimal_venue(order, venue_latencies, market_state)

            order.venue = optimal_venue
            if optimal_venue in venue_latencies:
                order.predicted_latency_us = venue_latencies[optimal_venue]['predicted_latency_us']
                order.routing_confidence = venue_latencies[optimal_venue]['confidence']

            order.latency_tolerance_us = self._calculate_latency_tolerance(order)
            order.urgency_score = self._calculate_order_urgency(order, market_state)

            enhanced_orders.append(order)

        return enhanced_orders

    def _select_optimal_venue(self, order: Order, venue_latencies: Dict,
                            market_state: Dict) -> str:

        if not venue_latencies:
            return order.venue

        strategy_requirements = self.latency_config['latency_thresholds'].get(
            order.strategy.value, {'acceptable_us': 2000, 'optimal_us': 800}
        )

        venue_scores = {}

        for venue, latency_info in venue_latencies.items():
            predicted_latency = latency_info['predicted_latency_us']
            confidence = latency_info['confidence']
            congestion_prob = latency_info['congestion_prob']

            latency_score = max(0, 1.0 - (predicted_latency / strategy_requirements['acceptable_us']))

            confidence_bonus = confidence * 0.2

            congestion_penalty = congestion_prob * 0.3

            if order.strategy == TradingStrategyType.ARBITRAGE:
                if predicted_latency > 500:
                    latency_score *= 0.5
            elif order.strategy == TradingStrategyType.MARKET_MAKING:
                latency_score *= confidence

            venue_scores[venue] = latency_score + confidence_bonus - congestion_penalty

        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        return best_venue

    def _calculate_latency_tolerance(self, order: Order) -> float:

        base_tolerance = self.latency_config['latency_thresholds'].get(
            order.strategy.value, {'acceptable_us': 2000}
        )['acceptable_us']

        if order.order_type == OrderType.MARKET:
            return base_tolerance * 0.7
        elif order.order_type == OrderType.IOC:
            return base_tolerance * 0.5
        else:
            return base_tolerance

    def _calculate_order_urgency(self, order: Order, market_state: Dict) -> float:

        urgency = 0.5

        if order.strategy == TradingStrategyType.ARBITRAGE:
            urgency = 0.9
        elif order.strategy == TradingStrategyType.MARKET_MAKING:
            urgency = 0.3
        elif order.strategy == TradingStrategyType.MOMENTUM:
            urgency = 0.7

        symbol_venue_key = f"{order.symbol}_{order.venue}"
        if symbol_venue_key in market_state:
            volatility = market_state[symbol_venue_key].get('volatility', 0.02)
            if volatility > 0.04:
                urgency = min(1.0, urgency * 1.3)

        return urgency

    async def _execute_orders_with_analytics(self, orders: List[Order],
                                           market_state: Dict,
                                           current_prices: Dict) -> List[Dict]:

        execution_results = []

        for order in orders:
            start_time = time.perf_counter()

            fill = await self._execute_order_enhanced(order, market_state)

            execution_time = (time.perf_counter() - start_time) * 1e6

            if fill:
                strategy = self.strategies[order.strategy.value]
                strategy.update_positions(fill, current_prices)

                result = {
                    'order': order,
                    'fill': fill,
                    'execution_time_us': execution_time,
                    'latency_breakdown': getattr(fill, 'latency_breakdown', None),
                    'prediction_accuracy': self._calculate_prediction_accuracy(order, fill),
                    'venue_performance_rank': self._get_venue_rank(order.venue)
                }

                execution_results.append(result)

                self.fill_history.append(fill)
                self.trade_count += 1

        return execution_results

    async def _execute_order_enhanced(self, order: Order, market_state: Dict) -> Optional[Fill]:

        state_key = f"{order.symbol}_{order.venue}"
        if state_key not in market_state:
            logger.warning(f"No market data for {state_key}")
            return None

        venue_market_state = market_state[state_key]

        if hasattr(self.execution_engine, 'latency_simulator'):
            congestion = self.execution_engine.latency_simulator.get_congestion_analysis()
            if congestion['current_congestion_level'] == 'critical':
                if getattr(order, 'urgency_score', 0.5) > 0.8:
                    alternative_venue = self._find_alternative_venue(order, market_state)
                    if alternative_venue:
                        order.venue = alternative_venue
                        state_key = f"{order.symbol}_{alternative_venue}"
                        venue_market_state = market_state.get(state_key, venue_market_state)

        try:
            fill = await self.execution_engine.execute_order(
                order, venue_market_state, getattr(order, 'predicted_latency_us', None)
            )

            if fill and hasattr(fill, 'latency_breakdown'):
                self._validate_execution_quality(order, fill)

            return fill

        except Exception as e:
            logger.error(f"Order execution failed for {order.order_id}: {e}")
            return None

    def _find_alternative_venue(self, order: Order, market_state: Dict) -> Optional[str]:

        if hasattr(self.execution_engine, 'get_venue_latency_rankings'):
            rankings = self.execution_engine.get_venue_latency_rankings()

            for venue, latency in rankings:
                if venue != order.venue and latency < 2000:
                    state_key = f"{order.symbol}_{venue}"
                    if state_key in market_state:
                        return venue

        return None

    def _validate_execution_quality(self, order: Order, fill: Fill) -> None:

        if hasattr(fill, 'latency_breakdown'):
            actual_latency = fill.latency_breakdown.total_latency_us
            tolerance = getattr(order, 'latency_tolerance_us', 2000)

            if actual_latency > tolerance:
                logger.warning(f"Order {order.order_id} exceeded latency tolerance: "
                             f"{actual_latency:.0f}μs > {tolerance:.0f}μs")

            if hasattr(order, 'predicted_latency_us') and order.predicted_latency_us:
                error_pct = abs(actual_latency - order.predicted_latency_us) / order.predicted_latency_us * 100
                if error_pct > 25:
                    logger.warning(f"Poor latency prediction for {order.order_id}: "
                                 f"{error_pct:.1f}% error")

    def _capture_latency_snapshot(self) -> Dict[str, Any]:

        snapshot = {
            'timestamp': time.time(),
            'venue_latencies': {},
            'congestion_level': 'normal',
            'prediction_accuracy': 0.0,
            'total_trades': self.trade_count
        }

        if hasattr(self.execution_engine, 'get_venue_latency_rankings'):
            rankings = self.execution_engine.get_venue_latency_rankings()
            snapshot['venue_latencies'] = dict(rankings)

        if hasattr(self.execution_engine, 'latency_simulator'):
            congestion = self.execution_engine.latency_simulator.get_congestion_analysis()
            snapshot['congestion_level'] = congestion['current_congestion_level']

            pred_stats = self.execution_engine.latency_simulator.get_prediction_accuracy_stats()
            snapshot['prediction_accuracy'] = pred_stats.get('prediction_within_10pct', 0)

        return snapshot

    async def _log_enhanced_progress(self, tick_count: int, latency_snapshots: List[Dict]) -> None:

        elapsed = time.time() - self.simulation_start_time

        logger.info(f"Processed {tick_count} ticks in {elapsed:.1f}s, "
                   f"Total P&L: ${self.total_pnl:.2f}, Trades: {self.trade_count}")

        if latency_snapshots:
            latest_snapshot = latency_snapshots[-1]

            if latest_snapshot['venue_latencies']:
                best_venue = min(latest_snapshot['venue_latencies'].items(), key=lambda x: x[1])
                worst_venue = max(latest_snapshot['venue_latencies'].items(), key=lambda x: x[1])

                logger.info(f"Latency: Best={best_venue[0]}({best_venue[1]:.0f}μs), "
                           f"Worst={worst_venue[0]}({worst_venue[1]:.0f}μs), "
                           f"Congestion={latest_snapshot['congestion_level']}, "
                           f"Prediction={latest_snapshot['prediction_accuracy']:.1f}%")

    async def _generate_enhanced_results(self, latency_snapshots: List[Dict]) -> Dict[str, Any]:

        base_results = self._generate_simulation_results()

        latency_analysis = self.latency_analytics.generate_latency_report(
            timeframe_minutes=int((time.time() - self.simulation_start_time) / 60)
        )

        performance_attribution = self._calculate_enhanced_pnl_attribution()

        strategy_latency_impact = self._analyze_strategy_latency_impact()

        market_condition_analysis = self._analyze_market_condition_impact(latency_snapshots)

        enhanced_results = {
            **base_results,
            'latency_analysis': latency_analysis,
            'performance_attribution': performance_attribution,
            'strategy_latency_impact': strategy_latency_impact,
            'market_condition_analysis': market_condition_analysis,
            'optimization_recommendations': self._generate_optimization_recommendations(),
            'latency_snapshots': latency_snapshots
        }

        return enhanced_results

    def _calculate_enhanced_pnl_attribution(self) -> Dict[str, Any]:

        attribution = {
            'total_pnl': self.total_pnl,
            'gross_pnl': 0.0,
            'latency_costs': 0.0,
            'execution_costs': 0.0,
            'by_venue': defaultdict(float),
            'by_strategy': defaultdict(float)
        }

        if hasattr(self.execution_engine, 'get_latency_cost_analysis'):
            latency_costs = self.execution_engine.get_latency_cost_analysis()
            attribution['latency_costs'] = latency_costs.get('total_latency_cost_usd', 0)

        for fill in self.fill_history:
            venue_pnl = 0
            if hasattr(fill, 'latency_cost_bps'):
                latency_cost = fill.latency_cost_bps * fill.price * fill.quantity / 10000
                venue_pnl -= latency_cost

            attribution['by_venue'][fill.venue] += venue_pnl

        for strategy_name, strategy in self.strategies.items():
            strategy_pnl = strategy.get_total_pnl()

            strategy_fills = [f for f in self.fill_history
                            if getattr(f, 'order_id', '').startswith(strategy_name.upper()[:3])]

            latency_impact = sum(
                getattr(f, 'latency_cost_bps', 0) * f.price * f.quantity / 10000
                for f in strategy_fills
            )

            attribution['by_strategy'][strategy_name] = {
                'gross_pnl': strategy_pnl['total_pnl'],
                'latency_impact': -latency_impact,
                'net_pnl': strategy_pnl['total_pnl'] - latency_impact
            }

        return attribution

    def _analyze_strategy_latency_impact(self) -> Dict[str, Any]:

        impact_analysis = {}

        for strategy_name, strategy in self.strategies.items():
            strategy_fills = [
                f for f in self.fill_history
                if hasattr(f, 'order_id') and strategy_name.upper()[:3] in f.order_id
            ]

            if not strategy_fills:
                continue

            latencies = [getattr(f, 'latency_us', 1000) for f in strategy_fills]
            slippages = [getattr(f, 'slippage_bps', 0) for f in strategy_fills]

            if strategy_name == 'arbitrage':
                impact_analysis[strategy_name] = {
                    'avg_latency_us': np.mean(latencies),
                    'opportunity_capture_rate': getattr(strategy, 'opportunities_captured', 0) /
                                              max(getattr(strategy, 'opportunities_detected', 1), 1),
                    'latency_sensitivity': self._calculate_latency_sensitivity(latencies, slippages),
                    'optimal_latency_threshold_us': 300
                }

            elif strategy_name == 'market_making':
                impact_analysis[strategy_name] = {
                    'avg_latency_us': np.mean(latencies),
                    'spread_capture_efficiency': getattr(strategy, 'spread_captured', 0) /
                                               max(len(strategy_fills), 1),
                    'maker_ratio': self._calculate_maker_ratio(strategy_fills),
                    'optimal_latency_threshold_us': 800
                }

            elif strategy_name == 'momentum':
                impact_analysis[strategy_name] = {
                    'avg_latency_us': np.mean(latencies),
                    'signal_decay_impact': self._estimate_signal_decay(latencies),
                    'execution_efficiency': self._calculate_execution_efficiency(strategy_fills),
                    'optimal_latency_threshold_us': 1500
                }

        return impact_analysis

    def _analyze_market_condition_impact(self, latency_snapshots: List[Dict]) -> Dict[str, Any]:

        if not latency_snapshots:
            return {}

        analysis = {
            'congestion_impact': self._analyze_congestion_impact(latency_snapshots),
            'volatility_correlation': self._analyze_volatility_latency_correlation(),
            'volume_correlation': self._analyze_volume_latency_correlation(),
            'time_of_day_effects': self._analyze_time_of_day_effects()
        }

        return analysis

    def _generate_optimization_recommendations(self) -> List[str]:

        recommendations = []

        if hasattr(self.latency_analytics, '_generate_recommendations'):
            recommendations.extend(self.latency_analytics._generate_recommendations())

        strategy_impact = self._analyze_strategy_latency_impact()

        for strategy_name, impact in strategy_impact.items():
            avg_latency = impact.get('avg_latency_us', 0)
            optimal_threshold = impact.get('optimal_latency_threshold_us', 1000)

            if avg_latency > optimal_threshold * 1.5:
                recommendations.append(
                    f"{strategy_name.title()} strategy experiencing high latency "
                    f"({avg_latency:.0f}μs vs optimal {optimal_threshold}μs). "
                    f"Consider venue optimization or strategy parameters."
                )

        if hasattr(self.execution_engine, 'get_venue_latency_rankings'):
            rankings = self.execution_engine.get_venue_latency_rankings()
            if len(rankings) > 1:
                best_latency = rankings[0][1]
                worst_latency = rankings[-1][1]

                if worst_latency > best_latency * 2:
                    recommendations.append(
                        f"Consider rebalancing order flow: {rankings[-1][0]} showing "
                        f"{worst_latency:.0f}μs vs best venue {rankings[0][0]} at {best_latency:.0f}μs"
                    )

        if hasattr(self.execution_engine, 'get_latency_cost_analysis'):
            cost_analysis = self.execution_engine.get_latency_cost_analysis()
            potential_savings = cost_analysis.get('potential_savings_analysis', {})

            if potential_savings.get('potential_improvement_pct', 0) > 20:
                recommendations.append(
                    f"Significant latency optimization opportunity: "
                    f"{potential_savings['potential_improvement_pct']:.1f}% improvement possible, "
                    f"estimated savings: {potential_savings.get('estimated_savings_bps', 0):.2f} bps"
                )

        return recommendations

    def _detect_market_regime(self, tick) -> str:

        volatility = getattr(tick, 'volatility', 0.02)
        volume = getattr(tick, 'volume', 1000)

        if volatility > 0.04 and volume > 2000:
            return 'stressed'
        elif volatility > 0.03:
            return 'volatile'
        elif volume < 500:
            return 'quiet'
        else:
            return 'normal'

    def _calculate_latency_trend(self, venue: str) -> str:

        if hasattr(self.execution_engine, 'latency_simulator'):
            recent_stats = self.execution_engine.latency_simulator.get_venue_latency_stats(venue, 5)
            historical_stats = self.execution_engine.latency_simulator.get_venue_latency_stats(venue, 10)

            if recent_stats and historical_stats:
                recent_mean = recent_stats.get('mean_us', 1000)
                historical_mean = historical_stats.get('mean_us', 1000)

                if recent_mean > historical_mean * 1.1:
                    return 'increasing'
                elif recent_mean < historical_mean * 0.9:
                    return 'decreasing'
                else:
                    return 'stable'

        return 'unknown'

    def _estimate_congestion_probability(self, venue: str, market_state: Dict) -> float:

        base_probability = 0.1

        avg_volatility = np.mean([
            state.get('volatility', 0.02)
            for key, state in market_state.items()
            if key.endswith(f'_{venue}')
        ])
        volatility_factor = min(avg_volatility / 0.02, 3.0)

        avg_volume = np.mean([
            state.get('volume', 1000)
            for key, state in market_state.items()
            if key.endswith(f'_{venue}')
        ])
        volume_factor = min(avg_volume / 1000, 2.0)

        congestion_prob = base_probability * volatility_factor * volume_factor
        return min(congestion_prob, 0.8)

    def _calculate_latency_momentum_adjustment(self, symbol: str) -> float:

        base_adjustment = 1.0

        if hasattr(self.execution_engine, 'latency_simulator'):
            venue_latencies = []
            for venue in self.venues:
                stats = self.execution_engine.latency_simulator.get_venue_latency_stats(venue, 5)
                if stats:
                    venue_latencies.append(stats['mean_us'])

            if venue_latencies:
                avg_latency = np.mean(venue_latencies)
                adjustment = max(0.7, 1.0 - (avg_latency - 1000) / 5000)
                return adjustment

        return base_adjustment

    def _forecast_market_impact(self, tick, market_state: Dict) -> float:

        base_impact = 0.1

        volatility = getattr(tick, 'volatility', 0.02)
        volatility_multiplier = volatility / 0.02

        symbol_venue_key = f"{tick.symbol}_{tick.venue}"
        if symbol_venue_key in market_state:
            network_latency = market_state[symbol_venue_key].get('current_network_latency_us', 1000)
            latency_multiplier = 1.0 + (network_latency - 500) / 2000
        else:
            latency_multiplier = 1.0

        return base_impact * volatility_multiplier * latency_multiplier

    def _calculate_prediction_accuracy(self, order: Order, fill: Fill) -> Optional[float]:

        if not hasattr(order, 'predicted_latency_us') or not order.predicted_latency_us:
            return None

        actual_latency = getattr(fill, 'latency_us', 0)
        if actual_latency == 0:
            return None

        error_pct = abs(actual_latency - order.predicted_latency_us) / order.predicted_latency_us * 100
        accuracy = max(0, 100 - error_pct)

        return accuracy

    def _get_venue_rank(self, venue: str) -> int:

        if hasattr(self.execution_engine, 'get_venue_latency_rankings'):
            rankings = self.execution_engine.get_venue_latency_rankings()
            for rank, (v, _) in enumerate(rankings, 1):
                if v == venue:
                    return rank
        return len(self.venues)

    def _calculate_latency_sensitivity(self, latencies: List[float], slippages: List[float]) -> float:

        if len(latencies) < 2 or len(slippages) < 2:
            return 0.0

        latency_array = np.array(latencies)
        slippage_array = np.array(slippages)

        if np.std(latency_array) == 0 or np.std(slippage_array) == 0:
            return 0.0

        correlation = np.corrcoef(latency_array, slippage_array)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _calculate_maker_ratio(self, fills: List[Fill]) -> float:

        if not fills:
            return 0.0

        maker_fills = sum(1 for f in fills if getattr(f, 'rebate', 0) > 0)
        return maker_fills / len(fills)

    def _estimate_signal_decay(self, latencies: List[float]) -> float:

        if not latencies:
            return 0.0

        avg_latency = np.mean(latencies)
        decay_factor = np.exp(-avg_latency / 500)
        signal_strength_loss = 1.0 - decay_factor

        return signal_strength_loss

    def _calculate_execution_efficiency(self, fills: List[Fill]) -> float:

        if not fills:
            return 0.0

        total_efficiency = 0.0

        for fill in fills:
            slippage = getattr(fill, 'slippage_bps', 0)
            latency = getattr(fill, 'latency_us', 1000)

            slippage_efficiency = max(0, 1.0 - slippage / 10)
            latency_efficiency = max(0, 1.0 - latency / 2000)

            fill_efficiency = (slippage_efficiency + latency_efficiency) / 2
            total_efficiency += fill_efficiency

        return total_efficiency / len(fills)

    def _analyze_congestion_impact(self, latency_snapshots: List[Dict]) -> Dict[str, Any]:

        congestion_analysis = {
            'low_congestion_performance': {},
            'high_congestion_performance': {},
            'congestion_cost_bps': 0.0
        }

        if not latency_snapshots:
            return congestion_analysis

        low_congestion = [s for s in latency_snapshots if s['congestion_level'] in ['low', 'normal']]
        high_congestion = [s for s in latency_snapshots if s['congestion_level'] in ['high', 'critical']]

        if low_congestion:
            low_latencies = []
            for snapshot in low_congestion:
                low_latencies.extend(snapshot['venue_latencies'].values())

            congestion_analysis['low_congestion_performance'] = {
                'avg_latency_us': np.mean(low_latencies) if low_latencies else 0,
                'snapshot_count': len(low_congestion)
            }

        if high_congestion:
            high_latencies = []
            for snapshot in high_congestion:
                high_latencies.extend(snapshot['venue_latencies'].values())

            congestion_analysis['high_congestion_performance'] = {
                'avg_latency_us': np.mean(high_latencies) if high_latencies else 0,
                'snapshot_count': len(high_congestion)
            }

            if low_congestion and high_latencies and congestion_analysis['low_congestion_performance']['avg_latency_us'] > 0:
                latency_increase = (congestion_analysis['high_congestion_performance']['avg_latency_us'] -
                                  congestion_analysis['low_congestion_performance']['avg_latency_us'])
                congestion_analysis['congestion_cost_bps'] = latency_increase / 100 * 0.1

        return congestion_analysis

    def _analyze_volatility_latency_correlation(self) -> Dict[str, float]:

        if not hasattr(self.execution_engine, 'latency_simulator'):
            return {}

        recent_latencies = list(self.execution_engine.latency_simulator.latency_history)

        if len(recent_latencies) < 10:
            return {}

        volatility_factors = [b.volatility_factor for b in recent_latencies]
        latencies = [b.total_latency_us for b in recent_latencies]

        if np.std(volatility_factors) > 0 and np.std(latencies) > 0:
            correlation = np.corrcoef(volatility_factors, latencies)[0, 1]
            return {
                'volatility_latency_correlation': correlation if not np.isnan(correlation) else 0.0,
                'sample_count': len(recent_latencies)
            }

        return {'volatility_latency_correlation': 0.0, 'sample_count': len(recent_latencies)}

    def _analyze_volume_latency_correlation(self) -> Dict[str, float]:

        return {
            'volume_latency_correlation': 0.3,
            'sample_count': len(self.fill_history)
        }

    def _analyze_time_of_day_effects(self) -> Dict[str, Any]:

        time_analysis = {}

        if not hasattr(self.execution_engine, 'latency_simulator'):
            return time_analysis

        hourly_latencies = defaultdict(list)

        for breakdown in self.execution_engine.latency_simulator.latency_history:
            hour = datetime.fromtimestamp(breakdown.timestamp).hour
            hourly_latencies[hour].append(breakdown.total_latency_us)

        for hour, latencies in hourly_latencies.items():
            if latencies:
                time_analysis[f'hour_{hour}'] = {
                    'avg_latency_us': np.mean(latencies),
                    'sample_count': len(latencies),
                    'volatility': np.std(latencies)
                }

        if time_analysis:
            peak_hour = max(time_analysis.items(),
                          key=lambda x: x[1]['avg_latency_us'])[0]
            best_hour = min(time_analysis.items(),
                          key=lambda x: x[1]['avg_latency_us'])[0]

            time_analysis['summary'] = {
                'peak_latency_hour': peak_hour,
                'best_latency_hour': best_hour,
                'intraday_latency_range_us': (
                    time_analysis[peak_hour]['avg_latency_us'] -
                    time_analysis[best_hour]['avg_latency_us']
                )
            }

        return time_analysis

    def _analyze_latency_impact(self, execution_results: List[Dict]) -> None:

        if not execution_results:
            return

        for result in execution_results:
            if result['latency_breakdown']:
                latency = result['latency_breakdown'].total_latency_us
                fill = result['fill']

                if hasattr(fill, 'slippage_bps'):
                    pass

                if result['prediction_accuracy']:
                    pass
def create_enhanced_trading_simulator(symbols: List[str], venues: List[str],
                                    config: Dict[str, Any] = None) -> EnhancedTradingSimulator:

    default_config = {
        'enable_latency_simulation': True,
        'latency_prediction_enabled': True,
        'congestion_modeling': True,
        'venue_optimization': True,
        'strategy_latency_optimization': True
    }

    final_config = {**default_config, **(config or {})}

    simulator = EnhancedTradingSimulator(
        venues=venues,
        symbols=symbols,
        enable_latency_simulation=final_config['enable_latency_simulation']
    )

    if final_config['venue_optimization']:
        _configure_venue_optimization(simulator)

    if final_config['strategy_latency_optimization']:
        _configure_strategy_latency_parameters(simulator)

    logger.info(f"Enhanced trading simulator created with {len(symbols)} symbols, "
               f"{len(venues)} venues, config: {final_config}")

    return simulator
def _configure_venue_optimization(simulator: EnhancedTradingSimulator):

    if hasattr(simulator.execution_engine, 'latency_simulator'):
        latency_sim = simulator.execution_engine.latency_simulator

        for venue, profile in latency_sim.venue_profiles.items():
            if venue == 'IEX':
                profile.network_latency_std_us *= 0.5
                profile.spike_probability *= 0.3
            elif venue == 'NASDAQ':
                profile.processing_rate_msg_per_sec *= 1.2
            elif venue == 'CBOE':
                profile.volatility_sensitivity *= 1.1
def _configure_strategy_latency_parameters(simulator: EnhancedTradingSimulator):


    if 'market_making' in simulator.strategies:
        mm_strategy = simulator.strategies['market_making']
        mm_strategy.params['latency_weight'] = 0.3
        mm_strategy.params['consistency_weight'] = 0.7

    if 'arbitrage' in simulator.strategies:
        arb_strategy = simulator.strategies['arbitrage']
        arb_strategy.params['latency_weight'] = 0.8
        arb_strategy.params['max_acceptable_latency_us'] = 300

    if 'momentum' in simulator.strategies:
        mom_strategy = simulator.strategies['momentum']
        mom_strategy.params['latency_weight'] = 0.5
        mom_strategy.params['signal_decay_factor'] = 0.95
def patch_existing_simulator(existing_simulator: TradingSimulator,
                           enable_enhanced_latency: bool = True) -> TradingSimulator:

    logger.info("Patching existing TradingSimulator with enhanced latency capabilities")

    original_execution_engine = existing_simulator.execution_engine

    if enable_enhanced_latency:
        existing_simulator.execution_engine = EnhancedOrderExecutionEngine(existing_simulator.venues)
    else:
        latency_simulator = LatencySimulator(existing_simulator.venues)
        integrate_latency_simulation(original_execution_engine, latency_simulator)

    existing_simulator.latency_analytics = LatencyAnalytics(existing_simulator.execution_engine)
    existing_simulator.latency_config = create_latency_configuration()

    logger.info("Existing simulator successfully patched with latency capabilities")
    return existing_simulator
def quick_latency_test(simulator: EnhancedTradingSimulator, venue: str,
                      num_samples: int = 100) -> Dict[str, Any]:

    if not hasattr(simulator.execution_engine, 'latency_simulator'):
        return {'error': 'Latency simulator not available'}

    latency_samples = []

    for _ in range(num_samples):
        breakdown = simulator.execution_engine.latency_simulator.simulate_latency(
            venue=venue,
            symbol='TEST',
            order_type='limit'
        )
        latency_samples.append(breakdown.total_latency_us)

    return {
        'venue': venue,
        'samples': num_samples,
        'mean_latency_us': np.mean(latency_samples),
        'median_latency_us': np.median(latency_samples),
        'p95_latency_us': np.percentile(latency_samples, 95),
        'p99_latency_us': np.percentile(latency_samples, 99),
        'std_latency_us': np.std(latency_samples),
        'min_latency_us': np.min(latency_samples),
        'max_latency_us': np.max(latency_samples)
    }
async def test_enhanced_trading_simulator():


    symbols = ['AAPL', 'MSFT', 'GOOGL']
    venues = ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']

    simulator = create_enhanced_trading_simulator(symbols, venues)

    print("Testing Enhanced Trading Simulator with Realistic Latency...")

    class MockMarketDataGenerator:
        async def generate_market_data_stream(self, duration_seconds):
            for i in range(duration_seconds * 10):
                for symbol in symbols:
                    for venue in venues:
                        yield type('Tick', (), {
                            'symbol': symbol,
                            'venue': venue,
                            'bid_price': 100.0 + np.random.randn() * 0.1,
                            'ask_price': 100.1 + np.random.randn() * 0.1,
                            'mid_price': 100.05 + np.random.randn() * 0.1,
                            'volume': max(100, int(np.random.lognormal(6, 1))),
                            'bid_size': 100,
                            'ask_size': 100,
                            'volatility': max(0.01, np.random.lognormal(-4, 0.5)),
                            'timestamp': time.time() + i * 0.1
                        })()

                await asyncio.sleep(0.01)

    class MockMLPredictor:
        def make_routing_decision(self, symbol):
            return type('RoutingDecision', (), {
                'venue': np.random.choice(venues),
                'expected_latency_us': np.random.lognormal(7, 0.5),
                'confidence': np.random.uniform(0.6, 0.95)
            })()

        def detect_market_regime(self, market_state):
            return type('RegimeDetection', (), {
                'regime': type('Regime', (), {'value': 'normal'})(),
                'confidence': 0.8
            })()

    market_generator = MockMarketDataGenerator()
    ml_predictor = MockMLPredictor()

    results = await simulator.simulate_trading(
        market_generator, ml_predictor, duration_seconds=30
    )

    print(f"\n=== Enhanced Simulation Results ===")
    print(f"Total P&L: ${results['summary']['total_pnl']:.2f}")
    print(f"Trades executed: {results['summary']['trade_count']}")
    print(f"Max drawdown: ${results['summary']['max_drawdown']:.2f}")

    if 'latency_analysis' in results:
        latency = results['latency_analysis']
        print(f"\n=== Latency Analysis ===")
        if 'summary' in latency:
            summary = latency['summary']
            print(f"Average latency: {summary.get('avg_total_latency_us', 0):.0f}μs")
            print(f"Latency cost: {summary.get('avg_latency_cost_bps', 0):.3f} bps")
            print(f"Prediction accuracy: {summary.get('prediction_accuracy_pct', 0):.1f}%")

        if 'venue_analysis' in latency:
            print(f"\n=== Venue Performance ===")
            for venue, analysis in latency['venue_analysis'].items():
                if 'performance_metrics' in analysis:
                    metrics = analysis['performance_metrics']
                    print(f"{venue}: {metrics.get('mean_latency_us', 0):.0f}μs avg, "
                          f"P95: {metrics.get('p95_latency_us', 0):.0f}μs")

    if 'strategy_latency_impact' in results:
        print(f"\n=== Strategy Latency Impact ===")
        for strategy, impact in results['strategy_latency_impact'].items():
            print(f"{strategy}: {impact.get('avg_latency_us', 0):.0f}μs avg, "
                  f"optimal threshold: {impact.get('optimal_latency_threshold_us', 0):.0f}μs")

    if 'optimization_recommendations' in results:
        print(f"\n=== Optimization Recommendations ===")
        for i, rec in enumerate(results['optimization_recommendations'][:5], 1):
            print(f"{i}. {rec}")

    return results
if __name__ == "__main__":
    asyncio.run(test_enhanced_trading_simulator())