#!/usr/bin/env python3
"""
Enhanced Trading Simulator Integration

Seamlessly integrates realistic latency simulation with existing trading_simulator.py
Provides drop-in replacement for OrderExecutionEngine with enhanced capabilities
"""

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

# Import from your existing modules
from simulator.trading_simulator import (
    TradingSimulator, OrderExecutionEngine, MarketImpactModel,
    Order, Fill, OrderType, OrderSide, OrderStatus, TradingStrategyType
)
from simulator.network_latency_simulator import NetworkLatencySimulator

# Import our enhanced latency system
from simulator.enhanced_latency_simulation import (
    LatencySimulator, EnhancedOrderExecutionEngine, LatencyAnalytics,
    create_latency_configuration, integrate_latency_simulation
)

logger = logging.getLogger(__name__)


class EnhancedTradingSimulator(TradingSimulator):
    """
    Enhanced Trading Simulator with realistic latency modeling
    
    Drop-in replacement for existing TradingSimulator with added latency analytics
    """
    
    def __init__(self, venues: List[str], symbols: List[str], 
                 enable_latency_simulation: bool = True):
        """
        Initialize enhanced trading simulator
        
        Args:
            venues: List of venue names
            symbols: List of trading symbols  
            enable_latency_simulation: Whether to use realistic latency simulation
        """
        # Initialize parent class
        super().__init__(venues, symbols)
        
        # Replace execution engine with enhanced version
        if enable_latency_simulation:
            self.execution_engine = EnhancedOrderExecutionEngine(venues)
            logger.info("Enhanced latency simulation enabled")
        else:
            # Keep original execution engine but add latency analytics
            self.latency_simulator = LatencySimulator(venues)
            integrate_latency_simulation(self.execution_engine, self.latency_simulator)
            logger.info("Basic latency simulation enabled")
        
        # Add latency analytics
        self.latency_analytics = LatencyAnalytics(self.execution_engine)
        
        # Enhanced performance tracking
        self.latency_performance_history = []
        self.venue_latency_rankings = []
        
        # Configuration
        self.latency_config = create_latency_configuration()
        
        # Network latency simulator for market data correlation
        self.network_simulator = NetworkLatencySimulator()
        
        logger.info(f"EnhancedTradingSimulator initialized with latency analytics")
    
    async def simulate_trading(self, market_data_generator, ml_predictor,
                              duration_seconds: int = 300) -> Dict[str, Any]:
        """
        Enhanced trading simulation with comprehensive latency analysis
        
        Args:
            market_data_generator: Phase 1 market data generator
            ml_predictor: Phase 2 ML prediction system
            duration_seconds: Simulation duration
            
        Returns:
            Enhanced performance results with latency analytics
        """
        logger.info(f"Starting enhanced trading simulation for {duration_seconds} seconds")
        
        self.simulation_start_time = time.time()
        tick_count = 0
        latency_snapshots = []
        
        # Market data aggregation
        market_state = {}
        current_prices = {}
        
        async for tick in market_data_generator.generate_market_data_stream(duration_seconds):
            # Update network simulator with market conditions
            self.network_simulator.update_market_conditions(
                volatility=getattr(tick, 'volatility', 0.02),
                volume_factor=getattr(tick, 'volume', 1000) / 1000
            )
            
            # Update market state with network latency considerations
            symbol_venue_key = f"{tick.symbol}_{tick.venue}"
            
            # Get current network latency for this venue
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
            
            # Update latency simulator with current conditions
            if hasattr(self.execution_engine, 'latency_simulator'):
                self.execution_engine.latency_simulator.update_market_conditions(
                    symbol=tick.symbol,
                    volatility=getattr(tick, 'volatility', 0.02),
                    volume_factor=tick.volume / 1000000 if hasattr(tick, 'volume') else 1.0,
                    timestamp=getattr(tick, 'timestamp', time.time())
                )
            
            # Track current prices for P&L
            current_prices[tick.symbol] = tick.mid_price
            
            # Process trading signals periodically
            if tick_count % 10 == 0:
                await self._process_trading_cycle(
                    market_state, current_prices, ml_predictor, tick
                )
                
                # Capture latency performance snapshot
                if tick_count % 100 == 0:
                    snapshot = self._capture_latency_snapshot()
                    latency_snapshots.append(snapshot)
            
            tick_count += 1
            
            # Periodic logging with latency metrics
            if tick_count % 1000 == 0:
                await self._log_enhanced_progress(tick_count, latency_snapshots)
        
        # Generate enhanced results
        results = await self._generate_enhanced_results(latency_snapshots)
        
        logger.info(f"Enhanced simulation complete: {tick_count} ticks, {self.trade_count} trades")
        return results
    
    async def _process_trading_cycle(self, market_state: Dict, current_prices: Dict,
                                   ml_predictor, current_tick) -> None:
        """Process one trading cycle with enhanced latency handling"""
        # Aggregate market data for strategies
        aggregated_market_data = {
            'symbols': self.symbols,
            'venues': self.venues,
            **market_state
        }
        
        # Get enhanced ML predictions including latency forecasts
        ml_predictions = await self._get_enhanced_ml_predictions(
            ml_predictor, current_tick, market_state
        )
        
        # Generate signals from each strategy
        all_orders = []
        for strategy_name, strategy in self.strategies.items():
            orders = await strategy.generate_signals(
                aggregated_market_data, ml_predictions
            )
            
            # Enhanced order routing based on latency predictions
            enhanced_orders = self._enhance_orders_with_latency_routing(
                orders, ml_predictions, market_state
            )
            
            all_orders.extend(enhanced_orders)
        
        # Execute orders with comprehensive latency tracking
        execution_results = await self._execute_orders_with_analytics(
            all_orders, market_state, current_prices
        )
        
        # Update performance metrics
        self._update_performance_metrics(current_prices)
        
        # Analyze latency impact on performance
        self._analyze_latency_impact(execution_results)
    
    async def _get_enhanced_ml_predictions(self, ml_predictor, tick, market_state) -> Dict:
        """Get enhanced ML predictions including latency forecasts"""
        predictions = {}
        
        # Get latency predictions for each venue with confidence intervals
        for venue in self.venues:
            routing_key = f'routing_{tick.symbol}_{venue}'
            
            # Make routing decision using ML with latency considerations
            routing_decision = ml_predictor.make_routing_decision(tick.symbol)
            
            # Get current venue latency statistics
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
        
        # Market regime detection with latency considerations
        regime_detection = ml_predictor.detect_market_regime(market_state)
        predictions['regime'] = regime_detection.regime.value
        predictions['regime_confidence'] = getattr(regime_detection, 'confidence', 0.7)
        
        # Enhanced volatility forecast
        predictions['volatility_forecast'] = market_state.get(
            f"{tick.symbol}_{tick.venue}", {}
        ).get('volatility', 0.01)
        
        # Latency-adjusted momentum signals
        base_momentum = np.random.randn() * 0.5
        latency_adjustment = self._calculate_latency_momentum_adjustment(tick.symbol)
        predictions[f'momentum_signal_{tick.symbol}'] = base_momentum * latency_adjustment
        
        # Market impact predictions considering latency
        predictions['market_impact_forecast'] = self._forecast_market_impact(tick, market_state)
        
        return predictions
    
    def _enhance_orders_with_latency_routing(self, orders: List[Order], 
                                           ml_predictions: Dict, 
                                           market_state: Dict) -> List[Order]:
        """Enhance orders with latency-aware routing decisions"""
        enhanced_orders = []
        
        for order in orders:
            # Get latency predictions for all venues
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
            
            # Choose optimal venue based on strategy requirements
            optimal_venue = self._select_optimal_venue(order, venue_latencies, market_state)
            
            # Update order with enhanced routing information
            order.venue = optimal_venue
            if optimal_venue in venue_latencies:
                order.predicted_latency_us = venue_latencies[optimal_venue]['predicted_latency_us']
                order.routing_confidence = venue_latencies[optimal_venue]['confidence']
            
            # Add latency-specific order attributes
            order.latency_tolerance_us = self._calculate_latency_tolerance(order)
            order.urgency_score = self._calculate_order_urgency(order, market_state)
            
            enhanced_orders.append(order)
        
        return enhanced_orders
    
    def _select_optimal_venue(self, order: Order, venue_latencies: Dict, 
                            market_state: Dict) -> str:
        """Select optimal venue based on strategy-specific latency requirements"""
        if not venue_latencies:
            return order.venue  # Fallback to original venue
        
        strategy_requirements = self.latency_config['latency_thresholds'].get(
            order.strategy.value, {'acceptable_us': 2000, 'optimal_us': 800}
        )
        
        # Score venues based on multiple factors
        venue_scores = {}
        
        for venue, latency_info in venue_latencies.items():
            predicted_latency = latency_info['predicted_latency_us']
            confidence = latency_info['confidence']
            congestion_prob = latency_info['congestion_prob']
            
            # Base score from latency (lower is better)
            latency_score = max(0, 1.0 - (predicted_latency / strategy_requirements['acceptable_us']))
            
            # Confidence bonus
            confidence_bonus = confidence * 0.2
            
            # Congestion penalty
            congestion_penalty = congestion_prob * 0.3
            
            # Strategy-specific adjustments
            if order.strategy == TradingStrategyType.ARBITRAGE:
                # Arbitrage needs very low latency
                if predicted_latency > 500:
                    latency_score *= 0.5
            elif order.strategy == TradingStrategyType.MARKET_MAKING:
                # Market making values consistency
                latency_score *= confidence
            
            venue_scores[venue] = latency_score + confidence_bonus - congestion_penalty
        
        # Select venue with highest score
        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        return best_venue
    
    def _calculate_latency_tolerance(self, order: Order) -> float:
        """Calculate acceptable latency tolerance for order"""
        base_tolerance = self.latency_config['latency_thresholds'].get(
            order.strategy.value, {'acceptable_us': 2000}
        )['acceptable_us']
        
        # Adjust based on order characteristics
        if order.order_type == OrderType.MARKET:
            return base_tolerance * 0.7  # Market orders need faster execution
        elif order.order_type == OrderType.IOC:
            return base_tolerance * 0.5  # IOC orders are time-sensitive
        else:
            return base_tolerance
    
    def _calculate_order_urgency(self, order: Order, market_state: Dict) -> float:
        """Calculate urgency score for order (0-1 scale)"""
        urgency = 0.5  # Base urgency
        
        # Strategy-based urgency
        if order.strategy == TradingStrategyType.ARBITRAGE:
            urgency = 0.9  # Very urgent
        elif order.strategy == TradingStrategyType.MARKET_MAKING:
            urgency = 0.3  # Less urgent
        elif order.strategy == TradingStrategyType.MOMENTUM:
            urgency = 0.7  # Moderately urgent
        
        # Market condition adjustments
        symbol_venue_key = f"{order.symbol}_{order.venue}"
        if symbol_venue_key in market_state:
            volatility = market_state[symbol_venue_key].get('volatility', 0.02)
            if volatility > 0.04:  # High volatility increases urgency
                urgency = min(1.0, urgency * 1.3)
        
        return urgency
    
    async def _execute_orders_with_analytics(self, orders: List[Order], 
                                           market_state: Dict, 
                                           current_prices: Dict) -> List[Dict]:
        """Execute orders with comprehensive latency analytics"""
        execution_results = []
        
        for order in orders:
            start_time = time.perf_counter()
            
            # Execute order with enhanced latency tracking
            fill = await self._execute_order_enhanced(order, market_state)
            
            execution_time = (time.perf_counter() - start_time) * 1e6  # Convert to microseconds
            
            if fill:
                # Update strategy positions
                strategy = self.strategies[order.strategy.value]
                strategy.update_positions(fill, current_prices)
                
                # Create execution result record
                result = {
                    'order': order,
                    'fill': fill,
                    'execution_time_us': execution_time,
                    'latency_breakdown': getattr(fill, 'latency_breakdown', None),
                    'prediction_accuracy': self._calculate_prediction_accuracy(order, fill),
                    'venue_performance_rank': self._get_venue_rank(order.venue)
                }
                
                execution_results.append(result)
                
                # Track fills
                self.fill_history.append(fill)
                self.trade_count += 1
        
        return execution_results
    
    async def _execute_order_enhanced(self, order: Order, market_state: Dict) -> Optional[Fill]:
        """Execute single order with enhanced error handling and analytics"""
        # Get market state for order
        state_key = f"{order.symbol}_{order.venue}"
        if state_key not in market_state:
            logger.warning(f"No market data for {state_key}")
            return None
        
        venue_market_state = market_state[state_key]
        
        # Check if venue is experiencing issues
        if hasattr(self.execution_engine, 'latency_simulator'):
            congestion = self.execution_engine.latency_simulator.get_congestion_analysis()
            if congestion['current_congestion_level'] == 'critical':
                # Consider alternative venue for critical orders
                if getattr(order, 'urgency_score', 0.5) > 0.8:
                    alternative_venue = self._find_alternative_venue(order, market_state)
                    if alternative_venue:
                        order.venue = alternative_venue
                        state_key = f"{order.symbol}_{alternative_venue}"
                        venue_market_state = market_state.get(state_key, venue_market_state)
        
        # Execute with enhanced error handling
        try:
            fill = await self.execution_engine.execute_order(
                order, venue_market_state, getattr(order, 'predicted_latency_us', None)
            )
            
            # Validate execution quality
            if fill and hasattr(fill, 'latency_breakdown'):
                self._validate_execution_quality(order, fill)
            
            return fill
            
        except Exception as e:
            logger.error(f"Order execution failed for {order.order_id}: {e}")
            return None
    
    def _find_alternative_venue(self, order: Order, market_state: Dict) -> Optional[str]:
        """Find alternative venue during congestion"""
        if hasattr(self.execution_engine, 'get_venue_latency_rankings'):
            rankings = self.execution_engine.get_venue_latency_rankings()
            
            # Find best performing venue that's not the current one
            for venue, latency in rankings:
                if venue != order.venue and latency < 2000:  # Max 2ms latency
                    state_key = f"{order.symbol}_{venue}"
                    if state_key in market_state:
                        return venue
        
        return None
    
    def _validate_execution_quality(self, order: Order, fill: Fill) -> None:
        """Validate execution met quality expectations"""
        if hasattr(fill, 'latency_breakdown'):
            actual_latency = fill.latency_breakdown.total_latency_us
            tolerance = getattr(order, 'latency_tolerance_us', 2000)
            
            if actual_latency > tolerance:
                logger.warning(f"Order {order.order_id} exceeded latency tolerance: "
                             f"{actual_latency:.0f}μs > {tolerance:.0f}μs")
            
            # Check prediction accuracy
            if hasattr(order, 'predicted_latency_us') and order.predicted_latency_us:
                error_pct = abs(actual_latency - order.predicted_latency_us) / order.predicted_latency_us * 100
                if error_pct > 25:  # > 25% error
                    logger.warning(f"Poor latency prediction for {order.order_id}: "
                                 f"{error_pct:.1f}% error")
    
    def _capture_latency_snapshot(self) -> Dict[str, Any]:
        """Capture current latency performance snapshot"""
        snapshot = {
            'timestamp': time.time(),
            'venue_latencies': {},
            'congestion_level': 'normal',
            'prediction_accuracy': 0.0,
            'total_trades': self.trade_count
        }
        
        # Capture venue latencies
        if hasattr(self.execution_engine, 'get_venue_latency_rankings'):
            rankings = self.execution_engine.get_venue_latency_rankings()
            snapshot['venue_latencies'] = dict(rankings)
        
        # Capture congestion level
        if hasattr(self.execution_engine, 'latency_simulator'):
            congestion = self.execution_engine.latency_simulator.get_congestion_analysis()
            snapshot['congestion_level'] = congestion['current_congestion_level']
            
            # Capture prediction accuracy
            pred_stats = self.execution_engine.latency_simulator.get_prediction_accuracy_stats()
            snapshot['prediction_accuracy'] = pred_stats.get('prediction_within_10pct', 0)
        
        return snapshot
    
    async def _log_enhanced_progress(self, tick_count: int, latency_snapshots: List[Dict]) -> None:
        """Log progress with enhanced latency metrics"""
        elapsed = time.time() - self.simulation_start_time
        
        # Basic progress
        logger.info(f"Processed {tick_count} ticks in {elapsed:.1f}s, "
                   f"Total P&L: ${self.total_pnl:.2f}, Trades: {self.trade_count}")
        
        # Latency metrics
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
        """Generate enhanced simulation results with comprehensive latency analysis"""
        # Get base results
        base_results = self._generate_simulation_results()
        
        # Add comprehensive latency analysis
        latency_analysis = self.latency_analytics.generate_latency_report(
            timeframe_minutes=int((time.time() - self.simulation_start_time) / 60)
        )
        
        # Enhanced performance attribution
        performance_attribution = self._calculate_enhanced_pnl_attribution()
        
        # Strategy-specific latency impact
        strategy_latency_impact = self._analyze_strategy_latency_impact()
        
        # Market condition analysis
        market_condition_analysis = self._analyze_market_condition_impact(latency_snapshots)
        
        # Enhanced results structure
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
        """Calculate P&L attribution including latency costs"""
        attribution = {
            'total_pnl': self.total_pnl,
            'gross_pnl': 0.0,
            'latency_costs': 0.0,
            'execution_costs': 0.0,
            'by_venue': defaultdict(float),
            'by_strategy': defaultdict(float)
        }
        
        # Calculate latency costs if available
        if hasattr(self.execution_engine, 'get_latency_cost_analysis'):
            latency_costs = self.execution_engine.get_latency_cost_analysis()
            attribution['latency_costs'] = latency_costs.get('total_latency_cost_usd', 0)
        
        # Venue attribution
        for fill in self.fill_history:
            venue_pnl = 0
            if hasattr(fill, 'latency_cost_bps'):
                # Estimate P&L impact from latency
                latency_cost = fill.latency_cost_bps * fill.price * fill.quantity / 10000
                venue_pnl -= latency_cost
            
            attribution['by_venue'][fill.venue] += venue_pnl
        
        # Strategy attribution with latency impact
        for strategy_name, strategy in self.strategies.items():
            strategy_pnl = strategy.get_total_pnl()
            
            # Estimate latency impact on strategy
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
        """Analyze how latency affects each trading strategy"""
        impact_analysis = {}
        
        for strategy_name, strategy in self.strategies.items():
            # Get fills for this strategy
            strategy_fills = [
                f for f in self.fill_history 
                if hasattr(f, 'order_id') and strategy_name.upper()[:3] in f.order_id
            ]
            
            if not strategy_fills:
                continue
            
            # Calculate latency statistics
            latencies = [getattr(f, 'latency_us', 1000) for f in strategy_fills]
            slippages = [getattr(f, 'slippage_bps', 0) for f in strategy_fills]
            
            # Strategy-specific analysis
            if strategy_name == 'arbitrage':
                # For arbitrage, analyze opportunity capture rate vs latency
                impact_analysis[strategy_name] = {
                    'avg_latency_us': np.mean(latencies),
                    'opportunity_capture_rate': getattr(strategy, 'opportunities_captured', 0) / 
                                              max(getattr(strategy, 'opportunities_detected', 1), 1),
                    'latency_sensitivity': self._calculate_latency_sensitivity(latencies, slippages),
                    'optimal_latency_threshold_us': 300  # Arbitrage needs very low latency
                }
            
            elif strategy_name == 'market_making':
                # For market making, analyze spread capture vs latency
                impact_analysis[strategy_name] = {
                    'avg_latency_us': np.mean(latencies),
                    'spread_capture_efficiency': getattr(strategy, 'spread_captured', 0) / 
                                               max(len(strategy_fills), 1),
                    'maker_ratio': self._calculate_maker_ratio(strategy_fills),
                    'optimal_latency_threshold_us': 800  # Market making can tolerate moderate latency
                }
            
            elif strategy_name == 'momentum':
                # For momentum, analyze signal decay vs latency
                impact_analysis[strategy_name] = {
                    'avg_latency_us': np.mean(latencies),
                    'signal_decay_impact': self._estimate_signal_decay(latencies),
                    'execution_efficiency': self._calculate_execution_efficiency(strategy_fills),
                    'optimal_latency_threshold_us': 1500  # Momentum can handle higher latency
                }
        
        return impact_analysis
    
    def _analyze_market_condition_impact(self, latency_snapshots: List[Dict]) -> Dict[str, Any]:
        """Analyze how market conditions affect latency performance"""
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
        """Generate actionable optimization recommendations"""
        recommendations = []
        
        # Get latency analytics recommendations
        if hasattr(self.latency_analytics, '_generate_recommendations'):
            recommendations.extend(self.latency_analytics._generate_recommendations())
        
        # Strategy-specific recommendations
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
        
        # Venue optimization recommendations
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
        
        # Cost optimization
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
    
    # Helper methods for analysis
    def _detect_market_regime(self, tick) -> str:
        """Detect current market regime"""
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
        """Calculate latency trend for venue"""
        if hasattr(self.execution_engine, 'latency_simulator'):
            # Simplified trend calculation
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
        """Estimate probability of congestion at venue"""
        base_probability = 0.1
        
        # Increase probability based on market volatility
        avg_volatility = np.mean([
            state.get('volatility', 0.02) 
            for key, state in market_state.items() 
            if key.endswith(f'_{venue}')
        ])
        volatility_factor = min(avg_volatility / 0.02, 3.0)
        
        # Increase probability based on volume
        avg_volume = np.mean([
            state.get('volume', 1000) 
            for key, state in market_state.items() 
            if key.endswith(f'_{venue}')
        ])
        volume_factor = min(avg_volume / 1000, 2.0)
        
        congestion_prob = base_probability * volatility_factor * volume_factor
        return min(congestion_prob, 0.8)  # Cap at 80%
    
    def _calculate_latency_momentum_adjustment(self, symbol: str) -> float:
        """Calculate momentum signal adjustment based on latency expectations"""
        # If latency is expected to be high, reduce momentum signal strength
        # since execution will be delayed
        base_adjustment = 1.0
        
        if hasattr(self.execution_engine, 'latency_simulator'):
            # Get expected latency for this symbol across venues
            venue_latencies = []
            for venue in self.venues:
                stats = self.execution_engine.latency_simulator.get_venue_latency_stats(venue, 5)
                if stats:
                    venue_latencies.append(stats['mean_us'])
            
            if venue_latencies:
                avg_latency = np.mean(venue_latencies)
                # Reduce signal strength for high latency (>1ms reduces signal by up to 30%)
                adjustment = max(0.7, 1.0 - (avg_latency - 1000) / 5000)
                return adjustment
        
        return base_adjustment
    
    def _forecast_market_impact(self, tick, market_state: Dict) -> float:
        """Forecast market impact considering latency"""
        base_impact = 0.1  # Base market impact in bps
        
        # Adjust for volatility
        volatility = getattr(tick, 'volatility', 0.02)
        volatility_multiplier = volatility / 0.02
        
        # Adjust for expected latency
        symbol_venue_key = f"{tick.symbol}_{tick.venue}"
        if symbol_venue_key in market_state:
            network_latency = market_state[symbol_venue_key].get('current_network_latency_us', 1000)
            # Higher latency increases market impact due to adverse selection
            latency_multiplier = 1.0 + (network_latency - 500) / 2000
        else:
            latency_multiplier = 1.0
        
        return base_impact * volatility_multiplier * latency_multiplier
    
    def _calculate_prediction_accuracy(self, order: Order, fill: Fill) -> Optional[float]:
        """Calculate prediction accuracy for executed order"""
        if not hasattr(order, 'predicted_latency_us') or not order.predicted_latency_us:
            return None
        
        actual_latency = getattr(fill, 'latency_us', 0)
        if actual_latency == 0:
            return None
        
        error_pct = abs(actual_latency - order.predicted_latency_us) / order.predicted_latency_us * 100
        accuracy = max(0, 100 - error_pct)
        
        return accuracy
    
    def _get_venue_rank(self, venue: str) -> int:
        """Get current performance rank of venue (1 = best)"""
        if hasattr(self.execution_engine, 'get_venue_latency_rankings'):
            rankings = self.execution_engine.get_venue_latency_rankings()
            for rank, (v, _) in enumerate(rankings, 1):
                if v == venue:
                    return rank
        return len(self.venues)  # Worst rank if not found
    
    def _calculate_latency_sensitivity(self, latencies: List[float], slippages: List[float]) -> float:
        """Calculate how sensitive slippage is to latency changes"""
        if len(latencies) < 2 or len(slippages) < 2:
            return 0.0
        
        # Simple correlation coefficient
        latency_array = np.array(latencies)
        slippage_array = np.array(slippages)
        
        if np.std(latency_array) == 0 or np.std(slippage_array) == 0:
            return 0.0
        
        correlation = np.corrcoef(latency_array, slippage_array)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_maker_ratio(self, fills: List[Fill]) -> float:
        """Calculate ratio of maker vs taker fills"""
        if not fills:
            return 0.0
        
        # Estimate maker fills (those with rebates)
        maker_fills = sum(1 for f in fills if getattr(f, 'rebate', 0) > 0)
        return maker_fills / len(fills)
    
    def _estimate_signal_decay(self, latencies: List[float]) -> float:
        """Estimate signal decay impact from latency"""
        if not latencies:
            return 0.0
        
        avg_latency = np.mean(latencies)
        # Assume signal decays exponentially with time
        # Half-life of 500μs for momentum signals
        decay_factor = np.exp(-avg_latency / 500)
        signal_strength_loss = 1.0 - decay_factor
        
        return signal_strength_loss
    
    def _calculate_execution_efficiency(self, fills: List[Fill]) -> float:
        """Calculate execution efficiency metric"""
        if not fills:
            return 0.0
        
        # Efficiency based on slippage and latency
        total_efficiency = 0.0
        
        for fill in fills:
            slippage = getattr(fill, 'slippage_bps', 0)
            latency = getattr(fill, 'latency_us', 1000)
            
            # Lower slippage and latency = higher efficiency
            slippage_efficiency = max(0, 1.0 - slippage / 10)  # Normalize by 10 bps
            latency_efficiency = max(0, 1.0 - latency / 2000)  # Normalize by 2ms
            
            fill_efficiency = (slippage_efficiency + latency_efficiency) / 2
            total_efficiency += fill_efficiency
        
        return total_efficiency / len(fills)
    
    def _analyze_congestion_impact(self, latency_snapshots: List[Dict]) -> Dict[str, Any]:
        """Analyze impact of network congestion on performance"""
        congestion_analysis = {
            'low_congestion_performance': {},
            'high_congestion_performance': {},
            'congestion_cost_bps': 0.0
        }
        
        if not latency_snapshots:
            return congestion_analysis
        
        # Separate snapshots by congestion level
        low_congestion = [s for s in latency_snapshots if s['congestion_level'] in ['low', 'normal']]
        high_congestion = [s for s in latency_snapshots if s['congestion_level'] in ['high', 'critical']]
        
        # Analyze performance differences
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
            
            # Estimate congestion cost
            if low_congestion and high_latencies and congestion_analysis['low_congestion_performance']['avg_latency_us'] > 0:
                latency_increase = (congestion_analysis['high_congestion_performance']['avg_latency_us'] - 
                                  congestion_analysis['low_congestion_performance']['avg_latency_us'])
                congestion_analysis['congestion_cost_bps'] = latency_increase / 100 * 0.1  # 0.1 bps per 100μs
        
        return congestion_analysis
    
    def _analyze_volatility_latency_correlation(self) -> Dict[str, float]:
        """Analyze correlation between market volatility and latency"""
        if not hasattr(self.execution_engine, 'latency_simulator'):
            return {}
        
        # Get recent latency data with timestamps
        recent_latencies = list(self.execution_engine.latency_simulator.latency_history)
        
        if len(recent_latencies) < 10:
            return {}
        
        # Extract volatility factors and latencies
        volatility_factors = [b.volatility_factor for b in recent_latencies]
        latencies = [b.total_latency_us for b in recent_latencies]
        
        # Calculate correlation
        if np.std(volatility_factors) > 0 and np.std(latencies) > 0:
            correlation = np.corrcoef(volatility_factors, latencies)[0, 1]
            return {
                'volatility_latency_correlation': correlation if not np.isnan(correlation) else 0.0,
                'sample_count': len(recent_latencies)
            }
        
        return {'volatility_latency_correlation': 0.0, 'sample_count': len(recent_latencies)}
    
    def _analyze_volume_latency_correlation(self) -> Dict[str, float]:
        """Analyze correlation between trading volume and latency"""
        # Similar to volatility analysis but for volume
        # Simplified implementation - would use actual volume data in production
        return {
            'volume_latency_correlation': 0.3,  # Positive correlation expected
            'sample_count': len(self.fill_history)
        }
    
    def _analyze_time_of_day_effects(self) -> Dict[str, Any]:
        """Analyze latency patterns by time of day"""
        time_analysis = {}
        
        if not hasattr(self.execution_engine, 'latency_simulator'):
            return time_analysis
        
        # Group latencies by hour
        hourly_latencies = defaultdict(list)
        
        for breakdown in self.execution_engine.latency_simulator.latency_history:
            hour = datetime.fromtimestamp(breakdown.timestamp).hour
            hourly_latencies[hour].append(breakdown.total_latency_us)
        
        # Calculate statistics for each hour
        for hour, latencies in hourly_latencies.items():
            if latencies:
                time_analysis[f'hour_{hour}'] = {
                    'avg_latency_us': np.mean(latencies),
                    'sample_count': len(latencies),
                    'volatility': np.std(latencies)
                }
        
        # Identify peak latency periods
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
        """Analyze latency impact on execution results"""
        if not execution_results:
            return
        
        # Track latency vs performance metrics
        for result in execution_results:
            if result['latency_breakdown']:
                latency = result['latency_breakdown'].total_latency_us
                fill = result['fill']
                
                # Track correlation between latency and slippage
                if hasattr(fill, 'slippage_bps'):
                    # Store for correlation analysis
                    pass  # Implementation would store this data
                
                # Track prediction accuracy
                if result['prediction_accuracy']:
                    # Store for ML model improvement
                    pass  # Implementation would update ML models


# Configuration and setup functions
def create_enhanced_trading_simulator(symbols: List[str], venues: List[str], 
                                    config: Dict[str, Any] = None) -> EnhancedTradingSimulator:
    """
    Factory function to create enhanced trading simulator with optimal configuration
    
    Args:
        symbols: Trading symbols
        venues: Trading venues  
        config: Optional configuration overrides
        
    Returns:
        Configured EnhancedTradingSimulator instance
    """
    # Default configuration
    default_config = {
        'enable_latency_simulation': True,
        'latency_prediction_enabled': True,
        'congestion_modeling': True,
        'venue_optimization': True,
        'strategy_latency_optimization': True
    }
    
    final_config = {**default_config, **(config or {})}
    
    # Create simulator
    simulator = EnhancedTradingSimulator(
        venues=venues,
        symbols=symbols,
        enable_latency_simulation=final_config['enable_latency_simulation']
    )
    
    # Configure latency thresholds based on venue characteristics
    if final_config['venue_optimization']:
        _configure_venue_optimization(simulator)
    
    # Configure strategy-specific latency parameters
    if final_config['strategy_latency_optimization']:
        _configure_strategy_latency_parameters(simulator)
    
    logger.info(f"Enhanced trading simulator created with {len(symbols)} symbols, "
               f"{len(venues)} venues, config: {final_config}")
    
    return simulator


def _configure_venue_optimization(simulator: EnhancedTradingSimulator):
    """Configure venue-specific optimization parameters"""
    # Adjust venue profiles based on empirical performance data
    if hasattr(simulator.execution_engine, 'latency_simulator'):
        latency_sim = simulator.execution_engine.latency_simulator
        
        # Fine-tune venue profiles for optimal performance
        for venue, profile in latency_sim.venue_profiles.items():
            if venue == 'IEX':
                # IEX has speed bump - adjust for consistency
                profile.network_latency_std_us *= 0.5  # More consistent
                profile.spike_probability *= 0.3  # Fewer spikes
            elif venue == 'NASDAQ':
                # NASDAQ optimized for speed
                profile.processing_rate_msg_per_sec *= 1.2  # Higher throughput
            elif venue == 'CBOE':
                # CBOE can be more variable
                profile.volatility_sensitivity *= 1.1  # More sensitive to volatility


def _configure_strategy_latency_parameters(simulator: EnhancedTradingSimulator):
    """Configure strategy-specific latency optimization parameters"""
    # Adjust strategy parameters based on latency requirements
    
    # Market making - optimize for consistency
    if 'market_making' in simulator.strategies:
        mm_strategy = simulator.strategies['market_making']
        mm_strategy.params['latency_weight'] = 0.3  # Moderate latency importance
        mm_strategy.params['consistency_weight'] = 0.7  # High consistency importance
    
    # Arbitrage - optimize for speed
    if 'arbitrage' in simulator.strategies:
        arb_strategy = simulator.strategies['arbitrage']
        arb_strategy.params['latency_weight'] = 0.8  # High latency importance
        arb_strategy.params['max_acceptable_latency_us'] = 300  # Very strict limit
    
    # Momentum - balance speed and accuracy
    if 'momentum' in simulator.strategies:
        mom_strategy = simulator.strategies['momentum']
        mom_strategy.params['latency_weight'] = 0.5  # Moderate latency importance
        mom_strategy.params['signal_decay_factor'] = 0.95  # Account for execution delay


# Utility functions for easy integration
def patch_existing_simulator(existing_simulator: TradingSimulator, 
                           enable_enhanced_latency: bool = True) -> TradingSimulator:
    """
    Patch an existing TradingSimulator instance with enhanced latency capabilities
    
    Args:
        existing_simulator: Existing TradingSimulator instance
        enable_enhanced_latency: Whether to enable full enhanced latency simulation
        
    Returns:
        The patched simulator (same instance, modified)
    """
    logger.info("Patching existing TradingSimulator with enhanced latency capabilities")
    
    # Store original execution engine
    original_execution_engine = existing_simulator.execution_engine
    
    if enable_enhanced_latency:
        # Replace with enhanced execution engine
        existing_simulator.execution_engine = EnhancedOrderExecutionEngine(existing_simulator.venues)
    else:
        # Add basic latency simulation to existing engine
        latency_simulator = LatencySimulator(existing_simulator.venues)
        integrate_latency_simulation(original_execution_engine, latency_simulator)
    
    # Add latency analytics
    existing_simulator.latency_analytics = LatencyAnalytics(existing_simulator.execution_engine)
    existing_simulator.latency_config = create_latency_configuration()
    
    logger.info("Existing simulator successfully patched with latency capabilities")
    return existing_simulator


def quick_latency_test(simulator: EnhancedTradingSimulator, venue: str, 
                      num_samples: int = 100) -> Dict[str, Any]:
    """
    Quick latency test for a specific venue
    
    Args:
        simulator: Enhanced trading simulator
        venue: Venue to test
        num_samples: Number of latency samples to generate
        
    Returns:
        Latency test results
    """
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


# Example usage and testing
async def test_enhanced_trading_simulator():
    """Test the enhanced trading simulator with realistic latency"""
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    venues = ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']
    
    # Create enhanced simulator
    simulator = create_enhanced_trading_simulator(symbols, venues)
    
    print("Testing Enhanced Trading Simulator with Realistic Latency...")
    
    # Mock market data generator and ML predictor for testing
    class MockMarketDataGenerator:
        async def generate_market_data_stream(self, duration_seconds):
            for i in range(duration_seconds * 10):  # 10 ticks per second
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
                
                await asyncio.sleep(0.01)  # 10ms between ticks
    
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
    
    # Run simulation
    market_generator = MockMarketDataGenerator()
    ml_predictor = MockMLPredictor()
    
    results = await simulator.simulate_trading(
        market_generator, ml_predictor, duration_seconds=30
    )
    
    # Display results
    print(f"\n=== Enhanced Simulation Results ===")
    print(f"Total P&L: ${results['summary']['total_pnl']:.2f}")
    print(f"Trades executed: {results['summary']['trade_count']}")
    print(f"Max drawdown: ${results['summary']['max_drawdown']:.2f}")
    
    # Latency analysis
    if 'latency_analysis' in results:
        latency = results['latency_analysis']
        print(f"\n=== Latency Analysis ===")
        if 'summary' in latency:
            summary = latency['summary']
            print(f"Average latency: {summary.get('avg_total_latency_us', 0):.0f}μs")
            print(f"Latency cost: {summary.get('avg_latency_cost_bps', 0):.3f} bps")
            print(f"Prediction accuracy: {summary.get('prediction_accuracy_pct', 0):.1f}%")
        
        # Venue performance
        if 'venue_analysis' in latency:
            print(f"\n=== Venue Performance ===")
            for venue, analysis in latency['venue_analysis'].items():
                if 'performance_metrics' in analysis:
                    metrics = analysis['performance_metrics']
                    print(f"{venue}: {metrics.get('mean_latency_us', 0):.0f}μs avg, "
                          f"P95: {metrics.get('p95_latency_us', 0):.0f}μs")
    
    # Strategy impact
    if 'strategy_latency_impact' in results:
        print(f"\n=== Strategy Latency Impact ===")
        for strategy, impact in results['strategy_latency_impact'].items():
            print(f"{strategy}: {impact.get('avg_latency_us', 0):.0f}μs avg, "
                  f"optimal threshold: {impact.get('optimal_latency_threshold_us', 0):.0f}μs")
    
    # Optimization recommendations
    if 'optimization_recommendations' in results:
        print(f"\n=== Optimization Recommendations ===")
        for i, rec in enumerate(results['optimization_recommendations'][:5], 1):
            print(f"{i}. {rec}")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_enhanced_trading_simulator())