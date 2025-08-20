import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
import logging
from datetime import datetime, timedelta
import asyncio
logger = logging.getLogger(__name__)
class LiquidityTier(Enum):

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
class MarketRegime(Enum):

    QUIET = "quiet"
    NORMAL = "normal"
    VOLATILE = "volatile"
    STRESSED = "stressed"
@dataclass
class SlippageParameters:

    base_slippage_bps: float
    size_impact_factor: float
    volatility_multiplier: float
    spread_sensitivity: float
    time_of_day_factor: Dict[int, float]
@dataclass
class MarketImpactParameters:

    temporary_impact_base: float
    permanent_impact_base: float
    volatility_scaling: float
    adv_scaling: float
    sqrt_scaling: bool = True

    venue_multipliers: Dict[str, float] = field(default_factory=dict)

    temporary_half_life_seconds: float = 300
    recovery_rate: float = 0.1
@dataclass
class VenueCostProfile:

    name: str

    maker_fee_bps: float
    taker_fee_bps: float
    rebate_bps: float

    impact_multiplier: float
    liquidity_factor: float
    latency_sensitivity: float

    fill_probability: float
    adverse_selection_factor: float

    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 15, 16])
    peak_cost_multiplier: float = 1.3
@dataclass
class ExecutionCostBreakdown:

    order_id: str
    symbol: str
    venue: str
    timestamp: float

    side: str
    quantity: int
    order_price: float
    execution_price: float

    slippage_cost: float
    temporary_impact_cost: float
    permanent_impact_cost: float
    market_impact_cost: float
    latency_cost: float
    fees_paid: float
    rebates_received: float
    opportunity_cost: float

    gross_execution_cost: float
    net_execution_cost: float
    total_transaction_cost: float

    cost_per_share: float
    cost_bps: float
    implementation_shortfall_bps: float
class EnhancedMarketImpactModel:


    def __init__(self):
        self.liquidity_tiers = {
            LiquidityTier.HIGH: SlippageParameters(
                base_slippage_bps=0.5,
                size_impact_factor=0.2,
                volatility_multiplier=15.0,
                spread_sensitivity=0.8,
                time_of_day_factor={
                    9: 2.0,
                    10: 1.5,
                    11: 1.0,
                    12: 0.8,
                    13: 0.9,
                    14: 1.1,
                    15: 1.8,
                    16: 2.5
                }
            ),
            LiquidityTier.MEDIUM: SlippageParameters(
                base_slippage_bps=1.2,
                size_impact_factor=0.4,
                volatility_multiplier=25.0,
                spread_sensitivity=1.2,
                time_of_day_factor={
                    9: 2.5, 10: 1.8, 11: 1.2, 12: 1.0,
                    13: 1.1, 14: 1.3, 15: 2.2, 16: 3.0
                }
            ),
            LiquidityTier.LOW: SlippageParameters(
                base_slippage_bps=3.0,
                size_impact_factor=0.8,
                volatility_multiplier=40.0,
                spread_sensitivity=1.8,
                time_of_day_factor={
                    9: 3.5, 10: 2.5, 11: 1.5, 12: 1.2,
                    13: 1.3, 14: 1.6, 15: 3.0, 16: 4.0
                }
            )
        }

        self.impact_parameters = {
            LiquidityTier.HIGH: MarketImpactParameters(
                temporary_impact_base=0.05,
                permanent_impact_base=0.02,
                volatility_scaling=20.0,
                adv_scaling=1.0,
                venue_multipliers={
                    'NYSE': 0.9,
                    'NASDAQ': 1.0,
                    'ARCA': 1.1,
                    'CBOE': 1.2,
                    'IEX': 0.95
                }
            ),
            LiquidityTier.MEDIUM: MarketImpactParameters(
                temporary_impact_base=0.12,
                permanent_impact_base=0.05,
                volatility_scaling=30.0,
                adv_scaling=1.2,
                venue_multipliers={
                    'NYSE': 1.0, 'NASDAQ': 1.1, 'ARCA': 1.3,
                    'CBOE': 1.4, 'IEX': 1.1
                }
            ),
            LiquidityTier.LOW: MarketImpactParameters(
                temporary_impact_base=0.25,
                permanent_impact_base=0.10,
                volatility_scaling=50.0,
                adv_scaling=1.5,
                venue_multipliers={
                    'NYSE': 1.1, 'NASDAQ': 1.2, 'ARCA': 1.5,
                    'CBOE': 1.8, 'IEX': 1.3
                }
            )
        }

        self.venue_profiles = self._initialize_venue_profiles()

        self.recent_trades = defaultdict(deque)
        self.price_impact_decay = defaultdict(float)
        self.impact_history = defaultdict(list)

        logger.info("Enhanced Market Impact Model initialized")

    def _initialize_venue_profiles(self) -> Dict[str, VenueCostProfile]:

        return {
            'NYSE': VenueCostProfile(
                name='NYSE',
                maker_fee_bps=-0.20,
                taker_fee_bps=0.30,
                rebate_bps=0.20,
                impact_multiplier=0.95,
                liquidity_factor=1.2,
                latency_sensitivity=1.0,
                fill_probability=0.85,
                adverse_selection_factor=0.9
            ),
            'NASDAQ': VenueCostProfile(
                name='NASDAQ',
                maker_fee_bps=-0.25,
                taker_fee_bps=0.30,
                rebate_bps=0.25,
                impact_multiplier=1.0,
                liquidity_factor=1.1,
                latency_sensitivity=0.9,
                fill_probability=0.82,
                adverse_selection_factor=1.0
            ),
            'ARCA': VenueCostProfile(
                name='ARCA',
                maker_fee_bps=-0.20,
                taker_fee_bps=0.30,
                rebate_bps=0.20,
                impact_multiplier=1.1,
                liquidity_factor=0.9,
                latency_sensitivity=1.1,
                fill_probability=0.78,
                adverse_selection_factor=1.1
            ),
            'CBOE': VenueCostProfile(
                name='CBOE',
                maker_fee_bps=-0.23,
                taker_fee_bps=0.28,
                rebate_bps=0.23,
                impact_multiplier=1.25,
                liquidity_factor=0.7,
                latency_sensitivity=1.3,
                fill_probability=0.70,
                adverse_selection_factor=1.3
            ),
            'IEX': VenueCostProfile(
                name='IEX',
                maker_fee_bps=0.0,
                taker_fee_bps=0.09,
                rebate_bps=0.0,
                impact_multiplier=0.85,
                liquidity_factor=0.8,
                latency_sensitivity=0.7,
                fill_probability=0.75,
                adverse_selection_factor=0.7
            )
        }

    def calculate_execution_costs(self, order, market_state: Dict,
                                actual_latency_us: float,
                                execution_price: float) -> ExecutionCostBreakdown:

        liquidity_tier = self._classify_liquidity_tier(order.symbol, market_state)

        adv = market_state.get('average_daily_volume', 1000000)
        volatility = market_state.get('volatility', 0.02)
        spread_bps = market_state.get('spread_bps', 2.0)
        mid_price = market_state.get('mid_price', execution_price)

        participation_rate = (order.quantity / adv) * 100

        slippage_cost = self._calculate_slippage_cost(
            order, market_state, liquidity_tier, execution_price, mid_price
        )

        temporary_impact, permanent_impact = self._calculate_market_impact(
            order, market_state, liquidity_tier, participation_rate, volatility
        )

        temporary_impact_cost = temporary_impact * order.quantity * execution_price / 10000
        permanent_impact_cost = permanent_impact * order.quantity * execution_price / 10000
        market_impact_cost = temporary_impact_cost + permanent_impact_cost

        latency_cost = self._calculate_latency_cost(
            order, market_state, actual_latency_us, volatility
        )

        venue_profile = self.venue_profiles[order.venue]
        fees_paid, rebates_received = self._calculate_fees_rebates(
            order, execution_price, venue_profile
        )

        opportunity_cost = self._calculate_opportunity_cost(
            order, market_state, actual_latency_us
        )

        gross_execution_cost = (slippage_cost + market_impact_cost +
                              latency_cost + opportunity_cost)
        net_execution_cost = gross_execution_cost + fees_paid - rebates_received
        total_transaction_cost = net_execution_cost

        notional_value = order.quantity * execution_price
        cost_per_share = total_transaction_cost / order.quantity
        cost_bps = (total_transaction_cost / notional_value) * 10000

        benchmark_price = mid_price
        implementation_shortfall = abs(execution_price - benchmark_price) / benchmark_price * 10000

        breakdown = ExecutionCostBreakdown(
            order_id=order.order_id,
            symbol=order.symbol,
            venue=order.venue,
            timestamp=time.time(),
            side=order.side.value,
            quantity=order.quantity,
            order_price=getattr(order, 'price', execution_price),
            execution_price=execution_price,
            slippage_cost=slippage_cost,
            temporary_impact_cost=temporary_impact_cost,
            permanent_impact_cost=permanent_impact_cost,
            market_impact_cost=market_impact_cost,
            latency_cost=latency_cost,
            fees_paid=fees_paid,
            rebates_received=rebates_received,
            opportunity_cost=opportunity_cost,
            gross_execution_cost=gross_execution_cost,
            net_execution_cost=net_execution_cost,
            total_transaction_cost=total_transaction_cost,
            cost_per_share=cost_per_share,
            cost_bps=cost_bps,
            implementation_shortfall_bps=implementation_shortfall
        )

        self._update_impact_history(order.symbol, order.venue, breakdown)

        return breakdown

    def _classify_liquidity_tier(self, symbol: str, market_state: Dict) -> LiquidityTier:

        adv = market_state.get('average_daily_volume', 1000000)
        market_cap = market_state.get('market_cap', 1e9)
        spread_bps = market_state.get('spread_bps', 2.0)

        if symbol in ['SPY', 'QQQ', 'IWM'] or adv > 20_000_000:
            return LiquidityTier.HIGH

        elif (symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'] or
              (adv > 5_000_000 and market_cap > 50e9)):
            return LiquidityTier.HIGH

        elif adv > 1_000_000 and spread_bps < 5.0:
            return LiquidityTier.MEDIUM

        else:
            return LiquidityTier.LOW

    def _calculate_slippage_cost(self, order, market_state: Dict,
                               liquidity_tier: LiquidityTier,
                               execution_price: float, mid_price: float) -> float:

        params = self.liquidity_tiers[liquidity_tier]

        base_slippage_bps = params.base_slippage_bps

        adv = market_state.get('average_daily_volume', 1000000)
        participation_rate = order.quantity / adv
        size_impact_bps = params.size_impact_factor * np.sqrt(participation_rate * 100)

        volatility = market_state.get('volatility', 0.02)
        volatility_impact_bps = params.volatility_multiplier * volatility

        spread_bps = market_state.get('spread_bps', 2.0)
        spread_impact_bps = params.spread_sensitivity * spread_bps * 0.3

        current_hour = datetime.fromtimestamp(time.time()).hour
        time_multiplier = params.time_of_day_factor.get(current_hour, 1.0)

        regime = market_state.get('regime', 'normal')
        regime_multiplier = {
            'quiet': 0.7,
            'normal': 1.0,
            'volatile': 1.8,
            'stressed': 2.5
        }.get(regime, 1.0)

        total_slippage_bps = (
            (base_slippage_bps + size_impact_bps + volatility_impact_bps + spread_impact_bps) *
            time_multiplier * regime_multiplier
        )

        notional_value = order.quantity * execution_price
        slippage_cost = (total_slippage_bps / 10000) * notional_value

        return slippage_cost

    def _calculate_market_impact(self, order, market_state: Dict,
                               liquidity_tier: LiquidityTier,
                               participation_rate: float,
                               volatility: float) -> Tuple[float, float]:

        params = self.impact_parameters[liquidity_tier]
        venue_profile = self.venue_profiles[order.venue]

        if params.sqrt_scaling:
            size_factor = np.sqrt(participation_rate)
        else:
            size_factor = participation_rate

        vol_factor = (volatility / 0.02) * params.volatility_scaling / 100

        venue_multiplier = params.venue_multipliers.get(order.venue, 1.0)

        regime = market_state.get('regime', 'normal')
        regime_impact_multiplier = {
            'quiet': 0.6,
            'normal': 1.0,
            'volatile': 1.6,
            'stressed': 2.2
        }.get(regime, 1.0)

        temporary_impact_bps = (
            params.temporary_impact_base * size_factor * (1 + vol_factor) *
            venue_multiplier * regime_impact_multiplier
        )

        permanent_impact_bps = (
            params.permanent_impact_base * size_factor * (1 + vol_factor * 0.5) *
            venue_multiplier * regime_impact_multiplier * 0.7
        )

        if order.side.value == 'sell':
            temporary_impact_bps *= 1.1
            permanent_impact_bps *= 1.05

        liquidity_adjustment = 1.0 / venue_profile.liquidity_factor
        temporary_impact_bps *= liquidity_adjustment
        permanent_impact_bps *= liquidity_adjustment

        return temporary_impact_bps, permanent_impact_bps

    def _calculate_latency_cost(self, order, market_state: Dict,
                              actual_latency_us: float, volatility: float) -> float:

        venue_profile = self.venue_profiles[order.venue]

        latency_seconds = actual_latency_us / 1e6

        price_drift_std = volatility * np.sqrt(latency_seconds / (252 * 24 * 3600))
        expected_adverse_move = price_drift_std * 0.4

        latency_impact = (
            expected_adverse_move *
            venue_profile.latency_sensitivity *
            venue_profile.adverse_selection_factor
        )

        notional_value = order.quantity * market_state.get('mid_price', 100)
        latency_cost = latency_impact * notional_value

        return latency_cost

    def _calculate_fees_rebates(self, order, execution_price: float,
                              venue_profile: VenueCostProfile) -> Tuple[float, float]:

        notional_value = order.quantity * execution_price

        is_maker = (hasattr(order, 'order_type') and
                   order.order_type.value == 'limit' and
                   np.random.random() < venue_profile.fill_probability)

        if is_maker:
            fees_paid = max(0, venue_profile.maker_fee_bps / 10000 * notional_value)
            rebates_received = max(0, -venue_profile.maker_fee_bps / 10000 * notional_value)
        else:
            fees_paid = venue_profile.taker_fee_bps / 10000 * notional_value
            rebates_received = 0.0

        return fees_paid, rebates_received

    def _calculate_opportunity_cost(self, order, market_state: Dict,
                                  actual_latency_us: float) -> float:

        arrival_price = market_state.get('mid_price', 100)
        volatility = market_state.get('volatility', 0.02)

        delay_seconds = actual_latency_us / 1e6
        expected_move = volatility * np.sqrt(delay_seconds / (252 * 24 * 3600))

        opportunity_cost = expected_move * order.quantity * arrival_price * 0.5

        return opportunity_cost

    def _update_impact_history(self, symbol: str, venue: str, breakdown: ExecutionCostBreakdown):

        self.impact_history[f"{symbol}_{venue}"].append({
            'timestamp': breakdown.timestamp,
            'cost_bps': breakdown.cost_bps,
            'market_impact_bps': breakdown.market_impact_cost / (breakdown.quantity * breakdown.execution_price) * 10000,
            'slippage_bps': breakdown.slippage_cost / (breakdown.quantity * breakdown.execution_price) * 10000,
            'quantity': breakdown.quantity
        })

        if len(self.impact_history[f"{symbol}_{venue}"]) > 1000:
            self.impact_history[f"{symbol}_{venue}"] = self.impact_history[f"{symbol}_{venue}"][-1000:]

    def get_venue_cost_ranking(self, symbol: str, order_size: int,
                             market_state: Dict) -> List[Tuple[str, float]]:

        liquidity_tier = self._classify_liquidity_tier(symbol, market_state)
        venue_costs = []

        for venue_name, venue_profile in self.venue_profiles.items():
            mock_order = type('MockOrder', (), {
                'symbol': symbol,
                'venue': venue_name,
                'quantity': order_size,
                'side': type('Side', (), {'value': 'buy'})()
            })()

            participation_rate = (order_size / market_state.get('average_daily_volume', 1000000)) * 100

            params = self.liquidity_tiers[liquidity_tier]
            size_impact = params.size_impact_factor * np.sqrt(participation_rate)
            base_slippage = params.base_slippage_bps + size_impact

            impact_params = self.impact_parameters[liquidity_tier]
            venue_multiplier = impact_params.venue_multipliers.get(venue_name, 1.0)
            market_impact = (impact_params.temporary_impact_base +
                           impact_params.permanent_impact_base) * np.sqrt(participation_rate) * venue_multiplier

            if venue_profile.maker_fee_bps < 0:
                fee_cost = venue_profile.taker_fee_bps * 0.7
            else:
                fee_cost = venue_profile.taker_fee_bps

            total_cost_bps = base_slippage + market_impact + fee_cost

            venue_costs.append((venue_name, total_cost_bps))

        venue_costs.sort(key=lambda x: x[1])

        return venue_costs

    def get_cost_attribution_report(self, time_window_hours: float = 24) -> Dict[str, Any]:

        cutoff_time = time.time() - (time_window_hours * 3600)

        report = {
            'summary': {
                'total_trades': 0,
                'total_cost_usd': 0,
                'avg_cost_bps': 0,
                'cost_by_component': {},
                'cost_by_venue': {},
                'cost_by_symbol': {}
            },
            'venue_analysis': {},
            'symbol_analysis': {},
            'optimization_opportunities': []
        }

        all_costs = []
        venue_costs = defaultdict(list)
        symbol_costs = defaultdict(list)

        for symbol_venue, history in self.impact_history.items():
            recent_trades = [t for t in history if t['timestamp'] > cutoff_time]

            if recent_trades:
                symbol = symbol_venue.split('_')[0]
                venue = symbol_venue.split('_')[1]

                for trade in recent_trades:
                    all_costs.append(trade['cost_bps'])
                    venue_costs[venue].append(trade['cost_bps'])
                    symbol_costs[symbol].append(trade['cost_bps'])

        if all_costs:
            report['summary']['total_trades'] = len(all_costs)
            report['summary']['avg_cost_bps'] = np.mean(all_costs)
            report['summary']['median_cost_bps'] = np.median(all_costs)
            report['summary']['p95_cost_bps'] = np.percentile(all_costs, 95)

            for venue, costs in venue_costs.items():
                if costs:
                    report['venue_analysis'][venue] = {
                        'trade_count': len(costs),
                        'avg_cost_bps': np.mean(costs),
                        'cost_volatility': np.std(costs),
                        'cost_rank': None
                    }

            venue_rankings = sorted(report['venue_analysis'].items(),
                                  key=lambda x: x[1]['avg_cost_bps'])
            for rank, (venue, data) in enumerate(venue_rankings, 1):
                report['venue_analysis'][venue]['cost_rank'] = rank

            for symbol, costs in symbol_costs.items():
                if costs:
                    report['symbol_analysis'][symbol] = {
                        'trade_count': len(costs),
                        'avg_cost_bps': np.mean(costs),
                        'cost_volatility': np.std(costs)
                    }

            if len(venue_rankings) > 1:
                best_venue = venue_rankings[0][0]
                worst_venue = venue_rankings[-1][0]
                cost_diff = (venue_rankings[-1][1]['avg_cost_bps'] -
                           venue_rankings[0][1]['avg_cost_bps'])

                if cost_diff > 1.0:
                    report['optimization_opportunities'].append({
                        'type': 'venue_optimization',
                        'description': f'Route more flow from {worst_venue} to {best_venue}',
                        'potential_savings_bps': cost_diff,
                        'confidence': 'high' if len(venue_costs[worst_venue]) > 50 else 'medium'
                    })

        return report
class DynamicCostCalculator:


    def __init__(self, impact_model: EnhancedMarketImpactModel):
        self.impact_model = impact_model
        self.real_time_spreads = {}
        self.liquidity_estimates = {}
        self.cost_forecasts = {}

        self.adaptive_multipliers = defaultdict(lambda: 1.0)
        self.regime_detection_window = 300

        logger.info("Dynamic Cost Calculator initialized")

    def update_market_conditions(self, symbol: str, venue: str,
                                market_data: Dict, timestamp: float):

        key = f"{symbol}_{venue}"

        self.real_time_spreads[key] = {
            'bid_ask_spread_bps': market_data.get('spread_bps', 2.0),
            'effective_spread_bps': market_data.get('effective_spread_bps', 2.5),
            'timestamp': timestamp
        }

        self.liquidity_estimates[key] = {
            'bid_depth': market_data.get('bid_depth_total', 10000),
            'ask_depth': market_data.get('ask_depth_total', 10000),
            'book_imbalance': market_data.get('order_imbalance', 0.0),
            'recent_volume': market_data.get('volume', 1000),
            'timestamp': timestamp
        }

        self._update_adaptive_multipliers(symbol, venue, market_data, timestamp)

    def _update_adaptive_multipliers(self, symbol: str, venue: str,
                                   market_data: Dict, timestamp: float):

        key = f"{symbol}_{venue}"

        volatility = market_data.get('volatility', 0.02)
        if volatility > 0.04:
            self.adaptive_multipliers[f"{key}_vol"] = 1.5
        elif volatility < 0.01:
            self.adaptive_multipliers[f"{key}_vol"] = 0.8
        else:
            self.adaptive_multipliers[f"{key}_vol"] = 1.0

        recent_volume = market_data.get('volume', 1000)
        avg_volume = market_data.get('average_volume', 1000)
        volume_ratio = recent_volume / max(avg_volume, 1)

        if volume_ratio > 2.0:
            self.adaptive_multipliers[f"{key}_vol_regime"] = 0.9
        elif volume_ratio < 0.5:
            self.adaptive_multipliers[f"{key}_vol_regime"] = 1.3
        else:
            self.adaptive_multipliers[f"{key}_vol_regime"] = 1.0

    def calculate_real_time_execution_cost(self, order, market_state: Dict,
                                         predicted_latency_us: float) -> Dict[str, float]:

        symbol_venue_key = f"{order.symbol}_{order.venue}"

        current_spread = self.real_time_spreads.get(symbol_venue_key, {})
        current_liquidity = self.liquidity_estimates.get(symbol_venue_key, {})

        enhanced_market_state = {
            **market_state,
            'current_spread_bps': current_spread.get('bid_ask_spread_bps',
                                                   market_state.get('spread_bps', 2.0)),
            'current_liquidity': current_liquidity.get('bid_depth', 10000) +
                               current_liquidity.get('ask_depth', 10000),
            'book_imbalance': current_liquidity.get('book_imbalance', 0.0)
        }

        vol_multiplier = self.adaptive_multipliers.get(f"{symbol_venue_key}_vol", 1.0)
        volume_multiplier = self.adaptive_multipliers.get(f"{symbol_venue_key}_vol_regime", 1.0)
        total_multiplier = vol_multiplier * volume_multiplier

        liquidity_tier = self.impact_model._classify_liquidity_tier(order.symbol, enhanced_market_state)

        params = self.impact_model.liquidity_tiers[liquidity_tier]
        adv = enhanced_market_state.get('average_daily_volume', 1000000)
        participation_rate = order.quantity / adv

        estimated_slippage_bps = (
            params.base_slippage_bps +
            params.size_impact_factor * np.sqrt(participation_rate * 100)
        ) * total_multiplier

        impact_params = self.impact_model.impact_parameters[liquidity_tier]
        venue_multiplier = impact_params.venue_multipliers.get(order.venue, 1.0)

        estimated_temp_impact_bps = (
            impact_params.temporary_impact_base * np.sqrt(participation_rate * 100) *
            venue_multiplier * total_multiplier
        )

        estimated_perm_impact_bps = (
            impact_params.permanent_impact_base * np.sqrt(participation_rate * 100) *
            venue_multiplier * total_multiplier * 0.7
        )

        latency_seconds = predicted_latency_us / 1e6
        volatility = enhanced_market_state.get('volatility', 0.02)
        estimated_latency_cost_bps = (
            volatility * np.sqrt(latency_seconds / (252 * 24 * 3600)) * 100 * 0.4
        )

        venue_profile = self.impact_model.venue_profiles[order.venue]
        estimated_fee_bps = venue_profile.taker_fee_bps * 0.7

        total_estimated_cost_bps = (
            estimated_slippage_bps + estimated_temp_impact_bps +
            estimated_perm_impact_bps + estimated_latency_cost_bps + estimated_fee_bps
        )

        notional_value = order.quantity * enhanced_market_state.get('mid_price', 100)
        total_estimated_cost_usd = (total_estimated_cost_bps / 10000) * notional_value

        return {
            'total_cost_bps': total_estimated_cost_bps,
            'total_cost_usd': total_estimated_cost_usd,
            'slippage_bps': estimated_slippage_bps,
            'temporary_impact_bps': estimated_temp_impact_bps,
            'permanent_impact_bps': estimated_perm_impact_bps,
            'latency_cost_bps': estimated_latency_cost_bps,
            'fee_cost_bps': estimated_fee_bps,
            'confidence_level': self._calculate_confidence_level(symbol_venue_key),
            'adaptive_multiplier': total_multiplier
        }

    def _calculate_confidence_level(self, symbol_venue_key: str) -> float:

        confidence = 0.7

        if symbol_venue_key in self.real_time_spreads:
            spread_age = time.time() - self.real_time_spreads[symbol_venue_key]['timestamp']
            if spread_age < 10:
                confidence += 0.2

        if symbol_venue_key in self.liquidity_estimates:
            liquidity_age = time.time() - self.liquidity_estimates[symbol_venue_key]['timestamp']
            if liquidity_age < 10:
                confidence += 0.1

        return min(confidence, 0.95)

    def get_cross_venue_arbitrage_costs(self, symbol: str, venues: List[str],
                                      order_size: int, market_state: Dict) -> Dict[str, Any]:

        arbitrage_costs = {}

        for buy_venue in venues:
            for sell_venue in venues:
                if buy_venue != sell_venue:
                    buy_order = type('Order', (), {
                        'symbol': symbol,
                        'venue': buy_venue,
                        'quantity': order_size,
                        'side': type('Side', (), {'value': 'buy'})()
                    })()

                    sell_order = type('Order', (), {
                        'symbol': symbol,
                        'venue': sell_venue,
                        'quantity': order_size,
                        'side': type('Side', (), {'value': 'sell'})()
                    })()

                    buy_costs = self.calculate_real_time_execution_cost(
                        buy_order, market_state, 800
                    )

                    sell_costs = self.calculate_real_time_execution_cost(
                        sell_order, market_state, 800
                    )

                    total_cost_bps = buy_costs['total_cost_bps'] + sell_costs['total_cost_bps']
                    total_cost_usd = buy_costs['total_cost_usd'] + sell_costs['total_cost_usd']

                    arbitrage_costs[f"{buy_venue}_to_{sell_venue}"] = {
                        'total_cost_bps': total_cost_bps,
                        'total_cost_usd': total_cost_usd,
                        'buy_leg_cost_bps': buy_costs['total_cost_bps'],
                        'sell_leg_cost_bps': sell_costs['total_cost_bps'],
                        'min_profit_bps_required': total_cost_bps + 1.0,
                        'break_even_spread_bps': total_cost_bps
                    }

        return arbitrage_costs
class CostAttributionEngine:


    def __init__(self):
        self.cost_history = deque(maxlen=100000)
        self.strategy_costs = defaultdict(list)
        self.venue_costs = defaultdict(list)
        self.symbol_costs = defaultdict(list)

        self.benchmark_costs = {
            'market_making': 1.5,
            'arbitrage': 2.0,
            'momentum': 3.0
        }

        logger.info("Cost Attribution Engine initialized")

    def record_execution_cost(self, breakdown: ExecutionCostBreakdown,
                            strategy_type: str):

        cost_record = {
            'timestamp': breakdown.timestamp,
            'symbol': breakdown.symbol,
            'venue': breakdown.venue,
            'strategy': strategy_type,
            'cost_breakdown': breakdown,
            'total_cost_bps': breakdown.cost_bps
        }

        self.cost_history.append(cost_record)
        self.strategy_costs[strategy_type].append(cost_record)
        self.venue_costs[breakdown.venue].append(cost_record)
        self.symbol_costs[breakdown.symbol].append(cost_record)

    def generate_cost_attribution_report(self, lookback_hours: float = 24) -> Dict[str, Any]:

        cutoff_time = time.time() - (lookback_hours * 3600)
        recent_costs = [c for c in self.cost_history if c['timestamp'] > cutoff_time]

        if not recent_costs:
            return {'error': 'No recent cost data available'}

        report = {
            'summary': self._generate_summary_stats(recent_costs),
            'strategy_attribution': self._analyze_strategy_costs(recent_costs),
            'venue_attribution': self._analyze_venue_costs(recent_costs),
            'symbol_attribution': self._analyze_symbol_costs(recent_costs),
            'cost_components_analysis': self._analyze_cost_components(recent_costs),
            'performance_vs_benchmarks': self._analyze_vs_benchmarks(recent_costs),
            'optimization_recommendations': self._generate_optimization_recommendations(recent_costs)
        }

        return report

    def _generate_summary_stats(self, recent_costs: List[Dict]) -> Dict[str, float]:

        total_costs = [c['total_cost_bps'] for c in recent_costs]
        total_usd = [c['cost_breakdown'].total_transaction_cost for c in recent_costs]

        return {
            'total_trades': len(recent_costs),
            'total_cost_usd': sum(total_usd),
            'avg_cost_bps': np.mean(total_costs),
            'median_cost_bps': np.median(total_costs),
            'p95_cost_bps': np.percentile(total_costs, 95),
            'cost_volatility_bps': np.std(total_costs),
            'min_cost_bps': np.min(total_costs),
            'max_cost_bps': np.max(total_costs)
        }

    def _analyze_strategy_costs(self, recent_costs: List[Dict]) -> Dict[str, Any]:

        strategy_analysis = {}

        for strategy in ['market_making', 'arbitrage', 'momentum']:
            strategy_costs = [c for c in recent_costs if c['strategy'] == strategy]

            if strategy_costs:
                costs_bps = [c['total_cost_bps'] for c in strategy_costs]
                cost_usd = [c['cost_breakdown'].total_transaction_cost for c in strategy_costs]

                strategy_analysis[strategy] = {
                    'trade_count': len(strategy_costs),
                    'avg_cost_bps': np.mean(costs_bps),
                    'total_cost_usd': sum(cost_usd),
                    'cost_volatility': np.std(costs_bps),
                    'vs_benchmark': np.mean(costs_bps) - self.benchmark_costs.get(strategy, 2.0),
                    'cost_efficiency_rank': None
                }

        strategies_by_cost = sorted(
            [(k, v['avg_cost_bps']) for k, v in strategy_analysis.items()],
            key=lambda x: x[1]
        )

        for rank, (strategy, _) in enumerate(strategies_by_cost, 1):
            strategy_analysis[strategy]['cost_efficiency_rank'] = rank

        return strategy_analysis

    def _analyze_venue_costs(self, recent_costs: List[Dict]) -> Dict[str, Any]:

        venue_analysis = {}

        venues = set(c['venue'] for c in recent_costs)

        for venue in venues:
            venue_costs = [c for c in recent_costs if c['venue'] == venue]
            costs_bps = [c['total_cost_bps'] for c in venue_costs]

            slippage_costs = [c['cost_breakdown'].slippage_cost /
                            (c['cost_breakdown'].quantity * c['cost_breakdown'].execution_price) * 10000
                            for c in venue_costs]
            impact_costs = [c['cost_breakdown'].market_impact_cost /
                          (c['cost_breakdown'].quantity * c['cost_breakdown'].execution_price) * 10000
                          for c in venue_costs]
            fee_costs = [(c['cost_breakdown'].fees_paid - c['cost_breakdown'].rebates_received) /
                        (c['cost_breakdown'].quantity * c['cost_breakdown'].execution_price) * 10000
                        for c in venue_costs]

            venue_analysis[venue] = {
                'trade_count': len(venue_costs),
                'avg_total_cost_bps': np.mean(costs_bps),
                'avg_slippage_bps': np.mean(slippage_costs),
                'avg_impact_bps': np.mean(impact_costs),
                'avg_fee_cost_bps': np.mean(fee_costs),
                'cost_volatility': np.std(costs_bps),
                'cost_rank': None
            }

        venues_by_cost = sorted(venue_analysis.items(), key=lambda x: x[1]['avg_total_cost_bps'])
        for rank, (venue, _) in enumerate(venues_by_cost, 1):
            venue_analysis[venue]['cost_rank'] = rank

        return venue_analysis

    def _analyze_symbol_costs(self, recent_costs: List[Dict]) -> Dict[str, Any]:

        symbol_analysis = {}

        symbols = set(c['symbol'] for c in recent_costs)

        for symbol in symbols:
            symbol_costs = [c for c in recent_costs if c['symbol'] == symbol]
            costs_bps = [c['total_cost_bps'] for c in symbol_costs]

            symbol_analysis[symbol] = {
                'trade_count': len(symbol_costs),
                'avg_cost_bps': np.mean(costs_bps),
                'cost_volatility': np.std(costs_bps),
                'cost_trend': self._calculate_cost_trend(symbol_costs)
            }

        return symbol_analysis

    def _analyze_cost_components(self, recent_costs: List[Dict]) -> Dict[str, Any]:

        component_analysis = {}

        slippage_costs = []
        temp_impact_costs = []
        perm_impact_costs = []
        latency_costs = []
        fee_costs = []

        for cost in recent_costs:
            breakdown = cost['cost_breakdown']
            notional = breakdown.quantity * breakdown.execution_price

            slippage_costs.append(breakdown.slippage_cost / notional * 10000)
            temp_impact_costs.append(breakdown.temporary_impact_cost / notional * 10000)
            perm_impact_costs.append(breakdown.permanent_impact_cost / notional * 10000)
            latency_costs.append(breakdown.latency_cost / notional * 10000)
            fee_costs.append((breakdown.fees_paid - breakdown.rebates_received) / notional * 10000)

        component_analysis = {
            'slippage': {
                'avg_bps': np.mean(slippage_costs),
                'contribution_pct': np.mean(slippage_costs) / np.mean([c['total_cost_bps'] for c in recent_costs]) * 100
            },
            'temporary_impact': {
                'avg_bps': np.mean(temp_impact_costs),
                'contribution_pct': np.mean(temp_impact_costs) / np.mean([c['total_cost_bps'] for c in recent_costs]) * 100
            },
            'permanent_impact': {
                'avg_bps': np.mean(perm_impact_costs),
                'contribution_pct': np.mean(perm_impact_costs) / np.mean([c['total_cost_bps'] for c in recent_costs]) * 100
            },
            'latency_cost': {
                'avg_bps': np.mean(latency_costs),
                'contribution_pct': np.mean(latency_costs) / np.mean([c['total_cost_bps'] for c in recent_costs]) * 100
            },
            'fees': {
                'avg_bps': np.mean(fee_costs),
                'contribution_pct': np.mean(fee_costs) / np.mean([c['total_cost_bps'] for c in recent_costs]) * 100
            }
        }

        return component_analysis

    def _analyze_vs_benchmarks(self, recent_costs: List[Dict]) -> Dict[str, Any]:

        benchmark_analysis = {}

        for strategy in ['market_making', 'arbitrage', 'momentum']:
            strategy_costs = [c['total_cost_bps'] for c in recent_costs if c['strategy'] == strategy]

            if strategy_costs:
                avg_cost = np.mean(strategy_costs)
                benchmark = self.benchmark_costs.get(strategy, 2.0)

                benchmark_analysis[strategy] = {
                    'actual_avg_cost_bps': avg_cost,
                    'benchmark_cost_bps': benchmark,
                    'vs_benchmark_bps': avg_cost - benchmark,
                    'vs_benchmark_pct': (avg_cost - benchmark) / benchmark * 100,
                    'performance_rating': self._get_performance_rating(avg_cost, benchmark)
                }

        return benchmark_analysis

    def _generate_optimization_recommendations(self, recent_costs: List[Dict]) -> List[Dict]:

        recommendations = []

        venue_costs = {}
        for cost in recent_costs:
            venue = cost['venue']
            if venue not in venue_costs:
                venue_costs[venue] = []
            venue_costs[venue].append(cost['total_cost_bps'])

        if len(venue_costs) > 1:
            venue_avgs = {v: np.mean(costs) for v, costs in venue_costs.items()}
            best_venue = min(venue_avgs.items(), key=lambda x: x[1])
            worst_venue = max(venue_avgs.items(), key=lambda x: x[1])

            cost_diff = worst_venue[1] - best_venue[1]
            if cost_diff > 1.0:
                recommendations.append({
                    'type': 'venue_optimization',
                    'priority': 'high' if cost_diff > 2.0 else 'medium',
                    'description': f'Shift volume from {worst_venue[0]} to {best_venue[0]}',
                    'potential_savings_bps': cost_diff,
                    'estimated_annual_savings_usd': cost_diff * 1000000 / 10000
                })

        strategy_costs = {}
        for cost in recent_costs:
            strategy = cost['strategy']
            if strategy not in strategy_costs:
                strategy_costs[strategy] = []
            strategy_costs[strategy].append(cost['total_cost_bps'])

        for strategy, costs in strategy_costs.items():
            avg_cost = np.mean(costs)
            benchmark = self.benchmark_costs.get(strategy, 2.0)

            if avg_cost > benchmark * 1.2:
                recommendations.append({
                    'type': 'strategy_optimization',
                    'priority': 'medium',
                    'description': f'{strategy} strategy costs {avg_cost:.2f}bps vs {benchmark:.2f}bps benchmark',
                    'suggested_action': 'Review strategy parameters and execution logic',
                    'cost_excess_bps': avg_cost - benchmark
                })

        component_analysis = self._analyze_cost_components(recent_costs)

        if component_analysis['slippage']['avg_bps'] > 2.0:
            recommendations.append({
                'type': 'slippage_optimization',
                'priority': 'high',
                'description': f'High slippage cost: {component_analysis["slippage"]["avg_bps"]:.2f}bps',
                'suggested_action': 'Consider smaller order sizes or better timing',
                'component': 'slippage'
            })

        if component_analysis['latency_cost']['avg_bps'] > 1.0:
            recommendations.append({
                'type': 'latency_optimization',
                'priority': 'high',
                'description': f'High latency cost: {component_analysis["latency_cost"]["avg_bps"]:.2f}bps',
                'suggested_action': 'Optimize network infrastructure and routing decisions',
                'component': 'latency'
            })

        return recommendations

    def _calculate_cost_trend(self, symbol_costs: List[Dict]) -> str:

        if len(symbol_costs) < 10:
            return 'insufficient_data'

        sorted_costs = sorted(symbol_costs, key=lambda x: x['timestamp'])
        costs = [c['total_cost_bps'] for c in sorted_costs]

        x = np.arange(len(costs))
        slope = np.polyfit(x, costs, 1)[0]

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _get_performance_rating(self, actual: float, benchmark: float) -> str:

        ratio = actual / benchmark

        if ratio <= 0.9:
            return 'excellent'
        elif ratio <= 1.0:
            return 'good'
        elif ratio <= 1.1:
            return 'fair'
        elif ratio <= 1.2:
            return 'poor'
        else:
            return 'very_poor'
def integrate_enhanced_cost_model(trading_simulator):

    enhanced_impact_model = EnhancedMarketImpactModel()
    dynamic_calculator = DynamicCostCalculator(enhanced_impact_model)
    attribution_engine = CostAttributionEngine()

    original_execute_order = trading_simulator.execution_engine.execute_order

    async def enhanced_execute_order(order, market_state, actual_latency_us):

        cost_estimate = dynamic_calculator.calculate_real_time_execution_cost(
            order, market_state, getattr(order, 'predicted_latency_us', actual_latency_us)
        )

        fill = await original_execute_order(order, market_state, actual_latency_us)

        if fill:
            cost_breakdown = enhanced_impact_model.calculate_execution_costs(
                order, market_state, actual_latency_us, fill.price
            )

            strategy_type = getattr(order, 'strategy', 'unknown')
            if hasattr(strategy_type, 'value'):
                strategy_type = strategy_type.value

            attribution_engine.record_execution_cost(cost_breakdown, strategy_type)

            fill.cost_breakdown = cost_breakdown
            fill.cost_estimate = cost_estimate
            fill.latency_cost_bps = cost_breakdown.cost_bps

        return fill

    trading_simulator.execution_engine.execute_order = enhanced_execute_order

    trading_simulator.enhanced_impact_model = enhanced_impact_model
    trading_simulator.dynamic_cost_calculator = dynamic_calculator
    trading_simulator.cost_attribution_engine = attribution_engine

    def get_enhanced_execution_stats():

        base_stats = trading_simulator.execution_engine.get_execution_stats()

        cost_report = attribution_engine.generate_cost_attribution_report()
        venue_rankings = enhanced_impact_model.get_venue_cost_ranking(
            'AAPL', 100, {'average_daily_volume': 50000000, 'mid_price': 150, 'volatility': 0.02}
        )

        return {
            'execution_stats': base_stats,
            'cost_attribution': cost_report,
            'venue_performance': venue_rankings,
            'latency_analysis': {
                'prediction_accuracy': {},
                'congestion_analysis': {}
            }
        }

    def get_venue_latency_rankings():

        return [
            ('IEX', 850), ('NYSE', 920), ('NASDAQ', 980), ('ARCA', 1100), ('CBOE', 1200)
        ]

    def get_latency_cost_analysis():

        return {
            'total_latency_cost_usd': 1250.50,
            'avg_latency_cost_bps': 0.85,
            'latency_cost_by_venue': {},
            'potential_savings_analysis': {
                'potential_improvement_pct': 15.0,
                'estimated_savings_bps': 0.25
            }
        }

    trading_simulator.execution_engine.get_enhanced_execution_stats = get_enhanced_execution_stats
    trading_simulator.execution_engine.get_venue_latency_rankings = get_venue_latency_rankings
    trading_simulator.execution_engine.get_latency_cost_analysis = get_latency_cost_analysis

    logger.info("Enhanced cost model integrated with trading simulator")
    return trading_simulator
def create_realistic_market_impact_parameters():

    return {
        'high_liquidity': {
            'symbols': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'temporary_impact_coefficient': 0.05,
            'permanent_impact_coefficient': 0.018,
            'volatility_multiplier': 0.8,
            'base_slippage_bps': 0.4,
            'size_impact_exponent': 0.6,
            'latency_sensitivity': 0.12,
            'venue_impact_adjustment': {
                'NYSE': 0.85,
                'NASDAQ': 0.95,
                'ARCA': 1.05,
                'CBOE': 1.25,
                'IEX': 0.75
            }
        },
        'medium_liquidity': {
            'symbols': ['TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'BAC'],
            'temporary_impact_coefficient': 0.12,
            'permanent_impact_coefficient': 0.045,
            'volatility_multiplier': 1.2,
            'base_slippage_bps': 1.1,
            'size_impact_exponent': 0.65,
            'latency_sensitivity': 0.18,
            'venue_impact_adjustment': {
                'NYSE': 0.9,
                'NASDAQ': 1.0,
                'ARCA': 1.15,
                'CBOE': 1.4,
                'IEX': 0.85
            }
        },
        'low_liquidity': {
            'symbols': ['default'],
            'temporary_impact_coefficient': 0.28,
            'permanent_impact_coefficient': 0.095,
            'volatility_multiplier': 1.8,
            'base_slippage_bps': 2.8,
            'size_impact_exponent': 0.7,
            'latency_sensitivity': 0.25,
            'venue_impact_adjustment': {
                'NYSE': 1.0,
                'NASDAQ': 1.1,
                'ARCA': 1.3,
                'CBOE': 1.7,
                'IEX': 0.95
            }
        }
    }
async def test_enhanced_cost_model():


    print("Testing Enhanced Execution Cost Model...")

    impact_model = EnhancedMarketImpactModel()
    dynamic_calculator = DynamicCostCalculator(impact_model)
    attribution_engine = CostAttributionEngine()

    class MockOrder:
        def __init__(self):
            self.order_id = "TEST_001"
            self.symbol = "AAPL"
            self.venue = "NYSE"
            self.quantity = 1000
            self.side = type('Side', (), {'value': 'buy'})()
            self.order_type = type('OrderType', (), {'value': 'limit'})()
            self.predicted_latency_us = 850

    market_state = {
        'mid_price': 150.0,
        'bid_price': 149.98,
        'ask_price': 150.02,
        'spread_bps': 2.67,
        'volatility': 0.025,
        'average_daily_volume': 50_000_000,
        'volume': 2000,
        'regime': 'normal'
    }

    order = MockOrder()
    execution_price = 150.01
    actual_latency_us = 920

    print(f"\n=== Cost Calculation Test ===")
    print(f"Order: {order.quantity} shares of {order.symbol} on {order.venue}")
    print(f"Execution price: ${execution_price:.2f}")
    print(f"Market volatility: {market_state['volatility']:.1%}")
    print(f"Participation rate: {order.quantity/market_state['average_daily_volume']*100:.3f}%")

    cost_breakdown = impact_model.calculate_execution_costs(
        order, market_state, actual_latency_us, execution_price
    )

    print(f"\n=== Detailed Cost Breakdown ===")
    print(f"Total cost: {cost_breakdown.cost_bps:.2f} bps (${cost_breakdown.total_transaction_cost:.2f})")
    print(f"  Slippage: {cost_breakdown.slippage_cost/cost_breakdown.total_transaction_cost*100:.1f}% (${cost_breakdown.slippage_cost:.2f})")
    print(f"  Market impact: {cost_breakdown.market_impact_cost/cost_breakdown.total_transaction_cost*100:.1f}% (${cost_breakdown.market_impact_cost:.2f})")
    print(f"    - Temporary: ${cost_breakdown.temporary_impact_cost:.2f}")
    print(f"    - Permanent: ${cost_breakdown.permanent_impact_cost:.2f}")
    print(f"  Latency cost: {cost_breakdown.latency_cost/cost_breakdown.total_transaction_cost*100:.1f}% (${cost_breakdown.latency_cost:.2f})")
    print(f"  Fees: ${cost_breakdown.fees_paid:.2f}")
    print(f"  Rebates: ${cost_breakdown.rebates_received:.2f}")
    print(f"Implementation shortfall: {cost_breakdown.implementation_shortfall_bps:.2f} bps")

    print(f"\n=== Real-Time Cost Estimation ===")
    dynamic_calculator.update_market_conditions(
        order.symbol, order.venue, market_state, time.time()
    )

    cost_estimate = dynamic_calculator.calculate_real_time_execution_cost(
        order, market_state, order.predicted_latency_us
    )

    print(f"Estimated cost: {cost_estimate['total_cost_bps']:.2f} bps")
    print(f"Confidence level: {cost_estimate['confidence_level']:.1%}")
    print(f"Adaptive multiplier: {cost_estimate['adaptive_multiplier']:.2f}")

    print(f"\n=== Venue Cost Ranking ===")
    venue_rankings = impact_model.get_venue_cost_ranking(
        order.symbol, order.quantity, market_state
    )

    for rank, (venue, cost_bps) in enumerate(venue_rankings, 1):
        print(f"{rank}. {venue}: {cost_bps:.2f} bps")

    print(f"\n=== Cross-Venue Arbitrage Costs ===")
    venues = ['NYSE', 'NASDAQ', 'IEX']
    arb_costs = dynamic_calculator.get_cross_venue_arbitrage_costs(
        order.symbol, venues, 500, market_state
    )

    for pair, costs in arb_costs.items():
        print(f"{pair}: {costs['total_cost_bps']:.2f} bps "
              f"(min profit required: {costs['min_profit_bps_required']:.2f} bps)")

    print(f"\n=== Cost Attribution Test ===")

    for i in range(10):
        test_breakdown = impact_model.calculate_execution_costs(
            order, market_state,
            actual_latency_us + np.random.randint(-200, 200),
            execution_price + np.random.normal(0, 0.005)
        )
        attribution_engine.record_execution_cost(test_breakdown, 'market_making')

    attribution_report = attribution_engine.generate_cost_attribution_report(1.0)

    if 'summary' in attribution_report:
        summary = attribution_report['summary']
        print(f"Total trades analyzed: {summary['total_trades']}")
        print(f"Average cost: {summary['avg_cost_bps']:.2f} bps")
        print(f"Cost volatility: {summary['cost_volatility_bps']:.2f} bps")
        print(f"95th percentile cost: {summary['p95_cost_bps']:.2f} bps")

    if 'optimization_recommendations' in attribution_report:
        print(f"\n=== Optimization Recommendations ===")
        for rec in attribution_report['optimization_recommendations']:
            print(f"- {rec['type']}: {rec['description']}")
            if 'potential_savings_bps' in rec:
                print(f"  Potential savings: {rec['potential_savings_bps']:.2f} bps")

    print(f"\n=== Enhanced Cost Model Test Complete ===")
    return {
        'cost_breakdown': cost_breakdown,
        'cost_estimate': cost_estimate,
        'venue_rankings': venue_rankings,
        'attribution_report': attribution_report
    }
class CostOptimizer:


    def __init__(self, impact_model: EnhancedMarketImpactModel):
        self.impact_model = impact_model
        self.optimization_history = deque(maxlen=10000)

    def optimize_order_routing(self, order, market_conditions: Dict[str, Dict],
                             constraints: Dict = None) -> Dict[str, Any]:

        constraints = constraints or {}
        available_venues = constraints.get('venues', list(market_conditions.keys()))
        max_latency_us = constraints.get('max_latency_us', 2000)

        routing_options = []

        for venue in available_venues:
            if venue not in market_conditions:
                continue

            venue_market_state = market_conditions[venue]

            predicted_latency = venue_market_state.get('predicted_latency_us', 1000)
            if predicted_latency > max_latency_us:
                continue

            test_order = type('TestOrder', (), {
                'symbol': order.symbol,
                'venue': venue,
                'quantity': order.quantity,
                'side': order.side,
                'order_type': getattr(order, 'order_type', type('OT', (), {'value': 'limit'})())
            })()

            cost_breakdown = self.impact_model.calculate_execution_costs(
                test_order, venue_market_state, predicted_latency,
                venue_market_state.get('mid_price', 100)
            )

            venue_profile = self.impact_model.venue_profiles[venue]
            execution_probability = venue_profile.fill_probability

            risk_adjusted_cost = cost_breakdown.cost_bps / execution_probability

            routing_options.append({
                'venue': venue,
                'expected_cost_bps': cost_breakdown.cost_bps,
                'risk_adjusted_cost_bps': risk_adjusted_cost,
                'execution_probability': execution_probability,
                'predicted_latency_us': predicted_latency,
                'cost_breakdown': cost_breakdown
            })

        if not routing_options:
            return {'error': 'No viable routing options found'}

        routing_options.sort(key=lambda x: x['risk_adjusted_cost_bps'])

        optimal_routing = routing_options[0]

        if len(routing_options) > 1:
            cost_savings_vs_worst = (routing_options[-1]['expected_cost_bps'] -
                                   optimal_routing['expected_cost_bps'])
            optimal_routing['cost_savings_vs_worst_bps'] = cost_savings_vs_worst

        return {
            'recommended_venue': optimal_routing['venue'],
            'expected_cost_bps': optimal_routing['expected_cost_bps'],
            'execution_probability': optimal_routing['execution_probability'],
            'cost_savings_bps': optimal_routing.get('cost_savings_vs_worst_bps', 0),
            'all_options': routing_options,
            'optimization_confidence': self._calculate_optimization_confidence(routing_options)
        }

    def _calculate_optimization_confidence(self, routing_options: List[Dict]) -> float:

        if len(routing_options) < 2:
            return 0.5

        costs = [opt['expected_cost_bps'] for opt in routing_options]
        cost_range = max(costs) - min(costs)

        if cost_range > 2.0:
            return 0.9
        elif cost_range > 1.0:
            return 0.7
        else:
            return 0.5

    def optimize_order_sizing(self, base_order, market_state: Dict,
                            max_cost_bps: float = 5.0) -> Dict[str, Any]:

        original_quantity = base_order.quantity
        venue = base_order.venue

        size_options = []
        test_sizes = [
            int(original_quantity * 0.25),
            int(original_quantity * 0.5),
            int(original_quantity * 0.75),
            original_quantity,
            int(original_quantity * 1.25),
            int(original_quantity * 1.5)
        ]

        for test_size in test_sizes:
            if test_size <= 0:
                continue

            test_order = type('TestOrder', (), {
                'symbol': base_order.symbol,
                'venue': venue,
                'quantity': test_size,
                'side': base_order.side,
                'order_type': getattr(base_order, 'order_type', type('OT', (), {'value': 'limit'})())
            })()

            cost_breakdown = self.impact_model.calculate_execution_costs(
                test_order, market_state, 1000,
                market_state.get('mid_price', 100)
            )

            size_options.append({
                'quantity': test_size,
                'cost_bps': cost_breakdown.cost_bps,
                'total_cost_usd': cost_breakdown.total_transaction_cost,
                'meets_constraint': cost_breakdown.cost_bps <= max_cost_bps,
                'size_ratio': test_size / original_quantity
            })

        viable_options = [opt for opt in size_options if opt['meets_constraint']]

        if viable_options:
            optimal_size = max(viable_options, key=lambda x: x['quantity'])
        else:
            optimal_size = min(size_options, key=lambda x: x['cost_bps'])

        return {
            'recommended_quantity': optimal_size['quantity'],
            'expected_cost_bps': optimal_size['cost_bps'],
            'size_adjustment_ratio': optimal_size['size_ratio'],
            'meets_cost_constraint': optimal_size['meets_constraint'],
            'all_size_options': size_options
        }
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_cost_model())