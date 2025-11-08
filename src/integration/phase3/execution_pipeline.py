"""Production execution pipeline."""

import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ProductionExecutionPipeline:
    """
    Integrated execution pipeline combining all phases - COMPLETE REPLACEMENT
    """

    def __init__(
        self,
        market_generator,
        network_simulator,
        order_book_manager,
        feature_extractor,
        latency_predictor,
        ensemble_model,
        routing_environment,
        market_regime_detector,
        trading_simulator,
        risk_manager,
        pnl_attribution,
    ):

        # Store all components
        self.market_generator = market_generator
        self.network_simulator = network_simulator
        self.order_book_manager = order_book_manager
        self.feature_extractor = feature_extractor
        self.latency_predictor = latency_predictor
        self.ensemble_model = ensemble_model
        self.routing_environment = routing_environment
        self.market_regime_detector = market_regime_detector
        self.trading_simulator = trading_simulator
        self.risk_manager = risk_manager
        self.pnl_attribution = pnl_attribution

        self.halt_trading = False

        # Initialize tracking variables for realistic trading
        self._last_arb_trade_time = 0
        self._last_trade_times = {}

    async def generate_trading_signals(self, tick, market_features, current_regime) -> List[Dict]:

        if self.halt_trading:
            return []

        signals = []

        # Get total stock count for scaling
        total_stocks = len(getattr(self.market_generator, "symbols", ["AAPL", "MSFT", "GOOGL"]))

        # ARBITRAGE: More stocks = more cross-venue opportunities
        if len(self.market_generator.arbitrage_opportunities) > 0:
            time_since_last_arb = tick.timestamp - self._last_arb_trade_time

            # Scale arbitrage frequency with stock count
            arb_interval = max(2.0, 8.0 / (total_stocks / 10))  # More stocks = more frequent arb

            if time_since_last_arb > arb_interval:
                arb_opp = self.market_generator.arbitrage_opportunities.popleft()

                if arb_opp["profit_per_share"] > 0.05:  # Lower threshold with more opportunities
                    signals.append(
                        {
                            "strategy": "arbitrage",
                            "symbol": arb_opp["symbol"],
                            "arbitrage_type": "cross_venue",
                            "buy_venue": arb_opp["buy_venue"],
                            "sell_venue": arb_opp["sell_venue"],
                            "buy_price": arb_opp["buy_price"],
                            "sell_price": arb_opp["sell_price"],
                            "quantity": min(100, arb_opp["max_size"]),
                            "urgency": 0.95,
                            "expected_pnl": arb_opp["profit_per_share"]
                            * min(100, arb_opp["max_size"]),
                            "confidence": 0.95,
                            "timestamp": arb_opp["timestamp"],
                        }
                    )

                    self._last_arb_trade_time = tick.timestamp
                    logger.info(
                        f" MULTI-STOCK ARBITRAGE: {arb_opp['symbol']} profit=${arb_opp['profit_per_share']:.3f}"
                    )
                    return signals

        # MARKET MAKING: Scale signal rate with stock count
        base_signal_rate = min(0.05, 0.8 / total_stocks)  # More stocks = more total opportunities

        # Stock-specific multipliers for different trading patterns
        symbol = tick.symbol
        stock_multipliers = {
            "SPY": 4.0,  # ETFs trade most frequently
            "QQQ": 3.5,
            "IWM": 2.5,
            "TSLA": 3.0,  # High volatility stocks
            "NVDA": 2.8,
            "META": 2.5,
            "AAPL": 2.0,  # Blue chips
            "MSFT": 2.0,
            "GOOGL": 1.8,
            "AMZN": 1.8,
            "JPM": 1.5,  # Financials
            "BAC": 1.3,
            "JNJ": 1.0,  # Defensive stocks
            "PG": 0.8,
            "GLD": 0.6,  # Commodities/bonds (less active)
            "TLT": 0.5,
        }

        # Cooldown check
        last_trade_time = self._last_trade_times.get(symbol, 0)
        time_since_last_trade = tick.timestamp - last_trade_time

        cooldown_time = 20 / stock_multipliers.get(
            symbol, 1.0
        )  # High-activity stocks have shorter cooldowns
        if time_since_last_trade < cooldown_time:
            return []

        # Regime adjustments
        regime_multipliers = {
            "volatile": 0.3,  # Fewer signals in volatile markets
            "quiet": 2.5,  # More signals when calm
            "normal": 1.0,
            "active": 1.8,
        }

        final_signal_rate = (
            base_signal_rate
            * stock_multipliers.get(symbol, 1.0)
            * regime_multipliers.get(current_regime, 1.0)
        )

        # Generate signal
        if np.random.random() < final_signal_rate:

            spread = getattr(tick, "spread", 0.02)
            mid_price = getattr(tick, "mid_price", 100.0)

            # Stock-specific P&L expectations
            if symbol in ["SPY", "QQQ", "IWM"]:
                # ETFs: More predictable, higher frequency
                expected_pnl = np.random.uniform(8, 35)
                confidence = np.random.uniform(0.5, 0.8)
            elif symbol in ["TSLA", "NVDA", "META"]:
                # High volatility: Higher risk/reward
                expected_pnl = np.random.uniform(-15, 80)
                confidence = np.random.uniform(0.3, 0.6)
            elif symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
                # Large cap tech: Solid opportunities
                expected_pnl = np.random.uniform(-8, 50)
                confidence = np.random.uniform(0.4, 0.7)
            elif symbol in ["JPM", "BAC", "WFC", "GS", "C"]:
                # Financials: Moderate volatility
                expected_pnl = np.random.uniform(-5, 35)
                confidence = np.random.uniform(0.4, 0.6)
            elif symbol in ["JNJ", "PG", "KO"]:
                # Defensive: Lower volatility, smaller moves
                expected_pnl = np.random.uniform(3, 20)
                confidence = np.random.uniform(0.5, 0.7)
            elif symbol in ["GLD", "TLT"]:
                # Alternative assets: Different patterns
                expected_pnl = np.random.uniform(1, 18)
                confidence = np.random.uniform(0.4, 0.6)
            else:
                # Default
                expected_pnl = np.random.uniform(-10, 40)
                confidence = np.random.uniform(0.3, 0.7)

            # Occasional negative expectation (realistic)
            if np.random.random() < 0.4:  # 40% of trades expected to lose
                expected_pnl = -abs(np.random.uniform(5, 25))

            signals.append(
                {
                    "strategy": "market_making",
                    "symbol": symbol,
                    "side": "buy" if np.random.random() < 0.5 else "sell",
                    "quantity": 100,
                    "urgency": np.random.uniform(0.2, 0.7),
                    "expected_pnl": expected_pnl,
                    "confidence": confidence,
                    "stock_type": self._classify_stock_type(symbol),
                }
            )

            self._last_trade_times[symbol] = tick.timestamp
            logger.info(
                f" {symbol} signal: expected_pnl=${expected_pnl:.2f} (type: {self._classify_stock_type(symbol)})"
            )

        return signals

    def _classify_stock_type(self, symbol):
        """Classify stock type for better signal generation"""
        if symbol in ["SPY", "QQQ", "IWM", "GLD", "TLT"]:
            return "ETF"
        elif symbol in ["TSLA", "NVDA", "META"]:
            return "HIGH_VOL_TECH"
        elif symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            return "LARGE_CAP_TECH"
        elif symbol in ["JPM", "BAC", "GS", "C", "WFC"]:
            return "FINANCIAL"
        elif symbol in ["JNJ", "PG", "KO"]:
            return "DEFENSIVE"
        else:
            return "OTHER"
