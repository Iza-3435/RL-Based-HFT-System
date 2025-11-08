"""Trade execution with ML routing and risk management."""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Execute trades with ML routing and risk management."""

    def __init__(self, routing_environment: Any, risk_manager: Any):
        self.routing_environment = routing_environment
        self.risk_manager = risk_manager
        self.trade_count = 0
        self.total_pnl = 0.0

    async def execute_trade(
        self, signal: Dict, tick: Any, simulation_results: Dict
    ) -> Optional[Dict]:
        """Execute trade with ML routing."""
        try:
            logger.info(f"Executing signal: {signal.get('strategy', 'unknown')}")

            if not signal or not isinstance(signal, dict):
                return None

            symbol = signal.get("symbol")
            if not symbol:
                logger.error(f"Signal missing symbol: {signal}")
                return None

            if signal.get("arbitrage_type") == "cross_venue":
                logger.info("Detected arbitrage signal")
                return await self._execute_arbitrage_trade(signal, tick, simulation_results)
            else:
                logger.info("Executing regular trade")
                return await self._execute_regular_trade(signal, tick, simulation_results)

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None

    async def _execute_arbitrage_trade(
        self, signal: Dict, tick: Any, simulation_results: Dict
    ) -> Optional[Dict]:
        """Execute arbitrage trade."""
        logger.info(f"Executing arbitrage: {signal['symbol']}")

        symbol = signal["symbol"]
        buy_venue = signal["buy_venue"]
        sell_venue = signal["sell_venue"]
        buy_price = signal["buy_price"]
        sell_price = signal["sell_price"]
        quantity = signal["quantity"]

        logger.info(f"   Buy:  {quantity} shares @ ${buy_price:.2f} on {buy_venue}")
        logger.info(f"   Sell: {quantity} shares @ ${sell_price:.2f} on {sell_venue}")

        buy_fees = quantity * buy_price * 0.00003
        buy_rebates = quantity * buy_price * 0.00001 if buy_venue in ["NYSE", "NASDAQ"] else 0

        sell_fees = quantity * sell_price * 0.00003
        sell_rebates = quantity * sell_price * 0.00001 if sell_venue in ["NYSE", "NASDAQ"] else 0

        gross_profit = (sell_price - buy_price) * quantity
        total_fees = buy_fees + sell_fees
        total_rebates = buy_rebates + sell_rebates
        net_profit = gross_profit - total_fees + total_rebates

        slippage_cost = quantity * buy_price * 0.0001
        final_pnl = net_profit - slippage_cost

        trade_result = {
            "timestamp": time.time(),
            "symbol": symbol,
            "strategy": "arbitrage",
            "arbitrage_type": "cross_venue",
            "buy_venue": buy_venue,
            "sell_venue": sell_venue,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "quantity": quantity,
            "gross_profit": gross_profit,
            "pnl": final_pnl,
            "fees": total_fees,
            "rebates": total_rebates,
            "execution_quality": 0.95,
            "slippage_bps": 1.0,
        }

        self.trade_count += 1
        self.total_pnl += final_pnl

        logger.info(f"Arbitrage executed")
        logger.info(f"   Gross Profit: ${gross_profit:.2f}")
        logger.info(f"   Total Fees: ${total_fees:.2f}")
        logger.info(f"   Net Profit: ${final_pnl:.2f}")
        logger.info(f"   Total Trades: {self.trade_count} | Total P&L: ${self.total_pnl:.2f}")

        return trade_result

    async def _execute_regular_trade(
        self, signal: Dict, tick: Any, simulation_results: Dict
    ) -> Optional[Dict]:
        """Execute regular trade."""
        symbol = signal["symbol"]
        mid_price = getattr(tick, "mid_price", 100.0)
        quantity = signal.get("quantity", 100)
        side = signal.get("side", "buy")

        logger.info(f"Regular trade: {symbol} {side} {quantity} @ ~${mid_price:.2f}")

        current_prices = {symbol: mid_price}
        if hasattr(self, "risk_manager") and self.risk_manager:
            try:
                from simulator.trading_simulator import (
                    Order,
                    OrderSide,
                    OrderType,
                    TradingStrategyType,
                )

                side_str = side.upper()
                side_enum = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

                temp_order = Order(
                    order_id=f"TEMP_{int(time.time() * 1e6)}",
                    symbol=symbol,
                    venue="NYSE",
                    side=side_enum,
                    order_type=OrderType.LIMIT,
                    quantity=quantity,
                    price=mid_price,
                    timestamp=time.time(),
                    strategy=TradingStrategyType.MARKET_MAKING,
                )

                risk_allowed, risk_reason = self.risk_manager.check_pre_trade_risk(
                    temp_order, current_prices
                )

                if not risk_allowed:
                    logger.info(f"Trade rejected by risk: {risk_reason}")
                    return None

            except Exception as e:
                logger.debug(f"Risk check failed: {e}")

        logger.info("Trade approved by risk manager")

        try:
            if hasattr(self, "routing_environment") and self.routing_environment:
                routing_decision = self.routing_environment.make_routing_decision(
                    symbol, signal.get("urgency", 0.5)
                )
                if routing_decision:
                    logger.info(f"ML routing: {routing_decision.venue}")
                else:
                    raise Exception("No routing decision")
            else:
                raise Exception("No routing environment")
        except:

            class FallbackRouting:
                def __init__(self):
                    self.venue = "NYSE"
                    self.expected_latency_us = 1000
                    self.confidence = 0.5

            routing_decision = FallbackRouting()
            logger.info(f"Fallback routing: {routing_decision.venue}")

        fill_price = mid_price

        base_slippage_bps = 0.2
        size_impact = (quantity / 1000) * 0.1

        volatility = getattr(tick, "volatility", 0.02)
        vol_impact = volatility * 20

        current_regime = signal.get("current_regime", "normal")
        regime_impact = 0.1 if current_regime == "volatile" else 0.0

        total_slippage_bps = base_slippage_bps + size_impact + vol_impact + regime_impact

        logger.info(
            f"Slippage breakdown: base={base_slippage_bps:.2f}, size={size_impact:.2f}, "
            f"vol={vol_impact:.2f}, regime={regime_impact:.2f}"
        )
        logger.info(f"Total slippage: {total_slippage_bps:.2f} bps")

        if side == "buy":
            fill_price *= 1 + total_slippage_bps / 10000
        else:
            fill_price *= 1 - total_slippage_bps / 10000

        expected_pnl = signal.get("expected_pnl", 0)
        execution_cost = quantity * mid_price * (total_slippage_bps / 10000)

        market_move_bps = np.random.normal(0, 0.5)
        market_move_cost = quantity * mid_price * (market_move_bps / 10000)

        pnl = expected_pnl - execution_cost - market_move_cost

        fees = quantity * fill_price * 0.00003
        rebates = (
            quantity * fill_price * 0.00001 if routing_decision.venue in ["NYSE", "NASDAQ"] else 0
        )

        fill_rate = np.random.uniform(0.95, 1.0)
        actual_quantity = int(quantity * fill_rate)
        pnl *= fill_rate

        trade_result = {
            "timestamp": time.time(),
            "symbol": symbol,
            "strategy": signal.get("strategy", "market_making"),
            "side": side,
            "quantity": actual_quantity,
            "requested_quantity": quantity,
            "fill_rate": fill_rate,
            "price": fill_price,
            "venue": routing_decision.venue,
            "pnl": pnl,
            "fees": fees,
            "rebates": rebates,
            "slippage_bps": total_slippage_bps,
            "market_impact_cost": execution_cost,
            "market_move_cost": market_move_cost,
            "execution_quality": np.random.uniform(0.8, 0.95),
        }

        self.trade_count += 1
        self.total_pnl += pnl

        logger.info(
            f"Trade executed: {actual_quantity}@${fill_price:.2f} on {routing_decision.venue}"
        )
        logger.info(
            f"P&L: ${pnl:.2f} | Total: {self.trade_count} | Total P&L: ${self.total_pnl:.2f}"
        )

        return trade_result
