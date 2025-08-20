import asyncio
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import time
from .professional_execution_display import (
    ProfessionalExecutionDisplay,
    LiveTrade,
    TradeStatus,
    Colors
)
class HFTDisplayManager:


    def __init__(self, enable_display: bool = True):
        self.enable_display = enable_display
        self.display: Optional[ProfessionalExecutionDisplay] = None
        self.trade_counter = 0

        if self.enable_display:
            self.display = ProfessionalExecutionDisplay()
            self.display.start_display()
            print(f"{Colors.GREEN}🚀 Professional execution display started!{Colors.RESET}")

    def log_trade_execution(self,
                          symbol: str,
                          venue: str,
                          side: str,
                          quantity: int,
                          entry_price: float,
                          current_price: float = None,
                          strategy: str = "UNKNOWN",
                          ml_confidence: float = None) -> str:

        if not self.enable_display:
            return ""

        self.trade_counter += 1
        trade_id = f"T{self.trade_counter:06d}"

        trade = LiveTrade(
            trade_id=trade_id,
            timestamp=time.time(),
            symbol=symbol.upper(),
            venue=venue.upper(),
            side=side.upper(),
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price or entry_price,
            strategy=strategy,
            ml_confidence=ml_confidence
        )

        self.display.add_trade(trade)
        return trade_id

    def update_trade_fill(self,
                         trade_id: str,
                         executed_price: float,
                         latency_us: float,
                         fees: float = 0.0,
                         fill_status: str = "FILLED"):

        if not self.enable_display or not trade_id:
            return

        if trade_id in self.display.active_trades:
            trade = self.display.active_trades[trade_id]

            slippage_bps = ((executed_price - trade.entry_price) / trade.entry_price) * 10000
            if trade.side.upper() == 'SELL':
                slippage_bps = -slippage_bps

            if trade.side.upper() == 'BUY':
                pnl = (executed_price - trade.entry_price) * trade.quantity - fees
            else:
                pnl = (trade.entry_price - executed_price) * trade.quantity - fees

            self.display.update_trade(
                trade_id,
                status=TradeStatus.FILLED if fill_status == "FILLED" else TradeStatus.PARTIAL,
                executed_price=executed_price,
                latency_us=latency_us,
                slippage_bps=slippage_bps,
                pnl=pnl,
                fees=fees
            )

    def update_market_price(self, symbol: str, price: float):

        if not self.enable_display:
            return

        for trade in self.display.active_trades.values():
            if trade.symbol == symbol.upper():
                trade.current_price = price

    def log_routing_decision(self,
                           symbol: str,
                           selected_venue: str,
                           predicted_latency_us: float,
                           ml_confidence: float,
                           alternative_venues: Dict[str, float] = None):

        if not self.enable_display:
            return

        pass

    def shutdown(self):

        if self.display:
            self.display.stop_display()
            print(f"{Colors.YELLOW}📊 Professional display stopped{Colors.RESET}")
def integrate_with_trading_simulator(trading_simulator, display_manager: HFTDisplayManager):

    original_execute_order = trading_simulator.execute_order

    def enhanced_execute_order(order, *args, **kwargs):
        trade_id = display_manager.log_trade_execution(
            symbol=order.symbol,
            venue=getattr(order, 'venue', 'UNKNOWN'),
            side=order.side.value if hasattr(order.side, 'value') else str(order.side),
            quantity=order.quantity,
            entry_price=order.price,
            strategy=getattr(order, 'strategy', 'UNKNOWN')
        )

        result = original_execute_order(order, *args, **kwargs)

        if result and hasattr(result, 'executed_price'):
            display_manager.update_trade_fill(
                trade_id=trade_id,
                executed_price=result.executed_price,
                latency_us=getattr(result, 'latency_us', 1000),
                fees=getattr(result, 'fees', 0.0)
            )

        return result

    trading_simulator.execute_order = enhanced_execute_order
    return trading_simulator
def integrate_with_rl_router(rl_router, display_manager: HFTDisplayManager):

    original_make_decision = rl_router.make_routing_decision

    def enhanced_make_decision(*args, **kwargs):
        result = original_make_decision(*args, **kwargs)

        if result and hasattr(result, 'venue'):
            display_manager.log_routing_decision(
                symbol=getattr(result, 'symbol', 'UNKNOWN'),
                selected_venue=result.venue,
                predicted_latency_us=getattr(result, 'expected_latency_us', 0),
                ml_confidence=getattr(result, 'confidence', 0)
            )

        return result

    rl_router.make_routing_decision = enhanced_make_decision
    return rl_router
def setup_professional_display_for_hft_system(
    trading_simulator=None,
    rl_router=None,
    market_data_generator=None,
    enable_display: bool = True
) -> HFTDisplayManager:


    display_manager = HFTDisplayManager(enable_display=enable_display)

    if trading_simulator:
        integrate_with_trading_simulator(trading_simulator, display_manager)

    if rl_router:
        integrate_with_rl_router(rl_router, display_manager)

    if market_data_generator and enable_display:
        original_generate = getattr(market_data_generator, 'generate_tick', None)
        if original_generate:
            def enhanced_generate(*args, **kwargs):
                tick = original_generate(*args, **kwargs)
                if tick and hasattr(tick, 'symbol') and hasattr(tick, 'mid_price'):
                    display_manager.update_market_price(tick.symbol, tick.mid_price)
                return tick
            market_data_generator.generate_tick = enhanced_generate

    return display_manager
def add_to_existing_integration():


    example_code = """
# Example integration code for HFT system
from monitoring.display_integration import setup_professional_display_for_hft_system

# In your Phase3CompleteIntegration class:
def initialize_monitoring(self):
    self.display_manager = setup_professional_display_for_hft_system()
    return self.display_manager
"""

    return example_code
if __name__ == "__main__":
    print("🎯 Display Integration Module Ready!")
    print("\nTo integrate with your existing HFT system, add this to phase3_complete_integration.py:")
    print(add_to_existing_integration())