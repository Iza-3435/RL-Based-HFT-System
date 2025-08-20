import yfinance as yf
import asyncio
import time
from datetime import datetime
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from typing import List, Dict, Optional, Any, Tuple
EXPANDED_STOCK_LIST = [
    'AAPL', 'MSFT', 'GOOGL',

    'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX',

    'JPM', 'BAC', 'WFC', 'GS', 'C',

    'JNJ', 'PFE', 'UNH', 'ABBV',

    'PG', 'KO', 'XOM', 'CVX', 'DIS',

    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT'
]

OPTIMAL_TICK_RATES = {
    'fast': {
        'target_ticks_per_minute': 600,  # Very low for fast testing
        'training_duration_minutes': 10,
        'symbols': 5  # Fewer symbols for speed
    },
    'development': {
        'target_ticks_per_minute': 1200,
        'training_duration_minutes': 10,
        'symbols': 27
    },
    'balanced': {
        'target_ticks_per_minute': 2400,
        'training_duration_minutes': 15,
        'symbols': 27
    },
    'production': {
        'target_ticks_per_minute': 3600,
        'training_duration_minutes': 30,
        'symbols': 27
    },
    'ultra_hf': {
        'target_ticks_per_minute': 6000,
        'training_duration_minutes': 20,
        'symbols': 27
    },
    'stress_test': {
        'target_ticks_per_minute': 12000,
        'training_duration_minutes': 5,
        'symbols': 27
    }
}

@dataclass
class RealMarketTick:

    symbol: str
    venue: str
    timestamp: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    real_spread: float
    market_cap: Optional[float] = None
    day_change: Optional[float] = None
    volatility: Optional[float] = None

    is_market_hours: bool = True
    exchange_status: str = "open"
    liquidity_tier: str = "high"

    @property
    def spread(self) -> float:
        return self.real_spread

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2
@dataclass
class VenueConfig:

    name: str
    base_latency_us: int
    jitter_range: Tuple[int, int]
    packet_loss_rate: float
    congestion_factor: float
class EnhancedTickGenerator:


    def __init__(self, base_update_interval=10):
        self.base_update_interval = base_update_interval
        self.tick_multipliers = {
                'SPY': 8,
                'QQQ': 7,
                'IWM': 6,

                'TSLA': 6,
                'NVDA': 6,
                'META': 5,

                'AAPL': 5,
                'MSFT': 5,
                'GOOGL': 4,
                'AMZN': 4,
                'NFLX': 4,

                'JPM': 3,
                'BAC': 3,
                'WFC': 3,
                'GS': 3,
                'C': 3,

                'JNJ': 2,
                'PFE': 2,
                'UNH': 3,
                'ABBV': 2,

                'PG': 2,
                'KO': 2,
                'XOM': 3,
                'CVX': 3,
                'DIS': 2,

                'GLD': 2,
                'TLT': 1,
            }

        self.latency_history = {venue: deque(maxlen=100) for venue in self.tick_multipliers.keys()}
        self.congestion_events = []

        print("🔧 Enhanced Tick Generator initialized with multipliers:")
        for symbol, multiplier in self.tick_multipliers.items():
            print(f"   {symbol}: {multiplier}x")
    def get_update_interval(self, symbol):

        multiplier = self.tick_multipliers.get(symbol, 3)
        return self.base_update_interval / multiplier

    def generate_intraday_ticks(self, symbol, base_data, num_ticks=100):

        ticks = []
        current_price = base_data['price']
        current_volume = base_data['volume']
        volatility = base_data.get('volatility', 0.02)

        import datetime
        start_time = time.time()

        for i in range(num_ticks):
            price_change_pct = np.random.normal(0, volatility/np.sqrt(252*390))

            if i > 0:
                momentum = (ticks[-1]['price'] - current_price) / current_price * 0.1
                price_change_pct += momentum

            new_price = current_price * (1 + price_change_pct)

            volume_multiplier = 1 + abs(price_change_pct) * 50
            new_volume = int(current_volume * np.random.uniform(0.5, volume_multiplier))

            base_spread_bps = 1.0
            vol_impact = abs(price_change_pct) * 1000
            volume_impact = max(0.5, min(2.0, 1000000 / new_volume))

            spread_bps = base_spread_bps * (1 + vol_impact) * volume_impact
            spread_dollars = new_price * spread_bps / 10000

            tick = {
                'timestamp': start_time + i * self.get_update_interval(symbol),
                'symbol': symbol,
                'price': new_price,
                'bid': new_price - spread_dollars/2,
                'ask': new_price + spread_dollars/2,
                'volume': new_volume,
                'spread_bps': spread_bps
            }
            ticks.append(tick)
            current_price = new_price

        return ticks

    def get_tick_frequency_for_mode(self, mode='balanced'):

        config = OPTIMAL_TICK_RATES.get(mode, OPTIMAL_TICK_RATES['balanced'])

        target_rate = config['target_ticks_per_minute'] / 60
        base_interval = 1.0 / target_rate

        return {
            'base_interval': base_interval,
            'target_ticks_per_minute': config['target_ticks_per_minute'],
            'recommended_symbols': config['symbols'],
            'training_duration': config['training_duration_minutes']
        }

    def generate_symbol_priorities(self, symbols, mode='balanced'):

        priorities = []
        for symbol in symbols:
            multiplier = self.tick_multipliers.get(symbol, 3)
            priorities.append((symbol, multiplier))

        priorities.sort(key=lambda x: x[1], reverse=True)

        config = OPTIMAL_TICK_RATES[mode]
        max_symbols = config['symbols']

        return [symbol for symbol, _ in priorities[:max_symbols]]
class UltraRealisticMarketDataGenerator:

    def __init__(self, symbols: List[str] = None, mode: str = 'balanced'):
        if symbols is None:
            symbols = EXPANDED_STOCK_LIST
        elif len(symbols) < 5:
            print(f"⚠️  Only {len(symbols)} symbols provided. Consider using more!")

        self.enhanced_tick_gen = EnhancedTickGenerator()
        self.mode = mode

        tick_config = self.enhanced_tick_gen.get_tick_frequency_for_mode(mode)
        self.base_update_interval = tick_config['base_interval']
        self.target_ticks_per_minute = tick_config['target_ticks_per_minute']

        self.symbols = symbols
        print(f"🎯 FORCED: Using ALL {len(self.symbols)} symbols (ignoring {mode} mode limits)")

        print(f"🚀 Enhanced Trading {len(self.symbols)} symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}")
        print(f"📊 Target tick rate: {self.target_ticks_per_minute} ticks/minute ({self.target_ticks_per_minute/60:.1f}/sec)")
        print(f"⏱️  Base update interval: {self.base_update_interval:.2f} seconds")
        self.real_venues = {
            'NYSE': {'endpoint': 'www.nyse.com', 'maker_fee': 0.0003, 'rebate': 0.0001},
            'NASDAQ': {'endpoint': 'www.nasdaq.com', 'maker_fee': 0.0002, 'rebate': 0.0001},
            'ARCA': {'endpoint': 'www.nyse.com', 'maker_fee': 0.0003, 'rebate': 0.0001},
            'IEX': {'endpoint': 'iextrading.com', 'maker_fee': 0.0000, 'rebate': 0.0000},
            'CBOE': {'endpoint': 'www.cboe.com', 'maker_fee': 0.0003, 'rebate': 0.0001}
        }

        self.arbitrage_opportunities = deque(maxlen=50)
        self.tick_count = 0
        self.current_prices = {}
        self.market_hours_cache = {}

        print("🌟 ENHANCED Market Data Generator initialized")
        print(f"📊 Symbols: {self.symbols}")
        print(f"🏛️  Real venues: {list(self.real_venues.keys())}")
    def _get_time_of_day_factors(self, current_time: datetime) -> Dict[str, float]:

        hour = current_time.hour
        minute = current_time.minute

        if 9 <= hour < 11:
            spread_multiplier = 1.4
            latency_multiplier = 1.6
            volume_multiplier = 2.2

        elif 12 <= hour < 13:
            spread_multiplier = 1.1
            latency_multiplier = 0.8
            volume_multiplier = 0.6

        elif hour >= 15 and (hour > 15 or minute >= 30):
            spread_multiplier = 1.5
            latency_multiplier = 2.1
            volume_multiplier = 3.0

        else:
            spread_multiplier = 1.0
            latency_multiplier = 1.0
            volume_multiplier = 1.0

        return {
            'spread_factor': spread_multiplier,
            'latency_factor': latency_multiplier,
            'volume_factor': volume_multiplier
        }

    def _apply_volume_spread_dynamics(self, base_spread: float, volume: int, avg_volume: int) -> float:

        if avg_volume <= 0:
            return base_spread

        volume_ratio = volume / avg_volume
        volume_factor = 1.0 / (1.0 + 0.15 * np.log(1 + volume_ratio))

        volume_factor = max(0.7, min(1.3, volume_factor))

        return base_spread * volume_factor

    def _get_average_volume(self, symbol: str) -> int:

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('averageVolume', 1000000)
        except:
            return 1000000

    def is_market_open(self) -> bool:

        now = datetime.now()

        return True

    async def get_ultra_realistic_data(self):

        ultra_real_ticks = []

        is_market_open = self.is_market_open()
        current_time = time.time()

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)

                recent_data = ticker.history(period="2d", interval="1m")
                if recent_data.empty:
                    continue

                latest = recent_data.iloc[-1]

                info = ticker.info

                current_price = float(latest['Close'])
                day_open = float(recent_data.iloc[0]['Open']) if len(recent_data) > 1 else current_price
                day_change = (current_price - day_open) / day_open if day_open > 0 else 0

                base_spread_pct = 0.001

                market_cap = info.get('marketCap', 1e12)
                volatility = self._calculate_real_volatility(recent_data)

                market_cap_factor = max(0.5, min(2.0, 1e12 / market_cap))
                volatility_factor = max(0.8, min(2.0, volatility * 20))

                real_spread_pct = base_spread_pct * market_cap_factor * volatility_factor
                real_spread_dollars = current_price * real_spread_pct

                real_spread_dollars = max(real_spread_dollars, 0.10)
                real_spread_dollars = min(real_spread_dollars, 5.00)
                time_factors = self._get_time_of_day_factors(datetime.fromtimestamp(current_time))
                enhanced_spread = real_spread_dollars * time_factors['spread_factor']
                avg_volume = info.get('averageVolume', 1000000)
                current_volume = int(latest['Volume'])
                final_spread_dollars = self._apply_volume_spread_dynamics(
                    enhanced_spread, current_volume, avg_volume
                )
                real_spread_dollars = final_spread_dollars

                half_spread = real_spread_dollars / 2
                real_bid = current_price - half_spread
                real_ask = current_price + half_spread

                avg_volume = info.get('averageVolume', 1000000)
                if avg_volume > 10_000_000:
                    liquidity_tier = "high"
                    base_size = 2000
                elif avg_volume > 1_000_000:
                    liquidity_tier = "medium"
                    base_size = 1000
                else:
                    liquidity_tier = "low"
                    base_size = 500

                print(f"📊 ULTRA-REAL {symbol}: ${current_price:.2f} "
                      f"spread:${real_spread_dollars:.3f} "
                      f"change:{day_change:.2%} "
                      f"liquidity:{liquidity_tier}")

                for i, (venue, venue_info) in enumerate(self.real_venues.items()):
                    venue_price_adj = np.random.uniform(-0.0002, 0.0002)
                    venue_spread_adj = np.random.uniform(0.95, 1.05)

                    venue_bid = real_bid * (1 + venue_price_adj)
                    venue_ask = real_ask * (1 + venue_price_adj) * venue_spread_adj

                    if venue_ask <= venue_bid:
                        venue_ask = venue_bid * 1.0002

                    venue_volume_factor = {
                        'NYSE': 0.3, 'NASDAQ': 0.25, 'ARCA': 0.2, 'IEX': 0.15, 'CBOE': 0.1
                    }
                    venue_volume = int(latest['Volume'] * venue_volume_factor.get(venue, 0.2))

                    tick = RealMarketTick(
                        symbol=symbol,
                        venue=venue,
                        timestamp=current_time + (i * 0.001),
                        bid_price=round(venue_bid, 2),
                        ask_price=round(venue_ask, 2),
                        bid_size=np.random.randint(base_size//2, base_size*2),
                        ask_size=np.random.randint(base_size//2, base_size*2),
                        last_price=current_price,
                        volume=venue_volume,
                        real_spread=venue_ask - venue_bid,
                        market_cap=market_cap,
                        day_change=day_change,
                        volatility=volatility,
                        is_market_hours=is_market_open,
                        exchange_status="open" if is_market_open else "closed",
                        liquidity_tier=liquidity_tier
                    )

                    ultra_real_ticks.append(tick)
                    self.current_prices[symbol] = current_price

            except Exception as e:
                print(f"❌ Error fetching ultra-real data for {symbol}: {e}")

        return ultra_real_ticks

    def _calculate_real_volatility(self, recent_data):

        if len(recent_data) < 10:
            return 0.02

        returns = recent_data['Close'].tail(20).pct_change().dropna()
        if len(returns) == 0:
            return 0.02

        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol)

    def _detect_ultra_realistic_arbitrage(self, ticks):

        by_symbol = {}
        for tick in ticks:
            if tick.symbol not in by_symbol:
                by_symbol[tick.symbol] = []
            by_symbol[tick.symbol].append(tick)

        opportunities_found = 0

        for symbol, symbol_ticks in by_symbol.items():
            if len(symbol_ticks) < 2:
                continue

            for buy_tick in symbol_ticks:
                for sell_tick in symbol_ticks:
                    if buy_tick.venue == sell_tick.venue:
                        continue

                    gross_profit = sell_tick.bid_price - buy_tick.ask_price

                    buy_fees = buy_tick.ask_price * self.real_venues[buy_tick.venue]['maker_fee']
                    sell_fees = sell_tick.bid_price * self.real_venues[sell_tick.venue]['maker_fee']
                    buy_rebates = buy_tick.ask_price * self.real_venues[buy_tick.venue]['rebate']
                    sell_rebates = sell_tick.bid_price * self.real_venues[sell_tick.venue]['rebate']

                    net_fees = buy_fees + sell_fees - buy_rebates - sell_rebates
                    net_profit = gross_profit - net_fees

                    if net_profit > 0.03:

                        if np.random.random() < 0.70:
                            continue

                        max_size = min(buy_tick.ask_size, sell_tick.bid_size, 500)

                        opportunity = {
                            'symbol': symbol,
                            'timestamp': time.time(),
                            'buy_venue': buy_tick.venue,
                            'sell_venue': sell_tick.venue,
                            'buy_price': buy_tick.ask_price,
                            'sell_price': sell_tick.bid_price,
                            'gross_profit_per_share': gross_profit,
                            'net_profit_per_share': net_profit,
                            'profit_per_share': net_profit,
                            'max_size': max_size,
                            'buy_fees': buy_fees,
                            'sell_fees': sell_fees,
                            'net_fees': net_fees,
                            'market_hours': buy_tick.is_market_hours,
                            'liquidity_score': (buy_tick.bid_size + sell_tick.ask_size) / 2
                        }

                        self.arbitrage_opportunities.append(opportunity)
                        opportunities_found += 1

                        print(f"🎯 ULTRA-REAL ARBITRAGE: {symbol} "
                              f"buy@{buy_tick.venue}:{buy_tick.ask_price:.2f} "
                              f"sell@{sell_tick.venue}:{sell_tick.bid_price:.2f} "
                              f"net_profit:${net_profit:.3f}")

        if opportunities_found == 0:
            print("📊 No arbitrage opportunities found this round")

    async def generate_market_data_stream(self, duration_seconds=60):

        print(f"🚀 Starting ENHANCED market data stream for {duration_seconds}s")
        print(f"📊 Mode: {self.mode} | Target: {self.target_ticks_per_minute} ticks/min")

        end_time = time.time() + duration_seconds
        last_updates = {symbol: 0 for symbol in self.symbols}
        tick_count_by_symbol = {symbol: 0 for symbol in self.symbols}

        while time.time() < end_time:
            try:
                current_time = time.time()

                symbols_to_update = []
                for symbol in self.symbols:
                    update_interval = self.enhanced_tick_gen.get_update_interval(symbol)
                    if current_time - last_updates[symbol] >= update_interval:
                        symbols_to_update.append(symbol)
                        last_updates[symbol] = current_time

                if symbols_to_update:
                    current_time_str = datetime.now().strftime('%H:%M:%S')
                    print(f"📡 [{current_time_str}] Updating {len(symbols_to_update)} symbols: {symbols_to_update}")

                    ultra_real_ticks = await self.get_ultra_realistic_data_for_symbols(symbols_to_update)

                    if ultra_real_ticks:
                        self._detect_ultra_realistic_arbitrage(ultra_real_ticks)

                        for tick in ultra_real_ticks:
                            self.tick_count += 1
                            tick_count_by_symbol[tick.symbol] += 1
                            yield tick

                            # Remove sleep for maximum speed - just yield control
                            if self.tick_count % 100 == 0:
                                await asyncio.sleep(0.0001)

                next_update_times = []
                for symbol in self.symbols:
                    interval = self.enhanced_tick_gen.get_update_interval(symbol)
                    next_update = last_updates[symbol] + interval
                    next_update_times.append(next_update)

                # Remove sleep for maximum tick generation speed
                if next_update_times:
                    # Only minimal yield for async, no actual sleeping
                    if self.tick_count % 1000 == 0:
                        await asyncio.sleep(0.0001)
                else:
                    await asyncio.sleep(0.0001)

            except Exception as e:
                print(f"❌ Error in enhanced stream: {e}")
                # No sleep on error, just continue

        total_ticks = sum(tick_count_by_symbol.values())
        actual_rate = total_ticks / (duration_seconds / 60) if duration_seconds > 0 else 0

        print(f"🏁 Enhanced stream complete!")
        print(f"📊 Total ticks: {total_ticks} | Target: {self.target_ticks_per_minute * (duration_seconds/60):.0f}")
        print(f"📈 Actual rate: {actual_rate:.1f} ticks/min | Target: {self.target_ticks_per_minute}")
        print(f"🎯 Rate efficiency: {(actual_rate/self.target_ticks_per_minute)*100:.1f}%")
        print(f"💎 Arbitrage opportunities: {len(self.arbitrage_opportunities)}")

        print("📋 Per-symbol tick counts:")
        for symbol, count in sorted(tick_count_by_symbol.items(), key=lambda x: x[1], reverse=True):
            multiplier = self.enhanced_tick_gen.tick_multipliers.get(symbol, 3)
            print(f"   {symbol}: {count} ticks (priority: {multiplier}x)")
    async def get_ultra_realistic_data_for_symbols(self, symbols_to_update):

        ultra_real_ticks = []
        current_time = time.time()
        is_market_open = self.is_market_open()

        for symbol in symbols_to_update:
            try:
                ticker = yf.Ticker(symbol)

                recent_data = ticker.history(period="2d", interval="1m")
                if recent_data.empty:
                    recent_data = ticker.history(period="5d", interval="1d")

                if recent_data.empty:
                    continue

                latest = recent_data.iloc[-1]

                try:
                    info = ticker.info
                except:
                    info = {'marketCap': 1e12, 'averageVolume': 5000000}

                current_price = float(latest['Close'])
                day_open = float(recent_data.iloc[0]['Open']) if len(recent_data) > 1 else current_price
                day_change = (current_price - day_open) / day_open if day_open > 0 else 0

                base_spread_pct = 0.001

                market_cap = info.get('marketCap', 1e12)
                volatility = self._calculate_real_volatility(recent_data)

                market_cap_factor = max(0.5, min(2.0, 1e12 / market_cap))
                volatility_factor = max(0.8, min(2.0, volatility * 20))

                real_spread_pct = base_spread_pct * market_cap_factor * volatility_factor
                real_spread_dollars = current_price * real_spread_pct
                real_spread_dollars = max(real_spread_dollars, 0.01)
                real_spread_dollars = min(real_spread_dollars, 2.00)

                time_factors = self._get_time_of_day_factors(datetime.fromtimestamp(current_time))
                enhanced_spread = real_spread_dollars * time_factors['spread_factor']
                avg_volume = info.get('averageVolume', 1000000)
                current_volume = int(latest['Volume']) if 'Volume' in latest.index else avg_volume
                final_spread_dollars = self._apply_volume_spread_dynamics(
                    enhanced_spread, current_volume, avg_volume
                )

                half_spread = final_spread_dollars / 2
                real_bid = current_price - half_spread
                real_ask = current_price + half_spread

                if avg_volume > 10_000_000:
                    liquidity_tier = "high"
                    base_size = 2000
                elif avg_volume > 1_000_000:
                    liquidity_tier = "medium"
                    base_size = 1000
                else:
                    liquidity_tier = "low"
                    base_size = 500

                print(f"📊 REAL {symbol}: ${current_price:.2f} "
                    f"spread:${final_spread_dollars:.3f} "
                    f"change:{day_change:.2%} "
                    f"liquidity:{liquidity_tier}")

                for i, (venue, venue_info) in enumerate(self.real_venues.items()):
                    venue_price_adj = np.random.uniform(-0.0002, 0.0002)
                    venue_spread_adj = np.random.uniform(0.95, 1.05)

                    venue_bid = real_bid * (1 + venue_price_adj)
                    venue_ask = real_ask * (1 + venue_price_adj) * venue_spread_adj

                    if venue_ask <= venue_bid:
                        venue_ask = venue_bid * 1.0002

                    venue_volume_factor = {
                        'NYSE': 0.3, 'NASDAQ': 0.25, 'ARCA': 0.2, 'IEX': 0.15, 'CBOE': 0.1
                    }
                    venue_volume = int(current_volume * venue_volume_factor.get(venue, 0.2))

                    tick = RealMarketTick(
                        symbol=symbol,
                        venue=venue,
                        timestamp=current_time + (i * 0.001),
                        bid_price=round(venue_bid, 2),
                        ask_price=round(venue_ask, 2),
                        bid_size=np.random.randint(base_size//2, base_size*2),
                        ask_size=np.random.randint(base_size//2, base_size*2),
                        last_price=current_price,
                        volume=venue_volume,
                        real_spread=venue_ask - venue_bid,
                        market_cap=market_cap,
                        day_change=day_change,
                        volatility=volatility,
                        is_market_hours=is_market_open,
                        exchange_status="open" if is_market_open else "closed",
                        liquidity_tier=liquidity_tier
                    )

                    ultra_real_ticks.append(tick)
                    self.current_prices[symbol] = current_price

            except Exception as e:
                print(f"❌ Error fetching ultra-real data for {symbol}: {e}")

        return ultra_real_ticks

    async def initialize_historical_calibration(self):

            print("📊 Performing real market analysis...")

            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)

                    hist = ticker.history(period="1mo", interval="1d")
                    if not hist.empty:
                        volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                        avg_volume = hist['Volume'].mean()

                        print(f"✅ {symbol}: {volatility:.1%} annual volatility, "
                            f"{avg_volume:,.0f} avg daily volume")

                except Exception as e:
                    print(f"⚠️  Could not calibrate {symbol}: {e}")

            print("✅ Real market calibration complete")

    def get_performance_metrics(self):

        return {
            'data_source': 'ULTRA_REALISTIC_LIVE',
            'total_ticks': self.tick_count,
            'real_arbitrage_opportunities': len(self.arbitrage_opportunities),
            'current_prices': dict(self.current_prices),
            'market_hours': self.is_market_open(),
            'last_update': time.time(),
            'venues_monitored': list(self.real_venues.keys())
        }