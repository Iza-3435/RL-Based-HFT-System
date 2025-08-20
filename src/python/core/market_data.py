import ctypes
import os
from ctypes import Structure, c_uint64, c_uint32, c_float, c_uint8, c_char_p, POINTER, c_size_t
from typing import List, NamedTuple, Optional
import asyncio
import time
import threading
from dataclasses import dataclass
import numpy as np

class MarketTickC(Structure):
    _fields_ = [
        ("timestamp_ns", c_uint64),
        ("symbol_id", c_uint32),
        ("bid_price", c_float),
        ("ask_price", c_float),
        ("bid_size", c_uint32),
        ("ask_size", c_uint32),
        ("last_price", c_float),
        ("volume", c_uint32),
        ("venue_id", c_uint8),
        ("spread_bps", c_float)
    ]

class MLFeaturesC(Structure):
    _fields_ = [
        ("price_change", c_float),
        ("volume_ratio", c_float),
        ("spread_bps", c_float),
        ("volatility_5min", c_float),
        ("momentum_1min", c_float),
        ("liquidity_score", c_float),
        ("venue_preference", c_float),
        ("timestamp_ns", c_uint64)
    ]

class PerformanceStatsC(Structure):
    _fields_ = [
        ("total_ticks", c_uint64),
        ("avg_generation_time_ns", c_uint64),
        ("ticks_per_second", c_uint64),
        ("cpu_efficiency_percent", c_float)
    ]

@dataclass
class MarketTick:
    timestamp_ns: int
    symbol_id: int
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    venue_id: int
    venue: str
    spread_bps: float
    
    # Order Book Depth (5 levels)
    bid_levels: list = None  # [(price, size), (price, size), ...]
    ask_levels: list = None  # [(price, size), (price, size), ...]
    total_bid_liquidity: int = 0
    total_ask_liquidity: int = 0

    @property
    def timestamp(self) -> float:

        return self.timestamp_ns / 1e9

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) * 0.5

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def volatility(self) -> float:

        return self.spread_bps / 10000.0
    
@dataclass
class MLFeatures:
    price_change: float
    volume_ratio: float
    spread_bps: float
    volatility_5min: float
    momentum_1min: float
    liquidity_score: float
    venue_preference: float
    timestamp_ns: int

class CppMarketDataGenerator:

    def __init__(self, ticks_per_second: int = 100, symbols: Optional[List[str]] = None):
        if symbols is None:
            symbols = [
                "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "NFLX",
                "JPM", "BAC", "WFC", "GS", "C", "JNJ", "PFE", "UNH", "ABBV",
                "PG", "KO", "XOM", "CVX", "DIS", "SPY", "QQQ", "IWM", "GLD", "TLT"
            ]

        self.symbols = symbols
        self.symbol_to_id = {symbol: i for i, symbol in enumerate(symbols)}
        self.venue_names = ["NYSE", "NASDAQ", "ARCA", "IEX", "CBOE"]
        self.ticks_per_second = ticks_per_second

        from collections import deque
        self.arbitrage_opportunities = deque(maxlen=100)
        self.venues = {name: i for i, name in enumerate(self.venue_names)}

        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            lib_path = os.path.join(project_root, "libs", "libhft_market_data.so")
            if not os.path.exists(lib_path):
                lib_path = os.path.join(os.path.dirname(__file__), "libhft_market_data.so")

            self.lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
            self._initialize_cpp_generator()

        except OSError as e:
            print(f"⚠️  C++ library not found, falling back to Python implementation")
            print(f"   Error: {e}")
            print(f"   Run 'make build-cpp' to compile the C++ library")
            self.lib = None
            self.generator_ptr = None
            self.processor_ptr = None

    def _setup_function_signatures(self):


        self.lib.create_tick_generator.argtypes = [c_uint32]
        self.lib.create_tick_generator.restype = ctypes.c_void_p

        self.lib.destroy_tick_generator.argtypes = [ctypes.c_void_p]
        self.lib.destroy_tick_generator.restype = None

        self.lib.generate_tick_c.argtypes = [ctypes.c_void_p, POINTER(MarketTickC)]
        self.lib.generate_tick_c.restype = ctypes.c_int

        self.lib.initialize_symbols_c.argtypes = [ctypes.c_void_p, POINTER(c_char_p), c_size_t]
        self.lib.initialize_symbols_c.restype = ctypes.c_int

        self.lib.create_processor.argtypes = []
        self.lib.create_processor.restype = ctypes.c_void_p

        self.lib.destroy_processor.argtypes = [ctypes.c_void_p]
        self.lib.destroy_processor.restype = None

        self.lib.process_tick_c.argtypes = [ctypes.c_void_p, POINTER(MarketTickC), POINTER(MLFeaturesC)]
        self.lib.process_tick_c.restype = ctypes.c_int

    def _initialize_cpp_generator(self):

        if not self.lib:
            return

        self.generator_ptr = self.lib.create_tick_generator(self.ticks_per_second)
        if not self.generator_ptr:
            raise RuntimeError("Failed to create C++ tick generator")

        self.processor_ptr = self.lib.create_processor()
        if not self.processor_ptr:
            raise RuntimeError("Failed to create C++ market data processor")

        symbol_array = (c_char_p * len(self.symbols))()
        for i, symbol in enumerate(self.symbols):
            symbol_array[i] = symbol.encode('utf-8')

        result = self.lib.initialize_symbols_c(self.generator_ptr, symbol_array, len(self.symbols))
        if not result:
            raise RuntimeError("Failed to initialize symbols in C++ generator")

    def generate_tick(self) -> MarketTick:

        if not self.lib or not self.generator_ptr:
            raise RuntimeError("C++ library not available")

        tick_c = MarketTickC()
        result = self.lib.generate_tick_c(self.generator_ptr, ctypes.byref(tick_c))

        if not result:
            raise RuntimeError("Failed to generate tick")

        symbol_id = tick_c.symbol_id
        symbol = self.symbols[symbol_id] if symbol_id < len(self.symbols) else f"UNK_{symbol_id}"
        venue = self.venue_names[tick_c.venue_id] if tick_c.venue_id < len(self.venue_names) else f"VEN_{tick_c.venue_id}"

        # Generate order book depth (5 levels each side)
        bid_levels, ask_levels, total_bid, total_ask = self._generate_order_book_depth(
            tick_c.bid_price, tick_c.ask_price, tick_c.bid_size, tick_c.ask_size
        )
        
        return MarketTick(
            timestamp_ns=tick_c.timestamp_ns,
            symbol_id=symbol_id,
            symbol=symbol,
            bid_price=tick_c.bid_price,
            ask_price=tick_c.ask_price,
            bid_size=tick_c.bid_size,
            ask_size=tick_c.ask_size,
            last_price=tick_c.last_price,
            volume=tick_c.volume,
            venue_id=tick_c.venue_id,
            venue=venue,
            spread_bps=tick_c.spread_bps,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            total_bid_liquidity=total_bid,
            total_ask_liquidity=total_ask
        )

    def _generate_order_book_depth(self, bid_price: float, ask_price: float, bid_size: int, ask_size: int):
        
        # Generate bid levels (descending prices)
        bid_levels = []
        total_bid_liquidity = bid_size
        current_bid = bid_price
        
        for level in range(5):
            if level == 0:
                bid_levels.append((bid_price, bid_size))
            else:
                price_decrement = np.random.uniform(0.01, 0.05) * level
                current_bid = bid_price - price_decrement
                size = int(bid_size * np.random.uniform(0.7, 1.3))
                bid_levels.append((current_bid, size))
                total_bid_liquidity += size

        # Generate ask levels (ascending prices)
        ask_levels = []
        total_ask_liquidity = ask_size
        current_ask = ask_price
        
        for level in range(5):
            if level == 0:
                ask_levels.append((ask_price, ask_size))
            else:
                price_increment = np.random.uniform(0.01, 0.05) * level
                current_ask = ask_price + price_increment
                size = int(ask_size * np.random.uniform(0.7, 1.3))
                ask_levels.append((current_ask, size))
                total_ask_liquidity += size
        
        return bid_levels, ask_levels, total_bid_liquidity, total_ask_liquidity

    def process_tick_features(self, tick: MarketTick) -> MLFeatures:

        if not self.lib or not self.processor_ptr:
            raise RuntimeError("C++ library not available")

        tick_c = MarketTickC()
        tick_c.timestamp_ns = tick.timestamp_ns
        tick_c.symbol_id = tick.symbol_id
        tick_c.bid_price = tick.bid_price
        tick_c.ask_price = tick.ask_price
        tick_c.bid_size = tick.bid_size
        tick_c.ask_size = tick.ask_size
        tick_c.last_price = tick.last_price
        tick_c.volume = tick.volume
        tick_c.venue_id = tick.venue_id
        tick_c.spread_bps = tick.spread_bps

        features_c = MLFeaturesC()
        result = self.lib.process_tick_c(self.processor_ptr, ctypes.byref(tick_c), ctypes.byref(features_c))

        if not result:
            raise RuntimeError("Failed to process tick features")

        return MLFeatures(
            price_change=features_c.price_change,
            volume_ratio=features_c.volume_ratio,
            spread_bps=features_c.spread_bps,
            volatility_5min=features_c.volatility_5min,
            momentum_1min=features_c.momentum_1min,
            liquidity_score=features_c.liquidity_score,
            venue_preference=features_c.venue_preference,
            timestamp_ns=features_c.timestamp_ns
        )

    async def generate_market_data_stream(self, duration_seconds: int):

        if not self.lib:
            print("⚠️  C++ library not available, cannot generate high-frequency stream")
            return

        print(f"🚀 Starting C++ high-frequency stream for {duration_seconds}s")
        print(f"📊 Target rate: {self.ticks_per_second:,} ticks/sec")

        start_time = time.time()
        end_time = start_time + duration_seconds
        tick_count = 0

        target_interval = 1.0 / self.ticks_per_second
        next_tick_time = start_time

        while time.time() < end_time:
            current_time = time.time()

            if current_time >= next_tick_time:
                try:
                    tick = self.generate_tick()
                    tick_count += 1

                    if tick_count % 1000 == 0:
                        elapsed = current_time - start_time
                        rate = (tick_count / elapsed) if elapsed > 0 else 0
                        print(f"📈 C++ Stream: {tick_count:,} ticks ({rate:.0f}/sec)")

                    yield tick

                    next_tick_time += target_interval

                except Exception as e:
                    print(f"❌ Error generating tick: {e}")
                    break
            else:
                await asyncio.sleep(0.00001)

        total_time = time.time() - start_time
        actual_rate = tick_count / total_time if total_time > 0 else 0
        efficiency = (actual_rate / self.ticks_per_second) * 100

        # Market data generation complete - metrics now only shown in final summary

    def benchmark_performance(self, test_duration_seconds: int = 10) -> dict:

        if not self.lib:
            return {"error": "C++ library not available"}

        print(f"🔥 Benchmarking C++ performance for {test_duration_seconds}s")

        start_time = time.perf_counter()
        tick_count = 0
        processing_times = []

        end_time = start_time + test_duration_seconds

        while time.perf_counter() < end_time:
            tick_start = time.perf_counter()

            try:
                tick = self.generate_tick()
                features = self.process_tick_features(tick)

                tick_end = time.perf_counter()
                processing_times.append((tick_end - tick_start) * 1_000_000)
                tick_count += 1

            except Exception as e:
                print(f"❌ Benchmark error: {e}")
                break

        total_time = time.perf_counter() - start_time

        results = {
            "total_ticks": tick_count,
            "total_time_seconds": total_time,
            "ticks_per_second": tick_count / total_time,
            "target_ticks_per_second": self.ticks_per_second,
            "efficiency_percent": (tick_count / total_time / self.ticks_per_second) * 100,
            "avg_processing_time_us": sum(processing_times) / len(processing_times) if processing_times else 0,
            "max_processing_time_us": max(processing_times) if processing_times else 0,
            "min_processing_time_us": min(processing_times) if processing_times else 0
        }

        return results

    def __del__(self):

        if self.lib:
            if hasattr(self, 'generator_ptr') and self.generator_ptr:
                self.lib.destroy_tick_generator(self.generator_ptr)
            if hasattr(self, 'processor_ptr') and self.processor_ptr:
                self.lib.destroy_processor(self.processor_ptr)
def create_high_frequency_generator(ticks_per_second: int = 1000) -> CppMarketDataGenerator:

    return CppMarketDataGenerator(ticks_per_second)
async def test_cpp_performance():

    print("🔥 Testing C++ Market Data Performance")

    generator = CppMarketDataGenerator(ticks_per_second=5000)

    if generator.lib:
        results = generator.benchmark_performance(test_duration_seconds=5)

        print("\n📊 C++ Performance Results:")
        print(f"   Total ticks: {results['total_ticks']:,}")
        print(f"   Target rate: {results['target_ticks_per_second']:,} ticks/sec")
        print(f"   Actual rate: {results['ticks_per_second']:,.0f} ticks/sec")
        print(f"   Efficiency: {results['efficiency_percent']:.1f}%")
        print(f"   Avg processing: {results['avg_processing_time_us']:.2f} μs")
        print(f"   Max processing: {results['max_processing_time_us']:.2f} μs")

        if results['efficiency_percent'] >= 95:
            print("   Status: ✅ EXCELLENT - C++ delivering target performance!")
        elif results['efficiency_percent'] >= 80:
            print("   Status: ⚡ GOOD - C++ performing well")
        else:
            print("   Status: ⚠️  Suboptimal - Check system resources")
    else:
        print("❌ C++ library not available for testing")
if __name__ == "__main__":
    asyncio.run(test_cpp_performance())