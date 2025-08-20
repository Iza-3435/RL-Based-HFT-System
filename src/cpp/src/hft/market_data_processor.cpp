// High-Performance Market Data Processor - C++ Implementation
// Ultra-fast tick generation and processing for HFT systems
// Designed for 180K+ ticks/second with sub-microsecond latency

#include "../../include/hft/market_data_processor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

namespace hft {
namespace market_data {

// Default symbol universe - major US equities and ETFs
// These symbols represent high-volume, liquid instruments suitable for HFT
static const std::vector<std::string> DEFAULT_SYMBOLS = {
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "NFLX",  // Tech giants
    "JPM", "BAC", "WFC", "GS", "C", "JNJ", "PFE", "UNH", "ABBV",       // Finance/Healthcare
    "PG", "KO", "XOM", "CVX", "DIS", "SPY", "QQQ", "IWM", "GLD", "TLT" // Consumer/ETFs
};

// Venue configurations with realistic fee structures and latency profiles
// maker_fee, taker_fee, rebate, avg_latency_us, jitter_us
static const std::vector<VenueInfo> DEFAULT_VENUES = {
    {"NYSE",   0.0003f, 0.0003f, 0.0001f, 250, 50},  // NYSE - traditional market maker friendly
    {"NASDAQ", 0.0003f, 0.0003f, 0.0001f, 230, 45},  // NASDAQ - slightly faster
    {"ARCA",   0.0002f, 0.0003f, 0.0002f, 240, 40},  // ARCA - higher rebates
    {"IEX",    0.0000f, 0.0009f, 0.0000f, 400, 100}, // IEX - no maker fees but higher latency
    {"CBOE",   0.0002f, 0.0003f, 0.0001f, 280, 60}   // CBOE - competitive pricing
};

// Base prices for realistic market simulation
// Reflects actual stock price levels for accurate market impact modeling
static const std::vector<float> BASE_PRICES = {
    227.21f, 521.75f, 201.00f, 339.18f, 182.09f, 765.52f, 221.37f, 1218.37f,
    289.71f, 46.19f, 77.61f, 719.33f, 92.31f, 174.04f, 24.61f, 252.41f, 198.64f,
    155.03f, 70.79f, 105.88f, 153.54f, 112.58f, 635.82f, 572.75f, 220.28f, 308.60f, 87.41f
};

// Tick generation frequency multipliers - simulates varying activity levels
// Higher values = more active symbols (e.g., AAPL, SPY get more ticks)
static const std::vector<uint32_t> TICK_MULTIPLIERS = {
    5, 5, 4, 6, 6, 5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 3, 2, 2, 2, 3, 3, 2, 8, 7, 6, 2, 1
};

HighFrequencyTickGenerator::HighFrequencyTickGenerator(uint32_t target_ticks_per_second) 
    : num_symbols_(0), num_venues_(0), total_ticks_generated_(0), generation_time_ns_(0) {
    
    
    auto now = std::chrono::high_resolution_clock::now();
    rng_state_.store(now.time_since_epoch().count());
    
    set_target_frequency(target_ticks_per_second);
    
    
    initialize_symbols(DEFAULT_SYMBOLS, DEFAULT_VENUES);
}

void HighFrequencyTickGenerator::initialize_symbols(const std::vector<std::string>& symbol_names,
                                                  const std::vector<VenueInfo>& venue_configs) {
    num_symbols_ = std::min(symbol_names.size(), MAX_SYMBOLS);
    num_venues_ = std::min(venue_configs.size(), MAX_VENUES);
    
    
    for (size_t i = 0; i < num_symbols_; ++i) {
        symbols_[i].symbol_name = symbol_names[i];
        symbols_[i].current_price = (i < BASE_PRICES.size()) ? BASE_PRICES[i] : 100.0f;
        symbols_[i].volatility = fast_random_float(0.15f, 0.45f);
        symbols_[i].avg_volume = fast_random_uint32(10000, 100000);
        symbols_[i].tick_multiplier = (i < TICK_MULTIPLIERS.size()) ? TICK_MULTIPLIERS[i] : 3;
        symbols_[i].last_update_ns = 0;
        symbols_[i].price_trend = fast_random_float(-0.02f, 0.02f);
    }
    
    
    for (size_t i = 0; i < num_venues_; ++i) {
        venues_[i] = venue_configs[i];
    }
}

MarketTick HighFrequencyTickGenerator::generate_tick() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    
    uint32_t total_weight = 0;
    for (size_t i = 0; i < num_symbols_; ++i) {
        total_weight += symbols_[i].tick_multiplier;
    }
    
    uint32_t random_weight = fast_random_uint32(0, total_weight - 1);
    size_t selected_symbol = 0;
    uint32_t cumulative_weight = 0;
    
    for (size_t i = 0; i < num_symbols_; ++i) {
        cumulative_weight += symbols_[i].tick_multiplier;
        if (random_weight < cumulative_weight) {
            selected_symbol = i;
            break;
        }
    }
    
    auto& symbol = symbols_[selected_symbol];
    
    
    float price_change = fast_random_float(-symbol.volatility * 0.01f, symbol.volatility * 0.01f);
    price_change += symbol.price_trend * 0.001f; 
    
    float new_price = symbol.current_price * (1.0f + price_change);
    new_price = std::max(new_price, 0.01f); 
    
    
    float spread_bps = fast_random_float(0.5f, 3.0f);
    if (symbol.current_price > 500.0f) spread_bps *= 2.0f; 
    
    float spread_dollars = (spread_bps / 10000.0f) * new_price;
    float bid_price = new_price - spread_dollars * 0.5f;
    float ask_price = new_price + spread_dollars * 0.5f;
    
    
    uint32_t volume = fast_random_uint32(
        static_cast<uint32_t>(symbol.avg_volume * 0.1f),
        static_cast<uint32_t>(symbol.avg_volume * 2.0f)
    );
    
    
    size_t venue_idx = fast_random_uint32(0, num_venues_ - 1);
    
    
    symbol.current_price = new_price;
    symbol.last_update_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    
    MarketTick tick;
    tick.timestamp_ns = symbol.last_update_ns;
    tick.symbol_id = static_cast<uint32_t>(selected_symbol);
    tick.bid_price = bid_price;
    tick.ask_price = ask_price;
    tick.bid_size = fast_random_uint32(100, 10000);
    tick.ask_size = fast_random_uint32(100, 10000);
    tick.last_price = new_price;
    tick.volume = volume;
    tick.venue_id = static_cast<uint8_t>(venue_idx);
    tick.spread_bps = spread_bps;
    
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    
    total_ticks_generated_.fetch_add(1, std::memory_order_relaxed);
    generation_time_ns_.fetch_add(duration_ns, std::memory_order_relaxed);
    
    return tick;
}

void HighFrequencyTickGenerator::generate_tick_batch(MarketTick* output_buffer, size_t count) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < count; ++i) {
        output_buffer[i] = generate_tick();
        
        
        if (i > 0) {
            output_buffer[i].timestamp_ns = output_buffer[i-1].timestamp_ns + 
                (target_tick_interval_ns_ + fast_random_uint32(0, target_tick_interval_ns_ / 10));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    generation_time_ns_.fetch_add(duration_ns, std::memory_order_relaxed);
}

HighFrequencyTickGenerator::PerformanceStats HighFrequencyTickGenerator::get_performance_stats() const {
    uint64_t total_ticks = total_ticks_generated_.load(std::memory_order_acquire);
    uint64_t total_time_ns = generation_time_ns_.load(std::memory_order_acquire);
    
    PerformanceStats stats;
    stats.total_ticks = total_ticks;
    stats.avg_generation_time_ns = (total_ticks > 0) ? (total_time_ns / total_ticks) : 0;
    stats.ticks_per_second = (total_time_ns > 0) ? 
        static_cast<uint64_t>((total_ticks * 1000000000ULL) / total_time_ns) : 0;
    stats.cpu_efficiency_percent = (target_tick_interval_ns_ > 0) ? 
        (static_cast<double>(stats.avg_generation_time_ns) / target_tick_interval_ns_) * 100.0 : 0.0;
    
    return stats;
}

void HighFrequencyTickGenerator::set_target_frequency(uint32_t ticks_per_second) {
    target_tick_interval_ns_ = (ticks_per_second > 0) ? (1000000000ULL / ticks_per_second) : 1000000ULL;
}

void HighFrequencyTickGenerator::reset_performance_counters() {
    total_ticks_generated_.store(0, std::memory_order_release);
    generation_time_ns_.store(0, std::memory_order_release);
}


MarketDataProcessor::MLFeatures MarketDataProcessor::process_tick(
    const MarketTick& tick, const MarketTick* history_buffer, size_t history_size) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MLFeatures features;
    features.timestamp_ns = tick.timestamp_ns;
    
    
    features.spread_bps = tick.spread_bps;
    features.liquidity_score = std::log(tick.bid_size + tick.ask_size + 1);
    features.venue_preference = static_cast<float>(tick.venue_id) / 10.0f;
    
    
    if (history_buffer && history_size > 1) {
        float price_sum = 0.0f;
        float volume_sum = 0.0f;
        
        
        size_t recent_history = std::min(history_size, static_cast<size_t>(10));
        for (size_t i = history_size - recent_history; i < history_size; ++i) {
            price_sum += history_buffer[i].last_price;
            volume_sum += history_buffer[i].volume;
        }
        
        float avg_price = price_sum / recent_history;
        float avg_volume = volume_sum / recent_history;
        
        features.price_change = (tick.last_price - avg_price) / avg_price;
        features.volume_ratio = tick.volume / std::max(avg_volume, 1.0f);
        
        
        float variance_sum = 0.0f;
        for (size_t i = history_size - recent_history; i < history_size; ++i) {
            float diff = history_buffer[i].last_price - avg_price;
            variance_sum += diff * diff;
        }
        features.volatility_5min = std::sqrt(variance_sum / recent_history);
        
        
        if (recent_history >= 5) {
            float old_price = history_buffer[history_size - 5].last_price;
            features.momentum_1min = (tick.last_price - old_price) / old_price;
        }
    } else {
        
        features.price_change = 0.0f;
        features.volume_ratio = 1.0f;
        features.volatility_5min = 0.02f; 
        features.momentum_1min = 0.0f;
    }
    
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    
    stats_.ticks_processed.fetch_add(1, std::memory_order_relaxed);
    stats_.processing_time_ns.fetch_add(duration_ns, std::memory_order_relaxed);
    stats_.feature_calculations.fetch_add(7, std::memory_order_relaxed); 
    
    return features;
}

MarketDataProcessor::RiskMetrics MarketDataProcessor::calculate_risk_metrics(
    const MLFeatures& features, float position_size) {
    
    RiskMetrics risk;
    
    
    risk.position_risk = std::abs(position_size) * features.volatility_5min * 1000.0f; 
    
    
    risk.market_impact_estimate = std::abs(position_size) * features.spread_bps * 0.1f;
    
    
    risk.execution_cost_estimate = std::abs(position_size) * (features.spread_bps * 0.5f + 0.5f); 
    
    
    risk.risk_limit_exceeded = (risk.position_risk > 10000.0f) || 
                              (std::abs(features.price_change) > 0.05f) || 
                              (features.volatility_5min > 0.10f); 
    
    return risk;
}

MarketDataProcessor::ProcessorStats MarketDataProcessor::get_processor_stats() const {
    uint64_t ticks = stats_.ticks_processed.load(std::memory_order_acquire);
    uint64_t time_ns = stats_.processing_time_ns.load(std::memory_order_acquire);
    
    ProcessorStats stats;
    stats.ticks_per_second = (time_ns > 0) ? 
        static_cast<uint64_t>((ticks * 1000000000ULL) / time_ns) : 0;
    stats.avg_processing_time_ns = (ticks > 0) ? (time_ns / ticks) : 0;
    stats.throughput_efficiency = (stats.avg_processing_time_ns > 0) ? 
        std::min((1000.0 / stats.avg_processing_time_ns) * 100.0, 100.0) : 0.0;
    
    return stats;
}


extern "C" {
    HighFrequencyTickGenerator* create_tick_generator(uint32_t ticks_per_second) {
        return new HighFrequencyTickGenerator(ticks_per_second);
    }
    
    void destroy_tick_generator(HighFrequencyTickGenerator* gen) {
        delete gen;
    }
    
    int generate_tick_c(HighFrequencyTickGenerator* gen, MarketTick* output) {
        if (!gen || !output) return 0;
        *output = gen->generate_tick();
        return 1;
    }
    
    int initialize_symbols_c(HighFrequencyTickGenerator* gen, const char** symbols, size_t symbol_count) {
        if (!gen || !symbols) return 0;
        
        std::vector<std::string> symbol_names;
        for (size_t i = 0; i < symbol_count; ++i) {
            symbol_names.emplace_back(symbols[i]);
        }
        
        gen->initialize_symbols(symbol_names, DEFAULT_VENUES);
        return 1;
    }
    
    MarketDataProcessor* create_processor() {
        return new MarketDataProcessor();
    }
    
    void destroy_processor(MarketDataProcessor* proc) {
        delete proc;
    }
    
    int process_tick_c(MarketDataProcessor* proc, const MarketTick* input, 
                      MarketDataProcessor::MLFeatures* output) {
        if (!proc || !input || !output) return 0;
        *output = proc->process_tick(*input, nullptr, 0);
        return 1;
    }
}

} 
} 