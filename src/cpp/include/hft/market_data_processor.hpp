#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <memory>
#include <random>
#include <array>
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif

namespace hft {
namespace market_data {

struct MarketTick {
    uint64_t timestamp_ns;
    uint32_t symbol_id;
    float bid_price;
    float ask_price;
    uint32_t bid_size;
    uint32_t ask_size;
    float last_price;
    uint32_t volume;
    uint8_t venue_id;
    float spread_bps;
    
    
    inline float mid_price() const { return (bid_price + ask_price) * 0.5f; }
    inline float spread() const { return ask_price - bid_price; }
    inline bool is_valid() const { return bid_price > 0 && ask_price > bid_price; }
} __attribute__((packed));

struct VenueInfo {
    std::string name;
    float maker_fee;
    float taker_fee;
    float rebate;
    uint32_t base_latency_us;
    uint32_t jitter_range_us;
};

class HighFrequencyTickGenerator {
private:
    
    static constexpr size_t MAX_SYMBOLS = 64;
    static constexpr size_t MAX_VENUES = 8;
    
    struct alignas(64) SymbolState {
        float current_price;
        float volatility;
        uint32_t avg_volume;
        uint32_t tick_multiplier;
        uint64_t last_update_ns;
        float price_trend;
        std::string symbol_name;
    };
    
    alignas(64) std::array<SymbolState, MAX_SYMBOLS> symbols_;
    alignas(64) std::array<VenueInfo, MAX_VENUES> venues_;
    
    
    alignas(64) mutable std::atomic<uint64_t> rng_state_;
    
    
    alignas(64) std::atomic<uint64_t> total_ticks_generated_;
    alignas(64) std::atomic<uint64_t> generation_time_ns_;
    
    size_t num_symbols_;
    size_t num_venues_;
    uint64_t target_tick_interval_ns_;
    
    
    inline uint64_t xorshift64() const {
        uint64_t x = rng_state_.load(std::memory_order_relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        rng_state_.store(x, std::memory_order_relaxed);
        return x;
    }
    
    inline float fast_random_float(float min, float max) const {
        uint64_t r = xorshift64();
        float normalized = static_cast<float>(r & 0xFFFFFF) / 16777216.0f;
        return min + normalized * (max - min);
    }
    
    inline uint32_t fast_random_uint32(uint32_t min, uint32_t max) const {
        return min + (xorshift64() % (max - min + 1));
    }
    
public:
    explicit HighFrequencyTickGenerator(uint32_t target_ticks_per_second = 100);
    
    
    void initialize_symbols(const std::vector<std::string>& symbol_names,
                          const std::vector<VenueInfo>& venue_configs);
    
    
    MarketTick generate_tick();
    
    
    void generate_tick_batch(MarketTick* output_buffer, size_t count);
    
    
    class TickStream {
    private:
        HighFrequencyTickGenerator& generator_;
        uint64_t next_tick_time_ns_;
        bool running_;
        
    public:
        explicit TickStream(HighFrequencyTickGenerator& gen);
        bool next_tick(MarketTick& tick);
        void stop() { running_ = false; }
    };
    
    std::unique_ptr<TickStream> create_stream();
    
    
    struct PerformanceStats {
        uint64_t total_ticks;
        uint64_t avg_generation_time_ns;
        uint64_t ticks_per_second;
        double cpu_efficiency_percent;
    };
    
    PerformanceStats get_performance_stats() const;
    void reset_performance_counters();
    
    
    void set_target_frequency(uint32_t ticks_per_second);
    void update_symbol_volatility(size_t symbol_idx, float new_volatility);
    void update_symbol_price(size_t symbol_idx, float new_price);
};


class MarketDataProcessor {
private:
    struct alignas(64) ProcessingStats {
        std::atomic<uint64_t> ticks_processed{0};
        std::atomic<uint64_t> processing_time_ns{0};
        std::atomic<uint64_t> feature_calculations{0};
    };
    
    ProcessingStats stats_;
    
    
    void calculate_price_features_avx2(const MarketTick* ticks, size_t count, float* features);
    void calculate_volume_features_avx2(const MarketTick* ticks, size_t count, float* features);
    void calculate_spread_features_avx2(const MarketTick* ticks, size_t count, float* features);
    
public:
    struct MLFeatures {
        float price_change;
        float volume_ratio;
        float spread_bps;
        float volatility_5min;
        float momentum_1min;
        float liquidity_score;
        float venue_preference;
        uint64_t timestamp_ns;
    };
    
    
    MLFeatures process_tick(const MarketTick& tick, const MarketTick* history_buffer, size_t history_size);
    
    
    void process_tick_batch(const MarketTick* input_ticks, MLFeatures* output_features, 
                           size_t count, const MarketTick* history_buffer, size_t history_size);
    
    
    struct RiskMetrics {
        float position_risk;
        float market_impact_estimate;
        float execution_cost_estimate;
        bool risk_limit_exceeded;
    };
    
    RiskMetrics calculate_risk_metrics(const MLFeatures& features, float position_size);
    
    
    struct ProcessorStats {
        uint64_t ticks_per_second;
        uint64_t avg_processing_time_ns;
        double throughput_efficiency;
    };
    
    ProcessorStats get_processor_stats() const;
};


extern "C" {
    
    HighFrequencyTickGenerator* create_tick_generator(uint32_t ticks_per_second);
    void destroy_tick_generator(HighFrequencyTickGenerator* gen);
    int generate_tick_c(HighFrequencyTickGenerator* gen, MarketTick* output);
    int initialize_symbols_c(HighFrequencyTickGenerator* gen, const char** symbols, size_t symbol_count);
    
    MarketDataProcessor* create_processor();
    void destroy_processor(MarketDataProcessor* proc);
    int process_tick_c(MarketDataProcessor* proc, const MarketTick* input, 
                      MarketDataProcessor::MLFeatures* output);
}

} 
} 