# Makefile for HFT Market Data C++ Library
# Ultra-fast compilation with optimization for tick generation

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -flto -ffast-math
CXXFLAGS += -Wall -Wextra -fPIC -DNDEBUG
CXXFLAGS += -funroll-loops -finline-functions

# ARM64-specific optimizations (Apple Silicon)
ifeq ($(shell uname -m),arm64)
    CXXFLAGS += -mcpu=apple-a14 -DARM64_NEON
else
    # x86_64 optimizations
    CXXFLAGS += -mavx2 -mfma
endif

# Include directories
INCLUDES = -Iinclude

# Source and target directories  
SRC_DIR = src/hft
INCLUDE_DIR = include/hft
BUILD_DIR = build

# Target library
TARGET = $(BUILD_DIR)/libhft_market_data.so

# Source files
SOURCES = $(SRC_DIR)/market_data_processor.cpp

# Object files
OBJECTS = $(BUILD_DIR)/market_data_processor.o

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@echo "🔨 Compiling $< with aggressive optimizations..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link shared library
$(TARGET): $(OBJECTS)
	@echo "🔗 Linking high-performance shared library..."
	$(CXX) -shared $(CXXFLAGS) $(OBJECTS) -o $(TARGET)
	@echo "✅ Built $(TARGET) successfully!"

# Main build target
build-cpp: $(TARGET)
	@echo "🚀 C++ HFT Market Data Library ready!"
	@echo "📊 Optimizations: AVX2, FMA, LTO, native arch tuning"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)

# Install library for Python
install: build-cpp
	@echo "📦 Installing library for Python integration..."
	cp $(TARGET) hfft_copy_6/libhft_market_data.so
	@echo "✅ Library installed in hfft_copy_6/"

# Quick performance test
test-cpp: install
	@echo "🔥 Running C++ performance test..."
	cd hfft_copy_6 && python -c "import asyncio; from cpp_market_data import test_cpp_performance; asyncio.run(test_cpp_performance())"

# Development build with debug symbols
debug: CXXFLAGS = -std=c++17 -O0 -g -fPIC -Wall -Wextra
debug: clean $(TARGET)
	@echo "🐛 Debug build complete"

# Benchmark specific build
benchmark: CXXFLAGS += -DBENCHMARK_MODE -pg
benchmark: clean $(TARGET)
	@echo "📊 Benchmark build complete"

# Check compiler and optimization support
check-compiler:
	@echo "🔍 Checking compiler capabilities..."
	@$(CXX) --version
	@echo "AVX2 support:" 
	@$(CXX) -march=native -dM -E - < /dev/null | grep -i avx2 || echo "❌ No AVX2"
	@echo "FMA support:"
	@$(CXX) -march=native -dM -E - < /dev/null | grep -i fma || echo "❌ No FMA"

# Show generated assembly (for optimization verification)
show-asm: $(SRC_DIR)/market_data_processor.cpp
	@echo "📋 Showing optimized assembly for tick generation..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -S $< -o $(BUILD_DIR)/market_data_processor.s
	@head -50 $(BUILD_DIR)/market_data_processor.s

.PHONY: build-cpp clean install test-cpp debug benchmark check-compiler show-asm