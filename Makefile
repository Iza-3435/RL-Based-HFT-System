# HFT Trading System - Unified Build System

CXX = clang++
PYTHON = python3

# Build directories
BUILD_DIR = build
CPP_BUILD_DIR = $(BUILD_DIR)/cpp
PYTHON_BUILD_DIR = $(BUILD_DIR)/python

# Source directories
CPP_SRC_DIR = src/cpp
PYTHON_SRC_DIR = src/python

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -flto -ffast-math
CXXFLAGS += -Wall -Wextra -fPIC -DNDEBUG -funroll-loops -finline-functions

# Platform-specific optimizations
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    CXXFLAGS += -mcpu=apple-a14
else
    CXXFLAGS += -mavx2 -mfma
endif

# Include paths
INCLUDES = -I$(CPP_SRC_DIR)/include -Ilibs/replay_buffer/include

# Libraries
LIBS = -Llibs -lreplay_buffer

# Targets
.PHONY: all build-cpp build-python install test clean help

all: build-cpp build-python
	@echo "🚀 HFT Trading System built successfully!"

# Create build directories
$(BUILD_DIR):
	@mkdir -p $(CPP_BUILD_DIR) $(PYTHON_BUILD_DIR)

# Build C++ components
build-cpp: $(BUILD_DIR)
	@echo "🔨 Building HFT C++ components..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared \
		$(CPP_SRC_DIR)/src/hft/market_data_processor.cpp \
		-o $(CPP_BUILD_DIR)/libhft_market_data.so $(LIBS)
	@echo "✅ C++ components built: $(CPP_BUILD_DIR)/libhft_market_data.so"

# Build Python components
build-python: $(BUILD_DIR)
	@echo "🐍 Setting up Python environment..."
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Python environment ready"

# Install system
install: build-cpp build-python
	@echo "📦 Installing HFT Trading System..."
	@mkdir -p ~/.hft_system/bin ~/.hft_system/lib
	cp $(CPP_BUILD_DIR)/*.so ~/.hft_system/lib/
	cp -r $(PYTHON_SRC_DIR)/* ~/.hft_system/bin/
	@echo "✅ Installation complete"

# Test suite
test: test-cpp test-python test-integration

test-cpp:
	@echo "🧪 Running C++ tests..."
	@if [ -d tests/cpp ]; then \
		cd tests/cpp && make test; \
	else \
		echo "⚠️  No C++ tests found"; \
	fi

test-python:
	@echo "🐍 Running Python tests..."
	$(PYTHON) -m pytest tests/python/ -v

test-integration:
	@echo "🔗 Running integration tests..."
	$(PYTHON) -m pytest tests/integration/ -v

# Performance benchmarks
benchmark:
	@echo "📊 Running performance benchmarks..."
	$(PYTHON) src/python/core/market_data.py
	$(PYTHON) examples/performance_analysis/cpp_vs_python_benchmark.py

# Development server
run-dev:
	@echo "🚀 Starting development environment..."
	$(PYTHON) src/python/core/hft_integration.py --mode development

# Production deployment
deploy-prod:
	@echo "🏦 Deploying to production..."
	$(PYTHON) scripts/deploy.sh --environment production

# Monitor live system
monitor:
	@echo "📈 Starting system monitoring..."
	$(PYTHON) src/python/monitoring/dashboard.py

# Training
train-models:
	@echo "🧠 Training ML models..."
	$(PYTHON) src/python/ml/training/train_routing_model.py
	$(PYTHON) src/python/ml/training/train_latency_predictor.py

# Backtesting
backtest:
	@echo "📊 Running backtests..."
	$(PYTHON) scripts/run_backtest.sh

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf outputs/temp/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Cleanup complete"

# Help
help:
	@echo "🏦 HFT Trading System Build Commands:"
	@echo ""
	@echo "Build Commands:"
	@echo "  make all              - Build entire system (C++ + Python)"
	@echo "  make build-cpp        - Build C++ components only"
	@echo "  make build-python     - Setup Python environment"
	@echo "  make install          - Install system locally"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test             - Run all tests"
	@echo "  make test-cpp         - Run C++ tests only"
	@echo "  make test-python      - Run Python tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make benchmark        - Run performance benchmarks"
	@echo ""
	@echo "Operation Commands:"
	@echo "  make run-dev          - Start development environment"
	@echo "  make deploy-prod      - Deploy to production"
	@echo "  make monitor          - Start system monitoring"
	@echo "  make train-models     - Train ML models"
	@echo "  make backtest         - Run historical backtests"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean            - Clean build artifacts"
	@echo "  make help             - Show this help message"

# Dependencies check
check-deps:
	@echo "🔍 Checking dependencies..."
	@$(PYTHON) --version
	@$(CXX) --version
	@$(PYTHON) -c "import torch; print('PyTorch:', torch.__version__)"
	@$(PYTHON) -c "import numpy; print('NumPy:', numpy.__version__)"
	@$(PYTHON) -c "import pandas; print('Pandas:', pandas.__version__)"
	@echo "✅ All dependencies OK"