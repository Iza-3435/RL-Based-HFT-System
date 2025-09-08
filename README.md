# HFT Trading System

**High frequency trading platform with reinforcement learning, real-time market data processing, and millisecond latency execution for algorithmic trading.**

## Overview

A production ready HFT system combining C++ performance with Python machine learning for algorithmic trading. Features include RL-based order routing, real-time risk management, and multi-venue execution across major US exchanges this will provide an simulation for HFT system.

**Key Capabilities:**
- Millisecond latency execution
- Reinforcement learning order routing 
- Real-time market data processing (10K+ ticks/sec)
- Multi-venue support (NYSE, NASDAQ, ARCA, IEX, CBOE)
- Professional backtesting and analytics

## Performance Metrics

| Component | Latency | Throughput |
|-----------|---------|------------|
| Market Data Processing | 1-5ms | 10K+ ticks/sec |
| ML Feature Extraction | 5-20ms | - |
| Order Routing Decision | sub 10ms | - |
| Trade Execution | 10-50ms avg | - |
| Risk Calculations | <5ms | Real-time |

## Quick Start

### Installation
```bash
git clone https://github.com/Iza-3435/RL-Based-HFT-System.git
cd RL-Based-HFT-System
pip install -r requirements.txt
```

### Docker Setup
```bash
# Development environment
./configs/deployment/docker-setup.sh setup development

# Production deployment  
./configs/deployment/docker-setup.sh setup production

# Basic setup (core services only)
./configs/deployment/docker-setup.sh setup basic
```

### Basic Usage
```bash
# Run example
python examples/basic_trading_example.py

# Start full system
python run.py

# Run tests
pytest tests/ -v
```

## System Architecture

The platform uses a hybrid C++/Python architecture optimized for ultra-low latency:

- **C++ Core**: Market data processing, order execution, risk calculations
- **Python ML**: Reinforcement learning models, strategy development, analytics
- **Real-time Pipeline**: Sub-microsecond feature extraction and decision making
- **Multi-venue Support**: NYSE, NASDAQ, ARCA, IEX, CBOE integration

### Data Flow Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│Market Data  │───▶│Feature       │───▶│ML Models    │───▶│Routing       │───▶│Execution    │
│Feed (C++)   │    │Extraction    │    │(LSTM/RL)    │    │Decision      │    │Engine       │
│10K+ tps     │    │5-15ms        │    │65-80% acc   │    │<10ms         │    │sim(5-15ms)  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│Order Book   │    │Risk          │    │Strategy     │    │Venue         │    │Trade        │
│Management   │    │Calculation   │    │Selection    │    │Selection     │    │Settlement   │
│5-level depth│    │Real-time     │    │MM/Arb/Mom   │    │NYSE/NASDAQ   │    │Confirmation │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

## Configuration

Configuration files are located in `configs/`:
- `trading/`: Venue settings, risk limits, execution parameters
- `models/`: ML model parameters, training configurations  
- `deployment/`: Docker and production settings

## Key Features

**Machine Learning**
- Deep Q-Network order routing
- Latency prediction models
- Real-time market regime detection
- Technical indicator generation (50+ features)

**Risk Management**
- Real-time position monitoring
- Dynamic risk limits
- Market impact estimation
- Automated circuit breakers

**Analytics**
- Professional backtesting framework
- Performance attribution analysis
- Monte Carlo simulations
- HTML/PDF reporting

## Author

**Sai Shinde** - Complete implementation and development of the HFT Trading System

## License

MIT License - see [LICENSE](LICENSE) file for details.
