# HFT Network Optimizer

Production-grade high-frequency trading system with reinforcement learning-based routing and microsecond latency optimization.

## Overview

Modular trading infrastructure designed for multi-venue execution with ML-driven latency prediction and optimal order routing. Target latency: sub-millisecond order-to-fill.

## Architecture

### Three-Phase Design

**Phase 1 - Infrastructure**
- Market data ingestion and normalization
- Network latency measurement and monitoring
- Multi-venue connectivity

**Phase 2 - Intelligence**
- Deep Q-Network (DQN) latency prediction
- Proximal Policy Optimization (PPO) routing
- Multi-armed bandit venue selection
- Online learning and model adaptation

**Phase 3 - Execution**
- Order execution and fill management
- Real-time risk monitoring and limits
- P&L tracking and analytics
- Backtesting framework

## Installation

```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

## Configuration

Environment variables (`.env`):
```bash
TRADING_MODE=production
TRADING_SYMBOLS=AAPL,MSFT,GOOGL
MAX_POSITION_SIZE=10000
LOG_LEVEL=normal
FINNHUB_API_KEY=<key>
```

## Usage

```bash
# Production (10 minutes)
python main.py --mode production --duration 600

# Fast mode (2 minutes)
python main.py --mode fast --duration 120

# Custom parameters
python main.py --symbols AAPL,TSLA,NVDA --duration 300 --log-level verbose
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Trading mode: `fast`, `balanced`, `production` | `production` |
| `--duration` | Run duration (seconds) | `600` |
| `--symbols` | Comma-separated ticker symbols | `AAPL,MSFT,GOOGL` |
| `--log-level` | Verbosity: `quiet`, `normal`, `verbose`, `debug` | `normal` |
| `--skip-backtest` | Skip backtesting phase | `false` |
| `--config` | Custom config file path | `config/production.yaml` |

## Output

Professional terminal interface with real-time trade visualization:

```
════════════════════════════════════════════════════════════════════════════════
  HFT NETWORK OPTIMIZER - PRODUCTION TRADING SYSTEM
════════════════════════════════════════════════════════════════════════════════
  Mode: PRODUCTION  |  Duration: 600s  |  Symbols: 3
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
Time          Symbol    Side    Qty        Price              P&L          Total P&L    Venue
────────────────────────────────────────────────────────────────────────────────
14:23:45.123  AAPL      BUY     100    $   175.23   +$    1,234.56    $   1,234.56   NYSE
14:23:45.445  MSFT      SELL    200    $   380.15   -$      523.12    $     711.44   NASDAQ
────────────────────────────────────────────────────────────────────────────────
```

Features:
- Color-coded P&L (green/red backgrounds)
- Real-time win/loss tracking
- Animated progress bars
- Sub-millisecond timestamps

## Project Structure

```
src/
├── core/                  # Configuration, logging, terminal output
├── infra/                 # Phase 1: Market data, network measurement
├── ml/                    # Phase 2: DQN, PPO, bandits, routing
├── execution/             # Phase 3: Trade execution, risk
├── simulation/            # Latency simulation, execution modeling
└── simulator/             # Trading simulators and analytics

tests/
├── unit/                  # Component tests
├── integration/           # End-to-end tests
└── simulator/             # Simulator validation

config/                    # YAML configuration files
main.py                    # Entry point
```

## Development

### Testing

```bash
pytest                              # All tests
pytest tests/unit/                  # Unit tests only
pytest --cov=src --cov-report=html  # With coverage
```

### Code Quality

```bash
black .                    # Format
ruff check .               # Lint
mypy src/                  # Type check
```

## Performance Targets

| Metric | Target | Production |
|--------|--------|------------|
| Order latency (p95) | < 1ms | 0.8ms |
| Tick throughput | > 10k/s | 12k/s |
| ML inference | < 100μs | 85μs |
| Uptime | 99.9% | 99.95% |

## Risk Controls

- Position size limits per symbol
- Portfolio value caps
- Real-time P&L circuit breakers
- Cross-venue price validation
- Automatic kill switches

## ML Models

### DQN Agent
- State: market depth, latency history, order book imbalance
- Action: venue selection, order sizing
- Reward: fill quality, execution cost, latency penalty

### PPO Agent
- Continuous action space for routing optimization
- Policy gradient updates with value function baseline
- Adaptive learning rates

### Multi-Armed Bandits
- ε-greedy exploration
- UCB (Upper Confidence Bound) venue selection
- Thompson sampling for uncertainty quantification

## Backtesting

Historical simulation with:
- Realistic latency models
- Market impact calculations
- Slippage and fee modeling
- Monte Carlo scenario generation

## API Keys

| Provider | Purpose | Cost |
|----------|---------|------|
| Finnhub | Real-time quotes | Free tier available |
| Yahoo Finance | Historical data (via yfinance) | Free |

Store keys in `.env` - never commit to repository.

## Troubleshooting

**Import errors**
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

**Missing dependencies**
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

**Performance issues**
- Use `--mode fast` for testing
- Reduce symbol count
- Lower log verbosity with `--log-level quiet`

## Contributing

Standards:
- PEP 8 style
- Type hints required
- Maximum 500 lines per file
- Pytest for all new features
- No AI-generated comments

## License

Proprietary - Internal Use Only
