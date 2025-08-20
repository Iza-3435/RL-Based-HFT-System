TRADING_CONFIG = {
    # Basic Trading Parameters
    'default_quantity': 100,
    'fee_rate': 0.00003, 
    'rebate_rate': 0.00001,

    # Venue Latency Ranges (microseconds)
    'venue_latency': {
        'NYSE': (800, 1200),
        'NASDAQ': (850, 1300), 
        'IEX': (750, 950),
        'ARCA': (780, 1050),
        'CBOE': (950, 1300)
    },
    

    'venue_weights': {
        'market_making': {'NYSE': 0.25, 'NASDAQ': 0.25, 'ARCA': 0.20, 'CBOE': 0.15, 'IEX': 0.15},
        'momentum': {'NASDAQ': 0.35, 'NYSE': 0.25, 'ARCA': 0.20, 'IEX': 0.15, 'CBOE': 0.05},
        'arbitrage': {'IEX': 0.30, 'ARCA': 0.25, 'NYSE': 0.20, 'NASDAQ': 0.15, 'CBOE': 0.10},
        'default': {'NYSE': 0.30, 'NASDAQ': 0.25, 'ARCA': 0.20, 'IEX': 0.15, 'CBOE': 0.10}
    },
    
    'win_rates': {
        'market_making': 0.62,
        'momentum': 0.58, 
        'arbitrage': 0.65,
        'default': 0.60
    },
    
    # Latency Penalty Thresholds (microseconds)
    'latency_penalties': {
        'general_threshold': 1000,
        'arbitrage_threshold': 800,
        'momentum_threshold': 1200
    }
}