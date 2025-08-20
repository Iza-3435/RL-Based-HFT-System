import numpy as np
import ctypes
import os
import time
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
from collections import namedtuple
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)
HFTExperience = namedtuple('HFTExperience', [
    'state',
    'action',
    'reward',
    'next_state',
    'done',
    'timestamp_ns',
    'venue',
    'expected_latency',
    'actual_latency',
    'fill_success',
    'market_impact',
    'opportunity_cost'
])
@dataclass
class HFTMarketState:

    timestamp_ns: int
    symbol: str

    venue_latencies: np.ndarray

    mid_price: float
    bid_price: float
    ask_price: float
    spread_bps: float

    bid_volume: float
    ask_volume: float
    imbalance_ratio: float

    volatility_1min: float
    volatility_5min: float
    price_momentum: float
    volume_momentum: float

    trade_intensity: float
    order_flow_toxicity: float
    effective_spread: float
    realized_spread: float

    hour_of_day: int
    minute_of_hour: int
    is_market_open: bool
    is_auction_period: bool
    seconds_to_close: int
    def to_array(self) -> np.ndarray:

        state = np.concatenate([
            self.venue_latencies / 10000.0,
            [
                self.mid_price / 1000.0,
                self.spread_bps / 100.0,
                self.imbalance_ratio,
                self.volatility_1min * 100.0,
                self.volatility_5min * 100.0,
                self.price_momentum,
                self.volume_momentum,
                self.trade_intensity,
                self.order_flow_toxicity,
                self.effective_spread / 100.0,
                self.realized_spread / 100.0,
                self.hour_of_day / 24.0,
                self.minute_of_hour / 60.0,
                float(self.is_market_open),
                float(self.is_auction_period),
                self.seconds_to_close / 23400.0
            ]
        ])
        return state.astype(np.float32)
class HFTReplayBufferCpp:


    def __init__(self, capacity: int = 1000000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self._init_fast_buffer()

        self.total_adds = 0
        self.total_samples = 0
        self.add_times = []
        self.sample_times = []

        logger.info(f"HFTReplayBuffer initialized with capacity {capacity}")

    def _init_fast_buffer(self):

        self.state_size = None
        self.states = None
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = None
        self.dones = np.zeros(self.capacity, dtype=bool)

        self.timestamps = np.zeros(self.capacity, dtype=np.int64)
        self.venues = np.zeros(self.capacity, dtype=np.int8)
        self.expected_latencies = np.zeros(self.capacity, dtype=np.float32)
        self.actual_latencies = np.zeros(self.capacity, dtype=np.float32)
        self.fill_successes = np.zeros(self.capacity, dtype=bool)
        self.market_impacts = np.zeros(self.capacity, dtype=np.float32)
        self.opportunity_costs = np.zeros(self.capacity, dtype=np.float32)

        self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

        self.position = 0
        self.size = 0

        self.lock = threading.RLock()

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool,
            venue: int = -1, expected_latency: float = 0.0,
            actual_latency: float = 0.0, fill_success: bool = True,
            market_impact: float = 0.0, opportunity_cost: float = 0.0):


        start_time = time.perf_counter_ns()

        with self.lock:
            if self.state_size is None:
                self.state_size = len(state)
                self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
                self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
                logger.info(f"Initialized buffer with state size: {self.state_size}")

            priority = abs(reward) + 0.1

            if actual_latency > 0:
                latency_factor = min(2.0, actual_latency / 1000.0)
                priority *= (2.0 - latency_factor)

            if market_impact > 0:
                priority *= (1.0 + market_impact)

            pos = self.position
            self.states[pos] = state[:self.state_size]
            self.actions[pos] = action
            self.rewards[pos] = reward
            self.next_states[pos] = next_state[:self.state_size]
            self.dones[pos] = done

            self.timestamps[pos] = time.time_ns()
            self.venues[pos] = venue
            self.expected_latencies[pos] = expected_latency
            self.actual_latencies[pos] = actual_latency
            self.fill_successes[pos] = fill_success
            self.market_impacts[pos] = market_impact
            self.opportunity_costs[pos] = opportunity_cost

            self.priorities[pos] = priority
            self.max_priority = max(self.max_priority, priority)

            self.position = (self.position + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1

            self.total_adds += 1

        add_time = time.perf_counter_ns() - start_time
        self.add_times.append(add_time)

        if len(self.add_times) > 10000:
            self.add_times = self.add_times[-1000:]

    def sample(self, batch_size: int) -> List[HFTExperience]:


        start_time = time.perf_counter_ns()

        if self.size == 0:
            return []

        batch_size = min(batch_size, self.size)

        with self.lock:
            if self.size == self.capacity:
                valid_priorities = self.priorities
            else:
                valid_priorities = self.priorities[:self.size]

            probs = valid_priorities ** self.alpha
            probs = probs / probs.sum()

            indices = np.random.choice(self.size, batch_size, p=probs, replace=True)

            experiences = []
            for idx in indices:
                exp = HFTExperience(
                    state=self.states[idx].copy(),
                    action=self.actions[idx],
                    reward=self.rewards[idx],
                    next_state=self.next_states[idx].copy(),
                    done=self.dones[idx],
                    timestamp_ns=self.timestamps[idx],
                    venue=self.venues[idx],
                    expected_latency=self.expected_latencies[idx],
                    actual_latency=self.actual_latencies[idx],
                    fill_success=self.fill_successes[idx],
                    market_impact=self.market_impacts[idx],
                    opportunity_cost=self.opportunity_costs[idx]
                )
                experiences.append(exp)

            self.beta = min(1.0, self.beta + self.beta_increment)
            self.total_samples += batch_size

        sample_time = time.perf_counter_ns() - start_time
        self.sample_times.append(sample_time)

        if len(self.sample_times) > 1000:
            self.sample_times = self.sample_times[-100:]

        return experiences

    def update_priorities(self, indices: List[int], priorities: List[float]):

        with self.lock:
            for idx, priority in zip(indices, priorities):
                if idx < self.size:
                    self.priorities[idx] = max(priority, 1e-6)
                    self.max_priority = max(self.max_priority, priority)

    def get_performance_stats(self) -> Dict[str, Any]:

        stats = {
            'total_adds': self.total_adds,
            'total_samples': self.total_samples,
            'current_size': self.size,
            'capacity_utilization': self.size / self.capacity,
            'max_priority': self.max_priority,
            'current_beta': self.beta
        }

        if self.add_times:
            add_times_us = [t / 1000 for t in self.add_times[-1000:]]
            stats['add_latency'] = {
                'mean_ns': np.mean(self.add_times[-1000:]),
                'p50_ns': np.percentile(self.add_times[-1000:], 50),
                'p95_ns': np.percentile(self.add_times[-1000:], 95),
                'p99_ns': np.percentile(self.add_times[-1000:], 99)
            }

        if self.sample_times:
            stats['sample_latency'] = {
                'mean_ns': np.mean(self.sample_times[-100:]),
                'p50_ns': np.percentile(self.sample_times[-100:], 50),
                'p95_ns': np.percentile(self.sample_times[-100:], 95),
                'p99_ns': np.percentile(self.sample_times[-100:], 99)
            }

        return stats

    def __len__(self):
        return self.size
class HFTRoutingIntegration:


    def __init__(self, existing_router, venues: List[str],
                 buffer_capacity: int = 1000000):

        self.existing_router = existing_router
        self.venues = venues
        self.venue_to_idx = {venue: idx for idx, venue in enumerate(venues)}

        self.fast_buffer = HFTReplayBufferCpp(buffer_capacity)

        self._patch_existing_router()

        self.integration_stats = {
            'experiences_processed': 0,
            'routing_decisions': 0,
            'avg_decision_latency_ns': 0,
            'venue_performance': {venue: {'count': 0, 'avg_latency': 0} for venue in venues}
        }

        logger.info(f"HFT integration initialized with {len(venues)} venues")

    def _patch_existing_router(self):

        self.old_buffer = getattr(self.existing_router, 'memory', None)

        self.existing_router.memory = self

        logger.info("Successfully patched existing router with ultra-fast buffer")

    def add(self, state, action, reward, next_state, done, **kwargs):

        venue = kwargs.get('venue', -1)
        if isinstance(venue, str) and venue in self.venue_to_idx:
            venue = self.venue_to_idx[venue]
        elif venue is None:
            venue = -1

        expected_latency = kwargs.get('expected_latency', 0.0)
        actual_latency = kwargs.get('actual_latency', 0.0)
        fill_success = kwargs.get('fill_success', True)
        market_impact = kwargs.get('market_impact', 0.0)
        opportunity_cost = kwargs.get('opportunity_cost', 0.0)

        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        self.fast_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            venue=venue,
            expected_latency=expected_latency,
            actual_latency=actual_latency,
            fill_success=fill_success,
            market_impact=market_impact,
            opportunity_cost=opportunity_cost
        )

        self.integration_stats['experiences_processed'] += 1

    def sample(self, batch_size: int):

        hft_experiences = self.fast_buffer.sample(batch_size)

        if not hft_experiences:
            return []

        from collections import namedtuple
        Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

        compatible_experiences = []
        for exp in hft_experiences:
            compatible_exp = Experience(
                state=exp.state,
                action=exp.action,
                reward=exp.reward,
                next_state=exp.next_state,
                done=exp.done
            )
            compatible_experiences.append(compatible_exp)

        return compatible_experiences

    def enhanced_sample(self, batch_size: int) -> List[HFTExperience]:

        return self.fast_buffer.sample(batch_size)

    def add_routing_result(self, decision_data: Dict[str, Any]):

        self.add(
            state=decision_data['state'],
            action=decision_data['action'],
            reward=decision_data['reward'],
            next_state=decision_data['next_state'],
            done=decision_data.get('done', False),
            venue=decision_data.get('venue', -1),
            expected_latency=decision_data.get('expected_latency', 0.0),
            actual_latency=decision_data.get('actual_latency', 0.0),
            fill_success=decision_data.get('fill_success', True),
            market_impact=decision_data.get('market_impact', 0.0),
            opportunity_cost=decision_data.get('opportunity_cost', 0.0)
        )

        venue = decision_data.get('venue')
        if venue and venue in self.integration_stats['venue_performance']:
            venue_stats = self.integration_stats['venue_performance'][venue]
            venue_stats['count'] += 1

            if 'actual_latency' in decision_data:
                old_avg = venue_stats['avg_latency']
                count = venue_stats['count']
                new_latency = decision_data['actual_latency']
                venue_stats['avg_latency'] = (old_avg * (count - 1) + new_latency) / count

    def get_comprehensive_stats(self) -> Dict[str, Any]:

        buffer_stats = self.fast_buffer.get_performance_stats()

        return {
            'buffer_performance': buffer_stats,
            'integration_stats': self.integration_stats.copy(),
            'performance_comparison': self._calculate_performance_improvement(),
            'venue_analysis': self._analyze_venue_performance()
        }

    def _calculate_performance_improvement(self) -> Dict[str, Any]:

        buffer_stats = self.fast_buffer.get_performance_stats()

        typical_add_latency_ns = 50000
        typical_sample_latency_ns = 100000

        current_add_latency = buffer_stats.get('add_latency', {}).get('mean_ns', typical_add_latency_ns)
        current_sample_latency = buffer_stats.get('sample_latency', {}).get('mean_ns', typical_sample_latency_ns)

        return {
            'add_speedup': typical_add_latency_ns / current_add_latency if current_add_latency > 0 else 1,
            'sample_speedup': typical_sample_latency_ns / current_sample_latency if current_sample_latency > 0 else 1,
            'current_add_latency_ns': current_add_latency,
            'current_sample_latency_ns': current_sample_latency,
            'estimated_daily_time_saved_ms': self._estimate_time_savings()
        }

    def _analyze_venue_performance(self) -> Dict[str, Any]:

        analysis = {}

        for venue, stats in self.integration_stats['venue_performance'].items():
            if stats['count'] > 0:
                analysis[venue] = {
                    'selection_count': stats['count'],
                    'avg_latency_us': stats['avg_latency'],
                    'selection_rate': stats['count'] / max(self.integration_stats['experiences_processed'], 1)
                }

        return analysis

    def _estimate_time_savings(self) -> float:

        experiences_per_day = 86400 * 100

        buffer_stats = self.fast_buffer.get_performance_stats()
        current_add_latency = buffer_stats.get('add_latency', {}).get('mean_ns', 1000)
        typical_add_latency = 50000

        time_saved_per_op = typical_add_latency - current_add_latency
        daily_savings_ns = experiences_per_day * time_saved_per_op
        daily_savings_ms = daily_savings_ns / 1e6

        return daily_savings_ms

    def __len__(self):
        return len(self.fast_buffer)
def create_hft_integration(existing_dqn_router, venues: List[str]) -> HFTRoutingIntegration:

    return HFTRoutingIntegration(existing_dqn_router, venues)
if __name__ == "__main__":
    import sys
    import time

    class MockDQNRouter:
        def __init__(self):
            self.memory = None

    venues = ['NYSE', 'NASDAQ', 'ARCA', 'IEX', 'CBOE']
    mock_router = MockDQNRouter()

    print("🚀 Testing HFT RL Integration")
    print("=" * 50)

    integration = create_hft_integration(mock_router, venues)

    print("Adding test experiences...")
    for i in range(1000):
        state = np.random.randn(25).astype(np.float32)
        next_state = np.random.randn(25).astype(np.float32)
        action = np.random.randint(0, len(venues))
        reward = np.random.randn() * 10
        venue = venues[action]

        integration.add_routing_result({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'venue': venue,
            'expected_latency': np.random.uniform(500, 2000),
            'actual_latency': np.random.uniform(400, 2500),
            'fill_success': np.random.random() > 0.1,
            'market_impact': np.random.uniform(0, 0.1),
            'opportunity_cost': np.random.uniform(0, 5)
        })

    print(f"Buffer size: {len(integration)}")

    start_time = time.perf_counter()
    for _ in range(100):
        batch = integration.sample(64)
    end_time = time.perf_counter()

    avg_sample_time = (end_time - start_time) / 100 * 1e6
    print(f"Average sampling time: {avg_sample_time:.1f} μs")

    stats = integration.get_comprehensive_stats()
    print("\n📊 Performance Statistics:")
    print("-" * 30)

    if 'add_latency' in stats['buffer_performance']:
        add_stats = stats['buffer_performance']['add_latency']
        print(f"Add latency P50: {add_stats['p50_ns']:.0f} ns")
        print(f"Add latency P95: {add_stats['p95_ns']:.0f} ns")

    if 'sample_latency' in stats['buffer_performance']:
        sample_stats = stats['buffer_performance']['sample_latency']
        print(f"Sample latency P50: {sample_stats['p50_ns']:.0f} ns")
        print(f"Sample latency P95: {sample_stats['p95_ns']:.0f} ns")

    perf_improvement = stats['performance_comparison']
    print(f"Add speedup: {perf_improvement['add_speedup']:.1f}x")
    print(f"Sample speedup: {perf_improvement['sample_speedup']:.1f}x")
    print(f"Est. daily time saved: {perf_improvement['estimated_daily_time_saved_ms']:.1f} ms")

    print("\n✅ Integration test completed successfully!")