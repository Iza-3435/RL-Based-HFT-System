"""Operational risk management."""

import time
from typing import Dict, Any
from collections import defaultdict, deque
import numpy as np

from src.core.logging_config import get_logger

logger = get_logger()


class OperationalRiskManager:
    """Manage operational risks in HFT trading."""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_order_rate': 1000,
            'max_error_rate': 0.01,
            'max_latency_ms': 10.0,
            'min_heartbeat_interval': 1.0,
            'circuit_breaker_threshold': 100
        }

        self.order_timestamps = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.latency_measurements = deque(maxlen=1000)
        self.last_heartbeat = defaultdict(float)

        self.system_healthy = True
        self.venue_status = defaultdict(lambda: True)
        self.error_messages = deque(maxlen=100)

    def check_order_rate(self, timestamp: float) -> bool:
        """Check if order rate is within limits."""
        self.order_timestamps.append(timestamp)

        cutoff = timestamp - 1.0
        recent_orders = sum(1 for t in self.order_timestamps if t > cutoff)

        if recent_orders > self.config['max_order_rate']:
            self._log_error(
                'order_rate_exceeded',
                f"Order rate {recent_orders}/s exceeds limit"
            )
            return False

        return True

    def record_latency(self, latency_ms: float, venue: str):
        """Record and check latency measurement."""
        self.latency_measurements.append({
            'timestamp': time.time(),
            'latency_ms': latency_ms,
            'venue': venue
        })

        if latency_ms > self.config['max_latency_ms']:
            self._log_error(
                'high_latency',
                f"High latency for {venue}: {latency_ms:.1f}ms"
            )

        recent_venue_latencies = [
            m['latency_ms'] for m in self.latency_measurements
            if m['venue'] == venue and m['timestamp'] > time.time() - 60
        ]

        if recent_venue_latencies:
            avg_latency = np.mean(recent_venue_latencies)
            if avg_latency > self.config['max_latency_ms']:
                self.venue_status[venue] = False
                self._log_error(
                    'venue_degraded',
                    f"Venue {venue} degraded: {avg_latency:.1f}ms"
                )

    def record_error(
        self,
        error_type: str,
        message: str,
        venue: str = None
    ):
        """Record trading error."""
        self.error_counts[error_type] += 1

        error_entry = {
            'timestamp': time.time(),
            'type': error_type,
            'message': message,
            'venue': venue
        }

        self.error_messages.append(error_entry)

        total_errors = sum(self.error_counts.values())
        if total_errors > self.config['circuit_breaker_threshold']:
            self.system_healthy = False
            self._log_error(
                'circuit_breaker_triggered',
                f"Circuit breaker triggered: {total_errors} errors"
            )

    def update_heartbeat(self, venue: str):
        """Update venue heartbeat."""
        self.last_heartbeat[venue] = time.time()

    def check_heartbeats(self) -> Dict[str, bool]:
        """Check all venue heartbeats."""
        current_time = time.time()
        heartbeat_status = {}

        for venue, last_beat in self.last_heartbeat.items():
            time_since_beat = current_time - last_beat
            is_alive = time_since_beat < self.config['min_heartbeat_interval'] * 2

            heartbeat_status[venue] = is_alive

            if not is_alive:
                self.venue_status[venue] = False
                self._log_error(
                    'heartbeat_timeout',
                    f"Heartbeat timeout for {venue}: {time_since_beat:.1f}s"
                )

        return heartbeat_status

    def _log_error(self, error_type: str, message: str):
        """Log operational error."""
        logger.error(f"OPERATIONAL RISK: [{error_type}] {message}")

    def get_health_report(self) -> Dict[str, Any]:
        """Get system health report."""
        total_orders = len(self.order_timestamps)
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / max(total_orders, 1)

        if self.latency_measurements:
            avg_latency = np.mean([
                m['latency_ms'] for m in self.latency_measurements
            ])
            p99_latency = np.percentile([
                m['latency_ms'] for m in self.latency_measurements
            ], 99)
        else:
            avg_latency = 0
            p99_latency = 0

        return {
            'system_healthy': self.system_healthy,
            'venue_status': dict(self.venue_status),
            'metrics': {
                'error_rate': error_rate,
                'total_errors': total_errors,
                'avg_latency_ms': avg_latency,
                'p99_latency_ms': p99_latency,
                'order_count': total_orders
            },
            'error_breakdown': dict(self.error_counts),
            'recent_errors': list(self.error_messages)[-10:]
        }
