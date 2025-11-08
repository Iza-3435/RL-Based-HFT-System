"""Message queue simulation for venue gateways."""

import time
from collections import deque
from typing import Dict, Deque, Tuple


class MessageQueue:
    """Simulates message queuing at venue gateways."""

    def __init__(self, capacity: int, processing_rate: int) -> None:
        self.capacity = capacity
        self.processing_rate = processing_rate
        self.queue: Deque[Tuple[float, str]] = deque()
        self.last_processed_time = time.time()
        self.queue_full_events = 0

    def add_message(self, message_id: str) -> Tuple[bool, float]:
        """Add message to queue.

        Returns:
            (success, queue_delay_seconds)
        """
        current_time = time.time()
        self._process_completed_messages(current_time)

        if len(self.queue) >= self.capacity:
            self.queue_full_events += 1
            queue_delay = len(self.queue) / self.processing_rate * 2.0
            return False, queue_delay

        self.queue.append((current_time, message_id))
        queue_position = len(self.queue) - 1
        queue_delay = queue_position / self.processing_rate

        return True, queue_delay

    def _process_completed_messages(self, current_time: float) -> None:
        """Remove messages that have been processed."""
        messages_to_process = int((current_time - self.last_processed_time) * self.processing_rate)

        for _ in range(min(messages_to_process, len(self.queue))):
            if self.queue:
                self.queue.popleft()

        self.last_processed_time = current_time

    def get_current_delay(self) -> float:
        """Get current queue delay in seconds."""
        return len(self.queue) / self.processing_rate

    def get_queue_stats(self) -> Dict[str, float]:
        """Get queue performance statistics."""
        return {
            "current_depth": len(self.queue),
            "capacity_utilization": len(self.queue) / self.capacity,
            "current_delay_ms": self.get_current_delay() * 1000,
            "queue_full_events": self.queue_full_events,
        }
