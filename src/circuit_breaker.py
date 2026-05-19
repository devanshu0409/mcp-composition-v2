"""Circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED state machine."""
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2

    _state: CBState = field(default=CBState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _success_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)

    @property
    def state(self) -> CBState:
        return self._state

    def is_open(self) -> bool:
        """Returns True if the call should be blocked (fail fast)."""
        if self._state == CBState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CBState.HALF_OPEN
                self._success_count = 0
                return False
            return True
        return False

    def record_success(self) -> None:
        if self._state == CBState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CBState.CLOSED
                self._failure_count = 0
        elif self._state == CBState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CBState.HALF_OPEN:
            self._state = CBState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CBState.OPEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_ago_s": round(time.monotonic() - self._last_failure_time, 1)
            if self._last_failure_time > 0 else None,
        }
