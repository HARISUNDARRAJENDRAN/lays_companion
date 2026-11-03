"""Pure safety predicates; native identity and readback are supplied by the backend."""
from __future__ import annotations

def in_owned_scope(candidate: dict, owned_window: str, owned_process: int | None) -> bool:
    return bool(owned_window) and candidate.get("window_id") == owned_window and candidate.get("process_id") == owned_process

def verify_readback(expected: str, actual: str | None) -> bool:
    return actual is not None and actual == expected

class RepeatGuard:
    """Blocks repeated object+operation attempts until the preceding effect was verified."""
    def __init__(self): self.pending = None
    def allow(self, object_key: str, operation: str) -> bool:
        return self.pending != (object_key, operation)
    def dispatched(self, object_key: str, operation: str): self.pending = (object_key, operation)
    def verified(self, object_key: str, operation: str, *, no_op: bool = False):
        if self.pending == (object_key, operation) and no_op: self.pending = None
        elif self.pending == (object_key, operation): self.pending = None
    def uncertain(self, object_key: str, operation: str): self.dispatched(object_key, operation)
