"""Frozen RL3-r5 selectors and safety guards for the Computah Windows runtime."""
from .selector import R5Selector, exact_substring, snap_point
from .guards import RepeatGuard, in_owned_scope, verify_readback

__all__ = ["R5Selector", "exact_substring", "snap_point", "RepeatGuard", "in_owned_scope", "verify_readback"]
