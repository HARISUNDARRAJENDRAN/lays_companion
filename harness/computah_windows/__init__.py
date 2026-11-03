"""Windows semantic runtime inspired by Computah's MIT-licensed typed core.

The runtime owns validation and verification; agent-desktop owns native UIA calls.
"""
from .runtime import Runtime, State, Candidate, Selector, source_span

__all__ = ["Runtime", "State", "Candidate", "Selector", "source_span"]
