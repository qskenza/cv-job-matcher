"""Optional Langfuse tracing.

If Langfuse isn't installed or its keys aren't set, every call here does nothing,
so the project and the tests work without a Langfuse account.
"""
import os
from contextlib import contextmanager

try:
    from langfuse import get_client
except ImportError:
    get_client = None

_client = None


def enabled() -> bool:
    return (
        get_client is not None
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def _langfuse():
    global _client
    if _client is None:
        _client = get_client()
    return _client


class _NoOp:
    """Stands in for a Langfuse observation when tracing is off."""

    def update(self, **kwargs) -> None:
        pass


@contextmanager
def trace_step(name: str, as_type: str = "span", **kwargs):
    """Records one step (span, LLM generation or embedding call) in Langfuse.

    Steps opened inside another step are nested under it, so one script run
    shows up as one trace tree. Errors raised inside are recorded automatically.
    """
    if not enabled():
        yield _NoOp()
        return
    with _langfuse().start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


def flush() -> None:
    """Sends buffered traces. Call it before a script exits."""
    if enabled():
        _langfuse().flush()
