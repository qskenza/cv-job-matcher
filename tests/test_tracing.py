"""Tracing must never break the pipeline when Langfuse isn't configured."""
from matcher import tracing


def test_tracing_is_a_no_op_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not tracing.enabled()
    with tracing.trace_step("step", as_type="generation", model="m", input="x") as obs:
        obs.update(output="y", usage_details={"input": 1})
    tracing.flush()
