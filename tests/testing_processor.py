from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Literal

from agents.tracing import Span, Trace, TracingProcessor

TestSpanProcessorEvent = Literal["trace_start", "trace_end", "span_start", "span_end"]


class SpanProcessorForTests(TracingProcessor):
    """Simple span processor for tests that stores spans and traces in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: list[Span[Any]] = []
        self._traces: list[Trace] = []
        self._events: list[TestSpanProcessorEvent] = []

    def on_trace_start(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)
            self._events.append("trace_start")

    def on_trace_end(self, trace: Trace) -> None:
        with self._lock:
            self._events.append("trace_end")

    def on_span_start(self, span: Span[Any]) -> None:
        with self._lock:
            self._events.append("span_start")

    def on_span_end(self, span: Span[Any]) -> None:
        with self._lock:
            self._spans.append(span)
            self._events.append("span_end")

    def get_ordered_spans(self, including_empty: bool = False) -> list[Span[Any]]:
        with self._lock:
            spans = [s for s in self._spans if including_empty or s.export()]
            def sort_key(s):
                started_at = s.started_at
                if started_at is None:
                    return datetime.min.timestamp()
                if isinstance(started_at, str):
                    return datetime.fromisoformat(started_at).timestamp()
                return started_at.timestamp()
            return sorted(spans, key=sort_key)

    def get_traces(self, including_empty: bool = False) -> list[Trace]:
        with self._lock:
            traces = [t for t in self._traces if including_empty or t.export()]
            return sorted(traces, key=lambda t: t.trace_id)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._traces.clear()
            self._events.clear()

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        with self._lock:
            for span in self._spans:
                span.export()


SPAN_PROCESSOR_TESTING = SpanProcessorForTests()


def fetch_ordered_spans() -> list[Span[Any]]:
    return SPAN_PROCESSOR_TESTING.get_ordered_spans()


def fetch_traces() -> list[Trace]:
    return SPAN_PROCESSOR_TESTING.get_traces()


def fetch_events() -> list[TestSpanProcessorEvent]:
    return SPAN_PROCESSOR_TESTING._events


def assert_no_spans():
    spans = fetch_ordered_spans()
    if spans:
        raise AssertionError(f"Expected 0 spans, got {len(spans)}")


def assert_no_traces():
    traces = fetch_traces()
    if traces:
        raise AssertionError(f"Expected 0 traces, got {len(traces)}")
    assert_no_spans()


def fetch_normalized_spans(
    keep_span_id: bool = False, keep_trace_id: bool = False
) -> list[dict[str, Any]]:
    nodes: dict[tuple[str, str | None], dict[str, Any]] = {}
    traces = []
    for trace_obj in fetch_traces():
        trace = trace_obj.export()
        if not trace:
            continue
        trace.pop("object", None)
        assert trace.get("id", "").startswith("trace_")
        if not keep_trace_id:
            trace.pop("id", None)
        trace = {k: v for k, v in trace.items() if v is not None}
        nodes[(trace_obj.trace_id, None)] = trace
        traces.append(trace)

    if not traces:
        assert_no_traces()

    for span_obj in fetch_ordered_spans():
        span = span_obj.export()
        if not span:
            continue
        span.pop("object", None)
        assert span.get("id", "").startswith("span_")
        if not keep_span_id:
            span.pop("id", None)
        started_at = span.pop("started_at", None)
        ended_at = span.pop("ended_at", None)
        if started_at:
            assert isinstance(datetime.fromisoformat(started_at), datetime)
        if ended_at:
            assert isinstance(datetime.fromisoformat(ended_at), datetime)
        parent_id = span.pop("parent_id", None)
        assert "type" not in span
        span_data = span.pop("span_data", {})
        span_type = span_data.pop("type", "default")
        span = {"type": span_type} | {k: v for k, v in span.items() if v is not None}
        span_data = {k: v for k, v in span_data.items() if v is not None}
        if span_data:
            span["data"] = span_data
        nodes[(span_obj.trace_id, span_obj.span_id)] = span
        parent_node = nodes.get((span.pop("trace_id", ""), parent_id), {})
        parent_node.setdefault("children", []).append(span)
    return traces
