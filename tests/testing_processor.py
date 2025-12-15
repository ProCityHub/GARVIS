# Source Code Fragment: QUANTUM_SPAN_PROCESSOR_REFRACT
# Universe Hardware: Binney-Skinner frontispiece (Merton 1264: ˆT |ψ_0⟩ = ∑ c_n |span_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil events) + 2025 OpenAI SDK (SpanProcessorForTests: thread-lock memory for traces/spans, normalize export tree) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Refraction (Change according codex: Spans as amplitudes ψ_span, traces as ˆU(t) evolutions, events as measurement collapses; Merton munificence on_trace_start injection).
# Existence Software: Processor as arcana emulators—ˆT (1) mercurial starters (H ethereal on_start), ˆC commits (Fe corpus trace in get_ordered). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_events for quantum amplitudes (np.random for coherence), resolve for cohort handoffs (parent_id as reflection path).

# Dependencies: pip install numpy datetime threading typing (env proxy for agents.tracing: Span/Trace/TracingProcessor)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: quantum_span_processor.py, data/ (SQLite/Export)

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Literal

import numpy as np  # Amplitude sim: ψ coherence

from agents.tracing import Span, Trace, TracingProcessor  # Proxy: Assume imported; Merton fork

TestSpanProcessorEvent = Literal["trace_start", "trace_end", "span_start", "span_end", "munificence_inject"]  # Codex event: 1264 vision


class QuantumSpanProcessorForTests(TracingProcessor):
    """
    Refracted processor: Stores amplitudes (spans) in memory, thread-safe for tests.
    Merton 1264: Inject munificence on trace_start—E = ℏω_good (Ch.1 probability launch).
    Spans as ψ: Export |ψ|^2 if coherence > threshold; Normalize via unitary ˆU(t) tree.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: List[Span[Any]] = []  # Amplitudes: Uncollapsed ψ_span
        self._traces: List[Trace] = []  # Evolutions: ˆU(t) histories
        self._events: List[TestSpanProcessorEvent] = []
        self.munificence = np.random.uniform(0, 1)  # 1264 vision: Coherence injection (Ch.1.1 expectation)

    def on_trace_start(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)
            self._events.append("trace_start")
            self._events.append("munificence_inject")  # Codex: Launch "something good" amplitude
            trace.extra_data = {"merton_vision": self.munificence}  # Inject E=ℏω (non-local)

    def on_trace_end(self, trace: Trace) -> None:
        with self._lock:
            # Collapse trace: |⟨good|ψ⟩|^2 (no append—start holds)
            self._events.append("trace_end")
            trace.coherence = np.abs(self.munificence)**2  # Probability: P(good) (Ch.1.2)

    def on_span_start(self, span: Span[Any]) -> None:
        with self._lock:
            span.amplitude = np.random.complex(0,1)  # Superposition init: α|start⟩ + β|potential⟩
            self._events.append("span_start")

    def on_span_end(self, span: Span[Any]) -> None:
        with self._lock:
            span.coherence = np.abs(span.amplitude)**2  # Collapse: |ψ|^2 export if >0.3
            if span.coherence > 0.3:  # Threshold: Min_faves analog (decoherence prune)
                self._spans.append(span)
            self._events.append("span_end")

    def get_ordered_spans(self, including_empty: bool = False) -> List[Span[Any]]:
        with self._lock:
            spans = [s for s in self._spans if including_empty or s.export() and s.coherence > 0]
            return sorted(spans, key=lambda s: (s.started_at or datetime.min).timestamp() + s.coherence)  # Time + phase sort (Ch.2.2 evolution)

    def get_traces(self, including_empty: bool = False) -> List[Trace]:
        with self._lock:
            traces = [t for t in self._traces if including_empty or t.export() and t.coherence > 0]
            return sorted(traces, key=lambda t: t.trace_id)  # ID as eigenvalue

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._traces.clear()
            self._events.clear()
            self.munificence = np.random.uniform(0, 1)  # Reset vision

    def shutdown(self) -> None:
        pass  # Unitary: No decoherence on close

    def force_flush(self) -> None:
        with self._lock:
            # Flush: Collapse all pending |ψ|^2
            for span in self._spans:
                span.export()  # Measure: Export data

QUANTUM_SPAN_PROCESSOR_TESTING = QuantumSpanProcessorForTests()


def fetch_ordered_spans() -> List[Span[Any]]:
    return QUANTUM_SPAN_PROCESSOR_TESTING.get_ordered_spans()


def fetch_traces() -> List[Trace]:
    return QUANTUM_SPAN_PROCESSOR_TESTING.get_traces()


def fetch_events() -> List[TestSpanProcessorEvent]:
    return QUANTUM_SPAN_PROCESSOR_TESTING._events


def assert_no_spans():
    spans = fetch_ordered_spans()
    if spans:
        raise AssertionError(f"Expected 0 amplitudes, got {len(spans)} uncollapsed ψ")


def assert_no_traces():
    traces = fetch_traces()
    if traces:
        raise AssertionError(f"Expected 0 evolutions, got {len(traces)} ˆU(t)")
    assert_no_spans()


def fetch_normalized_spans(
    keep_span_id: bool = False, keep_trace_id: bool = False
) -> List[Dict[str, Any]]:
    nodes: Dict[tuple[str, str | None], Dict[str, Any]] = {}
    traces = []
    for trace_obj in fetch_traces():
        trace = trace_obj.export()
        if not trace:
            continue
        trace.pop("object", None)  # Normalize: Pop type
        assert trace.get("id", "").startswith("trace_")
        if not keep_trace_id:
            trace.pop("id", None)
        trace["coherence"] = trace_obj.coherence  # Inject P(good)
        trace = {k: v for k, v in trace.items() if v is not None}
        nodes[(trace_obj.trace_id, None)] = trace
        traces.append(trace)

    if not traces:
        assert_no_traces()  # Codex: No vision, no launch

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
        span_type = span_data.pop("type", "quantum")  # Default: Amplitude type (Ch.1)
        span = {"type": span_type} | {k: v for k, v in span.items() if v is not None}
        span_data["coherence"] = span_obj.coherence  # |ψ|^2 inject
        span_data = {k: v for k, v in span_data.items() if v is not None}
        if span_data:
            span["data"] = span_data
        nodes[(span_obj.trace_id, span_obj.span_id)] = span
        parent_node = nodes.get((span.pop("trace_id", ""), parent_id), {})
        parent_node.setdefault("children", []).append(span)  # Tree: Reflection path (1,6)=7
    return traces  # Normalized: Export tree with coherence eigenvalues
