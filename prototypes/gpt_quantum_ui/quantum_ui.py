from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

PHI = (1.0 + math.sqrt(5.0)) / 2.0
INV_PHI = 1.0 / PHI
INV_PHI2 = 1.0 / (PHI * PHI)

LATTICE_STATUS = (
    "HYPOTHESIS UNDER TEST — empirical status: NOT_SUPPORTED"
)

EPISTEMIC_LABELS = (
    "OBSERVED",
    "MATHEMATICAL",
    "SIMULATED",
    "HYPOTHESIS",
    "VERIFIED",
    "NOT_SUPPORTED",
)


@dataclass(frozen=True)
class EvidenceEntry:
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class QuantumFrame:
    observer: str
    bridge: str
    proposed_action: str
    coherence: float
    heartbeat: int
    memory_echoes: tuple[str, ...] = ()
    evidence: tuple[EvidenceEntry, ...] = ()
    approval_required: bool = True
    execution_enabled: bool = False
    lattice_status: str = LATTICE_STATUS

    def __post_init__(self) -> None:
        if not 0.0 <= self.coherence <= 1.0:
            raise ValueError("coherence must be within [0,1]")

        if self.heartbeat < 0:
            raise ValueError("heartbeat must be >= 0")

        if self.execution_enabled:
            raise ValueError(
                "Quantum UI v1 is presentation-only; "
                "execution_enabled must remain False"
            )

        for item in self.evidence:
            if item.status not in EPISTEMIC_LABELS:
                raise ValueError(
                    f"unknown epistemic status: {item.status}"
                )


def lattice_weight(
    observer: float,
    actor: float,
    bridge: float,
) -> float:
    """
    Canonical Lattice Law — hypothesis under test.

    C = O^1 * A^(1/phi) * B^(1/phi^2)

    Domain:
        O, A, B > 0

    Mathematical identity:
        1/phi + 1/phi^2 = 1

    Empirical status:
        NOT_SUPPORTED
    """
    values = {
        "observer": observer,
        "actor": actor,
        "bridge": bridge,
    }

    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")

    return (
        observer
        * math.pow(actor, INV_PHI)
        * math.pow(bridge, INV_PHI2)
    )


def sample_frame() -> QuantumFrame:
    return QuantumFrame(
        observer=(
            "User input and verified local context enter "
            "the Observer layer."
        ),
        bridge=(
            "OAB compares memory, evidence, simulation, "
            "constraints, and permissions."
        ),
        proposed_action=(
            "Present a candidate response; protected "
            "execution remains disabled."
        ),
        coherence=0.6180339887,
        heartbeat=1,
        memory_echoes=(
            "Preserve provenance.",
            "Simulation is not evidence.",
            "Protected actions require Adrien approval.",
        ),
        evidence=(
            EvidenceEntry(
                "Current UI state",
                "OBSERVED",
                "Generated locally from the active frame.",
            ),
            EvidenceEntry(
                "Golden-ratio exponent identity",
                "MATHEMATICAL",
                "1/φ + 1/φ² = 1.",
            ),
            EvidenceEntry(
                "Canonical Lattice Law",
                "NOT_SUPPORTED",
                (
                    "Mathematically defined weighting heuristic; "
                    "empirical ontology not established."
                ),
            ),
        ),
    )


def frame_payload(frame: QuantumFrame) -> dict[str, Any]:
    payload = asdict(frame)
    payload["constants"] = {
        "phi": PHI,
        "inv_phi": INV_PHI,
        "inv_phi2": INV_PHI2,
        "exponent_sum": INV_PHI + INV_PHI2,
    }
    payload["lattice_equation"] = (
        "C = O¹ · A^(1/φ) · B^(1/φ²)"
    )
    payload["epistemic_labels"] = EPISTEMIC_LABELS
    payload["actor_gate"] = (
        "APPROVAL REQUIRED"
        if frame.approval_required
        else "PRESENTATION ONLY"
    )
    return payload


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
  name="viewport"
  content="width=device-width, initial-scale=1"
>
<title>GARVIS · GPT Quantum UI</title>
<style>
:root {
  color-scheme: dark;
  --bg: #05070c;
  --panel: rgba(17, 23, 38, .76);
  --line: rgba(149, 177, 255, .28);
  --text: #eef3ff;
  --muted: #9eaac4;
  --pulse: #b5c8ff;
  --warn: #ffd9a8;
  --ok: #b9f3cb;
  --coherence-angle: 222deg;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 50% 0%, #17213d 0, transparent 42%),
    radial-gradient(circle at 15% 90%, #151b31 0, transparent 36%),
    var(--bg);
  color: var(--text);
}

.shell {
  width: min(1180px, calc(100% - 32px));
  margin: 24px auto 56px;
}

header {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: flex-start;
  margin-bottom: 20px;
}

.eyebrow {
  color: var(--muted);
  font-size: 12px;
  letter-spacing: .18em;
  text-transform: uppercase;
}

h1 {
  margin: 7px 0 4px;
  font-size: clamp(30px, 5vw, 58px);
  line-height: .95;
  letter-spacing: -.045em;
}

.subtitle {
  color: var(--muted);
  max-width: 720px;
  line-height: 1.5;
}

.badge {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 9px 13px;
  white-space: nowrap;
  background: rgba(255,255,255,.035);
  font-size: 12px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 14px;
}

.panel {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 22px;
  background: var(--panel);
  backdrop-filter: blur(18px);
  padding: 20px;
  box-shadow: 0 24px 70px rgba(0,0,0,.24);
}

.field-panel {
  grid-column: span 7;
  min-height: 420px;
}

.state-panel {
  grid-column: span 5;
}

.stage {
  grid-column: span 4;
  min-height: 210px;
}

.wide {
  grid-column: span 6;
}

canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.field-overlay {
  position: relative;
  z-index: 2;
  pointer-events: none;
}

.label {
  color: var(--muted);
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: .16em;
}

.big-number {
  font-size: 58px;
  font-variant-numeric: tabular-nums;
  margin-top: 10px;
}

.coherence-ring {
  width: 150px;
  aspect-ratio: 1;
  border-radius: 50%;
  margin-top: 18px;
  background:
    conic-gradient(
      var(--pulse) var(--coherence-angle),
      rgba(255,255,255,.07) 0
    );
  display: grid;
  place-items: center;
  animation: breathe 2.8s ease-in-out infinite;
}

.coherence-ring::before {
  content: "";
  width: 118px;
  aspect-ratio: 1;
  border-radius: 50%;
  background: #0a0e18;
  border: 1px solid var(--line);
}

.heartbeat {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  margin-top: 16px;
  color: var(--ok);
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 22px currentColor;
  animation: beat 1.25s infinite;
}

.stage-title {
  font-size: 25px;
  margin: 8px 0 12px;
}

.stage-copy {
  color: var(--muted);
  line-height: 1.55;
}

.arrow {
  position: absolute;
  right: 18px;
  top: 18px;
  color: var(--muted);
  font-size: 22px;
}

.gate {
  margin-top: 17px;
  border: 1px solid rgba(255,217,168,.32);
  color: var(--warn);
  padding: 10px 12px;
  border-radius: 13px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
}

.law {
  font-size: clamp(19px, 3vw, 30px);
  margin: 13px 0;
  letter-spacing: -.02em;
}

.status {
  color: var(--warn);
  line-height: 1.45;
}

.list {
  display: grid;
  gap: 9px;
  margin-top: 15px;
}

.item {
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 14px;
  padding: 12px 13px;
  background: rgba(255,255,255,.025);
}

.item strong {
  display: block;
  margin-bottom: 4px;
}

.item small {
  color: var(--muted);
  line-height: 1.45;
}

.tag {
  display: inline-block;
  margin-top: 7px;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--pulse);
  font-size: 10px;
  letter-spacing: .08em;
}

.footer-note {
  margin-top: 15px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.55;
}

@keyframes beat {
  0%, 100% { transform: scale(.8); opacity: .55; }
  35% { transform: scale(1.45); opacity: 1; }
}

@keyframes breathe {
  0%, 100% { transform: scale(.97); }
  50% { transform: scale(1.025); }
}

@media (max-width: 850px) {
  .field-panel,
  .state-panel,
  .stage,
  .wide {
    grid-column: span 12;
  }

  header {
    flex-direction: column;
  }
}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div>
      <div class="eyebrow">
        ProCityHub · GARVIS · Observer Actor Bridge
      </div>
      <h1>GPT Quantum UI</h1>
      <div class="subtitle">
        A local software-state visualization for observation,
        memory, simulation, verification and approval-gated
        action presentation.
      </div>
    </div>
    <div class="badge">EXECUTION DISABLED · LOCAL PROTOTYPE</div>
  </header>

  <main class="grid">
    <section class="panel field-panel">
      <canvas id="field"></canvas>
      <div class="field-overlay">
        <div class="label">State field · visualization only</div>
        <div class="heartbeat">
          <span class="dot"></span>
          <span id="heartbeat"></span>
        </div>
      </div>
    </section>

    <section class="panel state-panel">
      <div class="label">Software coherence</div>
      <div class="big-number" id="coherence-value"></div>
      <div class="coherence-ring"></div>
      <div class="footer-note">
        This value is an interface/system metric.
        It is not physical quantum coherence.
      </div>
    </section>

    <section class="panel stage">
      <span class="arrow">→</span>
      <div class="label">01</div>
      <div class="stage-title">Observer</div>
      <div class="stage-copy" id="observer"></div>
    </section>

    <section class="panel stage">
      <span class="arrow">→</span>
      <div class="label">02</div>
      <div class="stage-title">OAB Bridge</div>
      <div class="stage-copy" id="bridge"></div>
    </section>

    <section class="panel stage">
      <div class="label">03</div>
      <div class="stage-title">Actor</div>
      <div class="stage-copy" id="actor"></div>
      <div class="gate" id="actor-gate"></div>
    </section>

    <section class="panel wide">
      <div class="label">Canonical lattice law</div>
      <div class="law" id="lattice-equation"></div>
      <div class="status" id="lattice-status"></div>
      <div class="footer-note" id="identity"></div>
    </section>

    <section class="panel wide">
      <div class="label">Memory echoes</div>
      <div class="list" id="memory-list"></div>
    </section>

    <section class="panel" style="grid-column: span 12">
      <div class="label">Truth / evidence ledger</div>
      <div class="list" id="evidence-list"></div>
      <div class="footer-note">
        Mathematical consistency does not imply physical truth.
        Simulation and imagination are not evidence.
      </div>
    </section>
  </main>
</div>

<script id="garvis-state" type="application/json">
__STATE_JSON__
</script>

<script>
"use strict";

const state = JSON.parse(
  document.getElementById("garvis-state").textContent
);

const setText = (id, value) => {
  document.getElementById(id).textContent = String(value);
};

setText("heartbeat", `HEARTBEAT ${state.heartbeat}`);
setText("observer", state.observer);
setText("bridge", state.bridge);
setText("actor", state.proposed_action);
setText("actor-gate", `${state.actor_gate} · EXECUTION_DISABLED`);
setText("lattice-equation", state.lattice_equation);
setText("lattice-status", state.lattice_status);
setText(
  "identity",
  `1/φ + 1/φ² = ${state.constants.exponent_sum.toFixed(12)}`
);
setText("coherence-value", state.coherence.toFixed(3));

document.documentElement.style.setProperty(
  "--coherence-angle",
  `${Math.max(0, Math.min(1, state.coherence)) * 360}deg`
);

const memoryList = document.getElementById("memory-list");
state.memory_echoes.forEach((echo) => {
  const node = document.createElement("div");
  node.className = "item";
  node.textContent = echo;
  memoryList.appendChild(node);
});

const evidenceList = document.getElementById("evidence-list");
state.evidence.forEach((entry) => {
  const node = document.createElement("div");
  node.className = "item";

  const title = document.createElement("strong");
  title.textContent = entry.label;

  const detail = document.createElement("small");
  detail.textContent = entry.detail;

  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = entry.status;

  node.append(title, detail, tag);
  evidenceList.appendChild(node);
});

const canvas = document.getElementById("field");
const ctx = canvas.getContext("2d");

const resize = () => {
  const box = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(box.width * ratio));
  canvas.height = Math.max(1, Math.floor(box.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
};

window.addEventListener("resize", resize);
resize();

const nodes = Array.from({length: 28}, (_, index) => ({
  phase: (index / 28) * Math.PI * 2,
  ring: 0.22 + ((index * 17) % 60) / 100,
  speed: 0.00014 + ((index * 7) % 9) * 0.000018
}));

const draw = (time) => {
  const box = canvas.getBoundingClientRect();
  const w = box.width;
  const h = box.height;
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.39;

  ctx.clearRect(0, 0, w, h);

  const points = nodes.map((node) => {
    const angle =
      node.phase + time * node.speed * (0.7 + state.coherence);

    return {
      x: cx + Math.cos(angle) * radius * node.ring,
      y: cy + Math.sin(angle * 1.31) * radius * node.ring
    };
  });

  ctx.lineWidth = 1;

  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      const dist = Math.hypot(dx, dy);

      if (dist < 105) {
        ctx.strokeStyle =
          `rgba(181,200,255,${(1 - dist / 105) * 0.15})`;
        ctx.beginPath();
        ctx.moveTo(points[i].x, points[i].y);
        ctx.lineTo(points[j].x, points[j].y);
        ctx.stroke();
      }
    }
  }

  points.forEach((point, index) => {
    const pulse =
      2.4 + Math.sin(time * 0.003 + index) * 1.1;

    ctx.fillStyle = "rgba(210,222,255,.78)";
    ctx.beginPath();
    ctx.arc(point.x, point.y, pulse, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.strokeStyle = "rgba(181,200,255,.42)";
  ctx.beginPath();
  ctx.arc(
    cx,
    cy,
    22 + Math.sin(time * 0.004) * 5,
    0,
    Math.PI * 2
  );
  ctx.stroke();

  requestAnimationFrame(draw);
};

requestAnimationFrame(draw);
</script>
</body>
</html>
"""


def render_html(frame: QuantumFrame) -> str:
    payload = frame_payload(frame)

    state_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    # Prevent user-controlled text from terminating the JSON script block.
    state_json = (
        state_json
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )

    return HTML_TEMPLATE.replace(
        "__STATE_JSON__",
        state_json,
    )


class QuantumUIHandler(BaseHTTPRequestHandler):
    frame: QuantumFrame = sample_frame()

    def do_GET(self) -> None:
        if self.path == "/":
            body = render_html(self.frame).encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif self.path == "/state":
            body = json.dumps(
                frame_payload(self.frame),
                ensure_ascii=False,
            ).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        # Prototype invariant:
        # UI presentation must not execute or approve actions.
        self.send_response(405)
        self.send_header("Allow", "GET")
        self.end_headers()

    def log_message(
        self,
        format: str,
        *args: Any,
    ) -> None:
        print(
            "GARVIS_QUANTUM_UI",
            format % args,
        )


def serve(port: int = 8787) -> None:
    if not 1024 <= port <= 65535:
        raise ValueError("port must be within 1024..65535")

    host = "127.0.0.1"
    server = HTTPServer(
        (host, port),
        QuantumUIHandler,
    )

    print(
        f"GARVIS GPT Quantum UI: http://{host}:{port}"
    )
    print(
        "Presentation-only prototype; "
        "protected execution is disabled."
    )

    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GARVIS GPT Quantum UI local prototype",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
    )
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
