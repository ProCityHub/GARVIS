from __future__ import annotations

import cmath
import hashlib
import math
from collections import Counter, deque
from typing import Any, Sequence

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

Grid = Sequence[Sequence[int]]
Point = tuple[int, int]

# GENESIS-X FP8 DNA-Lattice ARC-AGI-3 Candidate V1
# Parent: Gen8 late52_reverse
# Parent SHA-256: 734a1b1b5ad4177abc611a9fa6e6d8e0dfe33b472e837d5abf7ccc9256f85832
# Public-score reference: 0.27

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_A = 1.0 / PHI
PHI_B = 1.0 / (PHI * PHI)
assert abs(PHI_A + PHI_B - 1.0) < 1e-12

PRUNE_THRESHOLD = 0.27
RETAIN_THRESHOLD = 0.60
EXPLOIT_THRESHOLD = 0.70
EPS = 1e-6

LAMBDA_CANDIDATES = (0.27, 0.60, PHI_A, 0.70)
WEIGHT_FAMILIES = {
    "equal": (0.2, 0.2, 0.2, 0.2, 0.2),
    "phi": tuple(x / 3.0 for x in (1.0, PHI_A, PHI_B, PHI_A, PHI_B)),
    "prime": tuple(p / 28.0 for p in (2.0, 3.0, 5.0, 7.0, 11.0)),
}
for _w in WEIGHT_FAMILIES.values():
    assert abs(sum(_w) - 1.0) < 1e-12

FREQUENCY_BANKS = {
    "prime": (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0),
    "sacred": (396.0, 417.0, 528.0, 639.0, 741.0, 852.0),
    "neural_analog": (2.0, 6.0, 10.0, 20.0, 40.0),
}

OCTAGONAL_ROLES = (
    "observation", "memory", "hypothesis", "simulation",
    "contradiction", "correction", "action_plan", "consolidation",
)

DNA_MEMORY_LOCI = {
    "CORE": "immutable invariants",
    "WORKING": "recent actions",
    "EPISODIC": "witnessed prediction/action/outcome",
    "TRACE": "bounded recurrent states",
    "PROCEDURAL": "bounded state-action outcomes",
    "PROSPECTIVE": "candidate intentions",
    "SIMULATION": "candidate-only projections",
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _latest_grid(frame: FrameData) -> Grid:
    return frame.frame[-1] if frame.frame else []


def _dominant_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return counts.most_common(1)[0][0] if counts else 0


def _frame_signature(grid: Grid) -> str:
    payload = ";".join(",".join(str(cell) for cell in row) for row in grid)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _changed_pixels(before: Grid, after: Grid) -> int:
    if len(before) != len(after):
        return max(sum(map(len, before)), sum(map(len, after)))
    changed = 0
    for a, b in zip(before, after):
        if len(a) != len(b):
            changed += max(len(a), len(b))
        else:
            changed += sum(x != y for x, y in zip(a, b))
    return changed


def _weighted_geometric(values: Sequence[float], weights: Sequence[float]) -> float:
    return math.exp(sum(w * math.log(EPS + _clip01(v)) for v, w in zip(values, weights)))


def _component_targets(grid: Grid, limit: int = 32) -> list[Point]:
    if not grid:
        return []
    h = len(grid)
    w = min((len(row) for row in grid), default=0)
    if w == 0:
        return []
    background = _dominant_color(grid)
    visited: set[Point] = set()
    components: list[tuple[float, Point]] = []
    for y in range(h):
        for x in range(w):
            if (x, y) in visited or grid[y][x] == background:
                continue
            color = grid[y][x]
            stack = [(x, y)]
            visited.add((x, y))
            pts: list[Point] = []
            while stack:
                cx, cy = stack.pop()
                pts.append((cx, cy))
                for nx, ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and grid[ny][nx] == color:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            target = min(pts, key=lambda p: (p[0]-mx)**2 + (p[1]-my)**2)
            span = max(p[0] for p in pts)-min(p[0] for p in pts)+max(p[1] for p in pts)-min(p[1] for p in pts)
            components.append((1.0/len(pts) + 1.0/(1.0+span), target))
    components.sort(key=lambda x: x[0], reverse=True)
    out: list[Point] = []
    seen: set[Point] = set()
    for _, p in components:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= limit:
            break
    return out


class ActionEvidence:
    def __init__(self) -> None:
        self.attempts = 0
        self.changed_pixels = 0
        self.changed_events = 0
        self.level_gains = 0
        self.stagnant = 0

    def classical_score(self) -> float:
        if self.attempts == 0:
            return 4.0
        avg_change = self.changed_pixels / self.attempts
        return 12.0*self.level_gains + min(avg_change/32.0, 3.0) + 1.0/(1.0+self.attempts) - 0.35*self.stagnant

    def salience_vector(self) -> tuple[float, float, float, float, float]:
        n = max(1, self.attempts)
        relevance = _clip01(0.35 + 0.35*min(1.0, self.level_gains/2.0) + 0.30*min(1.0, self.changed_pixels/(32.0*n)))
        novelty = 1.0 if self.attempts == 0 else 1.0/(1.0+self.attempts)
        coherence = 1.0/(1.0+self.stagnant)
        persistence = _clip01(1.0-math.exp(-self.attempts/4.0))
        empirical = _clip01((2.0*self.level_gains + self.changed_events)/n)
        return relevance, novelty, coherence, persistence, empirical


class MyAgent(Agent):
    """FP8 DNA-Lattice descendant of the verified Gen8 late52_reverse parent."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._evidence = {a: ActionEvidence() for a in GameAction}
        self._recent_actions: deque[GameAction] = deque(maxlen=8)
        self._trace_states: deque[str] = deque(maxlen=512)
        self._seen_states: Counter[str] = Counter()
        self._witness: deque[dict[str, Any]] = deque(maxlen=64)
        self._state_action_failures: Counter[tuple[str, GameAction]] = Counter()
        self._weight_gene_reward = {k: 0.0 for k in WEIGHT_FAMILIES}
        self._lambda_gene_reward = {k: 0.0 for k in LAMBDA_CANDIDATES}
        self._frequency_gene_reward = {k: 0.0 for k in FREQUENCY_BANKS}
        self._last_action: GameAction | None = None
        self._last_grid: Grid = []
        self._last_level = 0
        self._last_signature = ""
        self._last_weight_gene = "equal"
        self._last_lambda = LAMBDA_CANDIDATES[0]
        self._last_frequency_gene = "prime"
        self._last_predicted_effective = False
        self._stagnation = 0
        self._contradiction_interrupt = False
        self._click_history: set[Point] = set()
        self._decision_index = 0

    @property
    def name(self) -> str:
        return f"{super().name}.genesis-x-fp8-dna-v1"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @staticmethod
    def _bound_reward(x: float) -> float:
        return max(-8.0, min(8.0, x))

    @staticmethod
    def _select_gene(rewards: dict[Any, float], priority: Sequence[Any]) -> Any:
        order = {v: i for i, v in enumerate(priority)}
        return max(rewards, key=lambda k: (rewards[k], -order.get(k, 999)))

    def _trace_add(self, sig: str) -> None:
        if len(self._trace_states) == self._trace_states.maxlen:
            old = self._trace_states[0]
            self._seen_states[old] -= 1
            if self._seen_states[old] <= 0:
                del self._seen_states[old]
        self._trace_states.append(sig)
        self._seen_states[sig] += 1

    def _prune_failure_memory(self) -> None:
        if len(self._state_action_failures) <= 512:
            return
        excess = len(self._state_action_failures) - 512
        ranked = sorted(self._state_action_failures.items(), key=lambda kv: (kv[1], kv[0][0]))
        for key, _ in ranked[:excess]:
            del self._state_action_failures[key]

    def _learn_from_observation(self, latest_frame: FrameData) -> None:
        grid = _latest_grid(latest_frame)
        sig = _frame_signature(grid)
        self._trace_add(sig)
        if self._last_action is None or not self._last_grid:
            return
        changed = _changed_pixels(self._last_grid, grid)
        level_gain = max(0, latest_frame.levels_completed-self._last_level)
        effective = changed > 0 or level_gain > 0
        ev = self._evidence[self._last_action]
        ev.attempts += 1
        ev.changed_pixels += changed
        ev.level_gains += level_gain
        if changed > 0:
            ev.changed_events += 1
        key = (self._last_signature, self._last_action)
        if effective:
            self._stagnation = 0
            if self._state_action_failures[key] > 0:
                self._state_action_failures[key] -= 1
            if level_gain > 0:
                self._click_history.clear()
        else:
            ev.stagnant += 1
            self._stagnation += 1
            if self._last_signature:
                self._state_action_failures[key] += 1
        contradiction = effective != self._last_predicted_effective
        self._contradiction_interrupt = contradiction
        self._witness.append({
            "raw_pre": self._last_signature,
            "frozen_prediction_effective": self._last_predicted_effective,
            "action": int(self._last_action.value),
            "raw_post": sig,
            "changed_pixels": int(changed),
            "level_gain": int(level_gain),
            "contradiction": contradiction,
        })
        reward = 2.0 if level_gain > 0 else (0.30 if changed > 0 else -0.50)
        if contradiction:
            reward -= 0.20
        self._weight_gene_reward[self._last_weight_gene] = self._bound_reward(self._weight_gene_reward[self._last_weight_gene] + reward)
        self._lambda_gene_reward[self._last_lambda] = self._bound_reward(self._lambda_gene_reward[self._last_lambda] + reward)
        self._frequency_gene_reward[self._last_frequency_gene] = self._bound_reward(self._frequency_gene_reward[self._last_frequency_gene] + reward)
        self._prune_failure_memory()

    @staticmethod
    def _candidate_actions(latest_frame: FrameData) -> list[GameAction]:
        raw = list(latest_frame.available_actions or [])
        if not raw:
            return [a for a in GameAction if a is not GameAction.RESET]
        out: list[GameAction] = []
        for r in raw:
            if isinstance(r, GameAction):
                a = r
            else:
                try:
                    a = GameAction.from_id(int(r))
                except (TypeError, ValueError):
                    continue
            if a is not GameAction.RESET:
                out.append(a)
        return out

    def _active_genes(self) -> tuple[str, float, str]:
        return (
            self._select_gene(self._weight_gene_reward, ("equal", "phi", "prime")),
            float(self._select_gene(self._lambda_gene_reward, LAMBDA_CANDIDATES)),
            self._select_gene(self._frequency_gene_reward, ("prime", "sacred", "neural_analog")),
        )

    def _frequency_coherence(self, action: GameAction, bank: str) -> float:
        t = self._decision_index + 1
        a = max(1, int(action.value))
        vectors = [cmath.exp(1j * 2.0*math.pi*((f*t*a) % 997.0)/997.0) for f in FREQUENCY_BANKS[bank]]
        return _clip01(abs(sum(vectors)/len(vectors)))

    def _phase_relation(self, action: GameAction) -> float:
        s = (self._decision_index % 8)*math.pi/4.0
        a = ((max(1, int(action.value))-1) % 8)*math.pi/4.0
        return _clip01(0.5*(1.0+math.cos(a-s)))

    def _resonance(self, action: GameAction, sig: str, wg: str, lam: float, fg: str) -> tuple[float, dict[str, float]]:
        v = self._evidence[action].salience_vector()
        geometric = _weighted_geometric(v, WEIGHT_FAMILIES[wg])
        O, A, B = v[0], v[4], v[2]
        c_lambda = max(EPS, O)*(max(EPS, A)**lam)*(max(EPS, B)**(1.0-lam))
        failures = self._state_action_failures[(sig, action)]
        contradiction_survival = 1.0/(1.0+failures)
        frequency = self._frequency_coherence(action, fg)
        phase = self._phase_relation(action)
        r = math.sqrt(max(EPS, geometric*c_lambda))
        r *= 0.90 + 0.20*frequency
        if self._stagnation >= 2:
            r *= 0.95 + 0.10*phase
        r *= contradiction_survival
        return _clip01(r), {
            "geometric_salience": geometric,
            "C_lambda": c_lambda,
            "frequency_coherence": frequency,
            "octagonal_phase_relation": phase,
            "contradiction_survival": contradiction_survival,
        }

    @staticmethod
    def _band(r: float) -> str:
        if r < PRUNE_THRESHOLD:
            return "prune"
        if r < RETAIN_THRESHOLD:
            return "echo"
        if r < EXPLOIT_THRESHOLD:
            return "simulate"
        return "strong_candidate"

    def _select_click(self, grid: Grid) -> Point | None:
        for p in _component_targets(grid):
            if p not in self._click_history:
                return p
        attempts = self._evidence[GameAction.ACTION6].attempts
        if attempts < 26:
            return None
        h = len(grid)
        w = min((len(row) for row in grid), default=0)
        if w <= 0 or h <= 0:
            return None
        sx, sy = max(1, w//4), max(1, h//4)
        probes = [(x, y) for y in range(sy//2, h, sy) for x in range(sx//2, w, sx)]
        if not probes:
            return None
        lattice_i = attempts - 26
        if lattice_i < 52:
            return probes[lattice_i % len(probes)]
        return probes[len(probes)-1-(lattice_i % len(probes))]

    def _select_action(self, candidates: list[GameAction], grid: Grid) -> tuple[GameAction, Point | None, str, dict[str, Any]]:
        sig = _frame_signature(grid)
        wg, lam, fg = self._active_genes()
        primary = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
        for a in primary:
            if a in candidates and self._evidence[a].attempts == 0:
                return a, None, "first-pass control exploration", {"weight_gene": wg, "lambda_gene": lam, "frequency_gene": fg, "pruning_band": "explore_unseen"}
        if GameAction.ACTION6 in candidates and self._stagnation >= 2:
            target = self._select_click(grid)
            if target is not None:
                return GameAction.ACTION6, target, "salient-object exploration", {"weight_gene": wg, "lambda_gene": lam, "frequency_gene": fg, "pruning_band": "salient_click"}
        inverse = {GameAction.ACTION1:GameAction.ACTION2, GameAction.ACTION2:GameAction.ACTION1, GameAction.ACTION3:GameAction.ACTION4, GameAction.ACTION4:GameAction.ACTION3}
        prev = self._recent_actions[-1] if self._recent_actions else None
        ranked = []
        for a in candidates:
            if a is GameAction.ACTION7 and self._stagnation < 8:
                continue
            r, diag = self._resonance(a, sig, wg, lam, fg)
            score = self._evidence[a].classical_score() + 1.25*r
            if prev is not None and inverse.get(prev) is a:
                score -= 0.75
            if a is GameAction.ACTION6:
                score += 0.35
            if self._contradiction_interrupt and prev is not None and a is prev:
                score -= 1.25
            band = self._band(r)
            score += {"prune":-1.0, "echo":-0.15, "simulate":0.15, "strong_candidate":0.35}[band]
            ranked.append((score, r, a, diag, band))
        if not ranked:
            a = candidates[0]
            return a, None, "fallback legal action", {"weight_gene":wg, "lambda_gene":lam, "frequency_gene":fg, "pruning_band":"fallback"}
        ranked.sort(key=lambda x: (x[0], x[1], -x[2].value), reverse=True)
        _, r, a, diag, band = ranked[0]
        d = {**diag, "weight_gene":wg, "lambda_gene":lam, "frequency_gene":fg, "resonant_salience":r, "pruning_band":band}
        if a is GameAction.ACTION6:
            target = self._select_click(grid)
            if target is not None:
                return a, target, "FP8 ranked click probe", d
        return a, None, "FP8 ranked action", d

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last_action = None
            self._last_grid = []
            self._last_signature = ""
            self._stagnation = 0
            self._contradiction_interrupt = False
            action = GameAction.RESET
            action.reasoning = {"agent":"GARVIS GENESIS-X FP8 DNA V1", "mode":"reset", "evidence":"environment requires reset"}
            return action
        self._learn_from_observation(latest_frame)
        grid = _latest_grid(latest_frame)
        sig = _frame_signature(grid)
        candidates = self._candidate_actions(latest_frame)
        if not candidates:
            action = GameAction.RESET
            action.reasoning = {"agent":"GARVIS GENESIS-X FP8 DNA V1", "mode":"failsafe", "evidence":"no legal non-reset actions exposed"}
            return action
        action, target, rationale, diag = self._select_action(candidates, grid)
        if target is not None:
            action.set_data({"x":target[0], "y":target[1]})
            self._click_history.add(target)
        elif action.is_complex():
            h = len(grid)
            w = min((len(row) for row in grid), default=1)
            action.set_data({"x":max(0,w//2), "y":max(0,h//2)})
        ev = self._evidence[action]
        predicted_effective = ev.attempts == 0 or ev.level_gains > 0 or (ev.attempts > 0 and ev.changed_events/ev.attempts >= 0.30)
        phase_i = self._decision_index % 8
        action.reasoning = {
            "agent":"GARVIS GENESIS-X FP8 DNA V1",
            "mode":"fp8_dna_lattice",
            "rationale":rationale,
            "stagnation":self._stagnation,
            "state_seen":self._seen_states.get(sig,0),
            "level":latest_frame.levels_completed,
            "octagonal_phase":phase_i,
            "octagonal_role":OCTAGONAL_ROLES[phase_i],
            "contradiction_interrupt":self._contradiction_interrupt,
            "frozen_prediction_effective":predicted_effective,
            "prune_threshold":PRUNE_THRESHOLD,
            "retain_threshold":RETAIN_THRESHOLD,
            "exploit_threshold":EXPLOIT_THRESHOLD,
            **diag,
        }
        self._last_action = action
        self._last_grid = [list(row) for row in grid]
        self._last_level = latest_frame.levels_completed
        self._last_signature = sig
        self._last_predicted_effective = predicted_effective
        self._last_weight_gene = str(diag.get("weight_gene", "equal"))
        self._last_lambda = float(diag.get("lambda_gene", LAMBDA_CANDIDATES[0]))
        self._last_frequency_gene = str(diag.get("frequency_gene", "prime"))
        self._recent_actions.append(action)
        self._decision_index += 1
        return action
