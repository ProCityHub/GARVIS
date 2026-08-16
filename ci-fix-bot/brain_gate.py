"""GARVIS Double-Slit Brain Gate — CLI interface for CI agents

Usage in any GitHub Actions workflow:

    # Before the bot acts — record a prediction
    python3 ci-fix-bot/brain_gate.py predict \\
        --agent ci-fix-bot \\
        --label "fix-yaml-quotes" \\
        --predicted 1.0 \\
        --confidence 0.9

    # After CI runs — observe the result
    python3 ci-fix-bot/brain_gate.py observe \\
        --agent ci-fix-bot \\
        --label "fix-yaml-quotes" \\
        --observed 1.0

    # Run the falsification loop
    python3 ci-fix-bot/brain_gate.py falsify --agent ci-fix-bot

    # Full summary
    python3 ci-fix-bot/brain_gate.py summary --agent ci-fix-bot

    # Self-test
    python3 ci-fix-bot/brain_gate.py self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure we can import the adapter
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from garvis_brain_adapter import BrainAdapter, epistemic_boundary


def cmd_predict(args: argparse.Namespace) -> int:
    brain = BrainAdapter(agent_name=args.agent, repo_path=args.repo)
    brain.predict(
        label=args.label,
        predicted=args.predicted,
        confidence=args.confidence,
    )
    print(json.dumps({
        "status": "PREDICTED",
        "agent": args.agent,
        "label": args.label,
        "predicted": args.predicted,
        "confidence": args.confidence,
        "epistemic_boundary": epistemic_boundary(),
    }, indent=2))
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    brain = BrainAdapter(agent_name=args.agent, repo_path=args.repo)
    result = brain.observe(
        label=args.label,
        observed=args.observed,
        phase=args.phase,
    )
    print(json.dumps({
        "status": result.verdict,
        "agent": result.agent_name,
        "label": result.label,
        "falsified": result.falsified,
        "path_class": result.path_class,
        "qm_residual": result.qm_residual,
        "phi_residual": result.phi_residual,
        "qm_visibility": result.qm_visibility,
        "phi_visibility": result.phi_visibility,
        "epistemic_boundary": result.epistemic_boundary,
    }, indent=2))
    return 1 if result.falsified else 0


def cmd_falsify(args: argparse.Namespace) -> int:
    brain = BrainAdapter(agent_name=args.agent, repo_path=args.repo)
    verdict = brain.falsify()
    print(json.dumps(verdict, indent=2))
    if verdict["verdict"] == "MODEL_FALSIFIED":
        return 1
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    brain = BrainAdapter(agent_name=args.agent, repo_path=args.repo)
    summary = brain.summary()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    brain = BrainAdapter(agent_name=args.agent, repo_path=args.repo)
    result = brain.run_self_test()
    print(json.dumps(result, indent=2))
    if result.get("SELF_TEST") == "PASS":
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GARVIS Double-Slit Brain Gate for CI agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared --repo argument helper
    def add_repo(p):
        p.add_argument("--repo", default=".", help="Repository root path")

    # predict
    p_predict = sub.add_parser("predict", help="Record a prediction before acting")
    p_predict.add_argument("--agent", required=True, help="Agent name")
    p_predict.add_argument("--label", required=True, help="Decision label")
    p_predict.add_argument("--predicted", type=float, default=1.0,
                           help="Predicted outcome amplitude [0, 1]")
    p_predict.add_argument("--confidence", type=float, default=0.9,
                           help="Confidence factor [0, 1]")
    add_repo(p_predict)
    p_predict.set_defaults(func=cmd_predict)

    # observe
    p_observe = sub.add_parser("observe", help="Record observation and run gate")
    p_observe.add_argument("--agent", required=True, help="Agent name")
    p_observe.add_argument("--label", required=True, help="Decision label")
    p_observe.add_argument("--observed", type=float, required=True,
                           help="Observed outcome amplitude [0, 1]")
    p_observe.add_argument("--phase", type=float, default=0.0,
                           help="Phase offset (delta_theta)")
    add_repo(p_observe)
    p_observe.set_defaults(func=cmd_observe)

    # falsify
    p_falsify = sub.add_parser("falsify", help="Run self-falsification loop")
    p_falsify.add_argument("--agent", required=True, help="Agent name")
    add_repo(p_falsify)
    p_falsify.set_defaults(func=cmd_falsify)

    # summary
    p_summary = sub.add_parser("summary", help="Full brain summary")
    p_summary.add_argument("--agent", required=True, help="Agent name")
    add_repo(p_summary)
    p_summary.set_defaults(func=cmd_summary)

    # self-test
    p_st = sub.add_parser("self-test", help="Run Genesis X brain self-test")
    p_st.add_argument("--agent", default="self-test", help="Agent name")
    add_repo(p_st)
    p_st.set_defaults(func=cmd_self_test)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
