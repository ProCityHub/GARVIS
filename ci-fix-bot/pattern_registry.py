"""
GARVIS CI Fix Pattern Registry

A durable learning system for CI failure patterns. Each pattern goes through
a lifecycle: DETECTED → DIAGNOSED → REVIEWED → TESTED → PROMOTED.

The bot does NOT silently learn arbitrary repair actions. New patterns become
automatic only after evidence that the diagnosis and fix are reproducible,
reviewed, and tested.

Lifecycle:
  1. DETECTED   — Bot sees a new failure pattern (fingerprint doesn't match any known pattern)
  2. DIAGNOSED  — Bot or human identifies the root cause and proposed fix
  3. REVIEWED   — Human reviews and approves the fix approach
  4. TESTED     — A regression test is added proving the fix works
  5. PROMOTED   — Fix is added to fix_engine.py and becomes automatic

Storage: GitHub Issues with labels tracking the lifecycle stage.
         The registry itself is a JSON file in the repo for programmatic access.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# Fingerprinting — normalize a failure log into a stable hash
# ============================================================

# Patterns to normalize away variable parts (file paths, line numbers, SHAs, etc.)
_NORMALIZE_PATTERNS = [
    # File paths with line numbers: file.py:42 → {file}:{line}
    (re.compile(r'[\w/.-]+\.py:\d+'), '{file}:{line}'),
    # Absolute paths: /home/runner/work/repo/repo/ → {repo_root}/
    (re.compile(r'/[\w/.-]+/work/[\w/.-]+/'), '{repo_root}/'),
    # SHA hashes: abc123def456 → {sha}
    (re.compile(r'\b[0-9a-f]{12,40}\b'), '{sha}'),
    # Line numbers in YAML errors: .yml:5 → .yml:{line}
    (re.compile(r'(\.ya?ml):\d+'), r'\1:{line}'),
    # Numbers in error counts: "Found 3 errors" → "Found {N} errors"
    (re.compile(r'Found \d+ error'), 'Found {N} error'),
    # Timestamps
    (re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'), '{timestamp}'),
    # Process IDs
    (re.compile(r'pid \d+'), 'pid {N}'),
    # Memory addresses
    (re.compile(r'0x[0-9a-f]+'), '0x{addr}'),
]


def fingerprint_failure(logs: str) -> str:
    """Create a stable fingerprint from a failure log.

    Normalizes away variable parts (paths, line numbers, SHAs) so the same
    error class produces the same fingerprint regardless of where it occurs.
    """
    normalized = logs
    for pattern, replacement in _NORMALIZE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)

    # Take the first 50 lines (enough to capture the error, not the noise)
    lines = normalized.split('\n')[:50]
    joined = '\n'.join(lines)

    return hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]


def extract_error_signature(logs: str) -> str:
    """Extract a human-readable error signature from logs.

    Returns the first meaningful error line, normalized.
    """
    error_keywords = [
        'error:', 'Error:', '##[error]', 'Traceback',
        'FAILED', 'Exception', 'assert',
    ]

    for line in logs.split('\n'):
        stripped = line.strip()
        # Remove timestamps and prefixes
        stripped = re.sub(r'^\d{4}-\d{2}-\d{2}T[\d:.Z]+Z\s*', '', stripped)
        for kw in error_keywords:
            if kw in stripped and 'DEP0169' not in stripped and 'url.parse' not in stripped:
                # Truncate to a reasonable length
                return stripped[:200]

    # Fallback: first non-empty line
    for line in logs.split('\n'):
        if line.strip():
            return line.strip()[:200]
    return 'unknown error'


# ============================================================
# Pattern Registry — the JSON store
# ============================================================

@dataclass
class PatternRecord:
    fingerprint: str
    signature: str
    status: str  # DETECTED | DIAGNOSED | REVIEWED | TESTED | PROMOTED
    pattern_name: str | None = None  # name in fix_engine.py, once promoted
    detected_count: int = 1
    first_seen: str = ''
    last_seen: str = ''
    diagnosis: str | None = None
    fix_description: str | None = None
    fix_files_changed: list[str] = field(default_factory=list)
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    regression_test: str | None = None  # path to test file
    issue_number: int | None = None  # GitHub issue tracking this pattern
    related_prs: list[int] = field(default_factory=list)


def load_registry(repo_root: Path) -> dict[str, Any]:
    """Load the pattern registry from the repo."""
    registry_path = repo_root / 'ci-fix-bot' / 'pattern_registry.json'
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {'patterns': {}, 'version': 1, 'total_detected': 0}


def save_registry(repo_root: Path, registry: dict[str, Any]) -> Path:
    """Save the pattern registry to the repo."""
    registry_path = repo_root / 'ci-fix-bot' / 'pattern_registry.json'
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    return registry_path


def record_failure(
    logs: str,
    repo_root: Path,
    recognized: bool,
    fix_applied: bool,
    fix_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a CI failure in the registry.

    Returns the pattern record and whether this is a new pattern.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    fingerprint = fingerprint_failure(logs)
    signature = extract_error_signature(logs)

    registry = load_registry(repo_root)
    patterns: dict[str, Any] = registry.get('patterns', {})

    is_new = fingerprint not in patterns

    if is_new:
        # New pattern — register it
        record = PatternRecord(
            fingerprint=fingerprint,
            signature=signature,
            status='DETECTED',
            detected_count=1,
            first_seen=now,
            last_seen=now,
        )
        if recognized and fix_applied and fix_result:
            # Known pattern that was auto-fixed — already promoted
            record.status = 'PROMOTED'
            record.pattern_name = fix_result.get('pattern', 'unknown')
            record.fix_description = fix_result.get('description', '')
            record.fix_files_changed = fix_result.get('files_changed', [])
        patterns[fingerprint] = {
            'fingerprint': record.fingerprint,
            'signature': record.signature,
            'status': record.status,
            'pattern_name': record.pattern_name,
            'detected_count': record.detected_count,
            'first_seen': record.first_seen,
            'last_seen': record.last_seen,
            'diagnosis': record.diagnosis,
            'fix_description': record.fix_description,
            'fix_files_changed': record.fix_files_changed,
            'reviewed_by': record.reviewed_by,
            'reviewed_at': record.reviewed_at,
            'regression_test': record.regression_test,
            'issue_number': record.issue_number,
            'related_prs': record.related_prs,
        }
        registry['total_detected'] = registry.get('total_detected', 0) + 1
    else:
        # Existing pattern — update last seen and count
        entry = patterns[fingerprint]
        entry['last_seen'] = now
        entry['detected_count'] = entry.get('detected_count', 1) + 1

    registry['patterns'] = patterns
    save_registry(repo_root, registry)

    return {
        'fingerprint': fingerprint,
        'is_new': is_new,
        'status': patterns[fingerprint]['status'],
        'detected_count': patterns[fingerprint]['detected_count'],
        'signature': signature,
    }


# ============================================================
# Regression test template — generated when a pattern is tested
# ============================================================

REGRESSION_TEST_TEMPLATE = '''#!/usr/bin/env python3
"""
Regression test for CI Fix Pattern: {pattern_name}

Error signature: {signature}
Fingerprint: {fingerprint}

This test proves that the fix for this failure pattern works correctly.
It was created as part of the pattern promotion lifecycle:
  DETECTED → DIAGNOSED → REVIEWED → TESTED → PROMOTED

Generated by GARVIS CI Fix Bot Pattern Registry.
"""

import pytest
from pathlib import Path
from ci_fix_bot.fix_engine import {fix_function}


class Test{test_class_name}:
    """Regression tests for the {pattern_name} fix pattern."""

    def test_recognizes_error_pattern(self):
        """The fix function should recognize this error pattern."""
        logs = {logs_literal}
        result = {fix_function}(logs, Path("/tmp"))
        assert result.fixed == True

    def test_fix_is_deterministic(self):
        """The fix should produce the same output every time."""
        logs = {logs_literal}
        result1 = {fix_function}(logs, Path("/tmp"))
        result2 = {fix_function}(logs, Path("/tmp"))
        assert result1.description == result2.description

    def test_no_false_positive_on_clean_logs(self):
        """The fix should not trigger on logs without this error pattern."""
        clean_logs = "All tests passed.\\nNo errors found."
        result = {fix_function}(clean_logs, Path("/tmp"))
        assert result.fixed == False
'''


def generate_regression_test(
    pattern_name: str,
    fingerprint: str,
    signature: str,
    logs_sample: str,
    fix_function: str,
    output_path: Path,
) -> Path:
    """Generate a regression test file for a promoted pattern."""
    test_class_name = ''.join(
        word.capitalize() for word in pattern_name.split('_')
    )
    logs_literal = repr(logs_sample)

    content = REGRESSION_TEST_TEMPLATE.format(
        pattern_name=pattern_name,
        signature=signature,
        fingerprint=fingerprint,
        test_class_name=test_class_name,
        logs_literal=logs_literal,
        fix_function=fix_function,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


# ============================================================
# Main entry point — called by the CI Fix Bot workflow
# ============================================================

def main() -> None:
    """
    Usage:
        python3 pattern_registry.py record <logs_file> <repo_root>
            [--recognized] [--fix-result <json>]
        python3 pattern_registry.py list <repo_root>
        python3 pattern_registry.py promote <fingerprint> <repo_root>
            [--reviewer <name>] [--test-path <path>]
    """
    if len(sys.argv) < 2:
        print("Usage: pattern_registry.py <command> [args]")
        print("Commands: record, list, promote, generate-test")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'record':
        logs_path = sys.argv[2]
        repo_root = Path(sys.argv[3])
        recognized = '--recognized' in sys.argv
        fix_result = None
        if '--fix-result' in sys.argv:
            idx = sys.argv.index('--fix-result')
            fix_result = json.loads(sys.argv[idx + 1])

        with open(logs_path, errors='replace') as f:
            logs = f.read()

        result = record_failure(
            logs, repo_root,
            recognized=recognized,
            fix_applied=fix_result is not None and fix_result.get('fixed', False),
            fix_result=fix_result,
        )
        print(json.dumps(result, indent=2))

    elif command == 'list':
        repo_root = Path(sys.argv[2])
        registry = load_registry(repo_root)
        patterns = registry.get('patterns', {})

        if not patterns:
            print("No patterns recorded yet.")
            return

        print(f"\nCI Fix Pattern Registry — {len(patterns)} patterns\n")
        print(f"{'Status':12s} {'Count':>5s}  {'Pattern':25s}  Signature")
        print('-' * 100)
        for fp, p in sorted(
            patterns.items(),
            key=lambda x: x[1].get('last_seen', ''),
            reverse=True,
        ):
            status = p.get('status', 'UNKNOWN')
            count = p.get('detected_count', 0)
            name = p.get('pattern_name') or fp[:8]
            sig = p.get('signature', '')[:60]
            print(f"{status:12s} {count:5d}  {name:25s}  {sig}")

    elif command == 'promote':
        fingerprint = sys.argv[2]
        repo_root = Path(sys.argv[3])
        reviewer = None
        test_path = None

        if '--reviewer' in sys.argv:
            idx = sys.argv.index('--reviewer')
            reviewer = sys.argv[idx + 1]
        if '--test-path' in sys.argv:
            idx = sys.argv.index('--test-path')
            test_path = sys.argv[idx + 1]

        registry = load_registry(repo_root)
        patterns = registry.get('patterns', {})

        if fingerprint not in patterns:
            print(f"Pattern {fingerprint} not found in registry.")
            sys.exit(1)

        patterns[fingerprint]['status'] = 'PROMOTED'
        if reviewer:
            patterns[fingerprint]['reviewed_by'] = reviewer
            from datetime import datetime, timezone
            patterns[fingerprint]['reviewed_at'] = datetime.now(timezone.utc).isoformat()
        if test_path:
            patterns[fingerprint]['regression_test'] = test_path

        registry['patterns'] = patterns
        save_registry(repo_root, registry)
        print(f"Promoted pattern {fingerprint} to PROMOTED.")

    elif command == 'generate-test':
        fingerprint = sys.argv[2]
        repo_root = Path(sys.argv[3])
        output_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None
        if not output_path:
            output_path = repo_root / 'tests' / 'ci_fix_bot' / f'test_{fingerprint[:8]}.py'

        registry = load_registry(repo_root)
        patterns = registry.get('patterns', {})

        if fingerprint not in patterns:
            print(f"Pattern {fingerprint} not found in registry.")
            sys.exit(1)

        p = patterns[fingerprint]
        generate_regression_test(
            pattern_name=p.get('pattern_name', 'unknown'),
            fingerprint=fingerprint,
            signature=p.get('signature', ''),
            logs_sample=p.get('diagnosis', ''),
            fix_function='fix_' + (p.get('pattern_name') or 'unknown'),
            output_path=output_path,
        )
        print(f"Generated regression test at {output_path}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
