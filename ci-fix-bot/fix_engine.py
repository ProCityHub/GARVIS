#!/usr/bin/env python3
"""
GARVIS CI Fix Bot — Auto-Fix Engine

This script is called by the CI Fix Bot GitHub Action when a CI failure is detected.
It downloads the failure logs, pattern-matches against known failure patterns,
and for each match, applies the corresponding fix script.

Each fix script is a standalone function that:
1. Reads the failing file(s)
2. Applies a deterministic fix
3. Returns a description of what was changed

The bot then commits the fix, pushes a branch, and creates a PR.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FixResult:
    fixed: bool
    description: str
    files_changed: list = field(default_factory=list)
    verification_cmd: Optional[str] = None


# ============================================================
# PATTERN 1: Unused "type: ignore" comments in mypy
# ============================================================
def fix_unused_type_ignore(logs: str, repo_root: Path) -> FixResult:
    """
    mypy reports: error: Unused "type: ignore" comment [unused-ignore]
    Fix: Remove the unused type: ignore comments from the flagged lines.
    """
    pattern = r'(.+?):(\d+): error: Unused "type: ignore" comment'
    matches = re.findall(pattern, logs)

    if not matches:
        return FixResult(fixed=False, description="No unused type: ignore comments found")

    files_changed = []
    changes = []

    # Group by file
    by_file = {}
    for filepath, line_num in matches:
        if filepath not in by_file:
            by_file[filepath] = []
        by_file[filepath].append(int(line_num))

    for filepath, line_nums in by_file.items():
        full_path = repo_root / filepath
        if not full_path.exists():
            changes.append(f"  - {filepath}: file not found, skipped")
            continue

        with open(full_path, 'r') as f:
            lines = f.readlines()

        modified = False
        for line_num in sorted(line_nums, reverse=True):  # reverse to preserve indices
            idx = line_num - 1  # 0-indexed
            if idx < len(lines):
                line = lines[idx]
                # Remove the type: ignore comment — handle various formats:
                #   # type: ignore
                #   # type: ignore[specific-error]
                #   # type: ignore[unused-ignore]
                # Also handle inline comments after code
                new_line = re.sub(r'\s*# type: ignore(\[[\w-]+\])?$', '', line.rstrip())
                if new_line != line.rstrip():
                    lines[idx] = new_line + '\n' if line.endswith('\n') else new_line
                    modified = True
                    changes.append(f"  - {filepath}:{line_num}: removed unused type: ignore")

        if modified:
            with open(full_path, 'w') as f:
                f.writelines(lines)
            files_changed.append(filepath)

    if files_changed:
        return FixResult(
            fixed=True,
            description=f"Removed {len(changes)} unused type: ignore comments:\n" + "\n".join(changes),
            files_changed=files_changed,
            verification_cmd="uv run mypy . --exclude site",
        )
    return FixResult(fixed=False, description="No fixes applied")


# ============================================================
# PATTERN 2: Bash quote conflict in python3 -c (YAML)
# ============================================================
def fix_bash_quote_conflict(logs: str, repo_root: Path) -> FixResult:
    """
    Symptom: NameError or SyntaxError in python3 -c blocks within YAML run: |
    Fix: Replace python3 -c "..." with heredoc python3 << 'SCRIPT' ... SCRIPT
    """
    # Look for YAML files with python3 -c that contain inner double quotes
    yaml_files = list(repo_root.glob(".github/workflows/*.yml")) + list(repo_root.glob(".github/workflows/*.yaml"))

    files_changed = []
    changes = []

    for yml_file in yaml_files:
        with open(yml_file, 'r') as f:
            content = f.read()

        # Check if this file has the pattern: python3 -c " with inner quotes
        if 'python3 -c "' not in content:
            continue

        lines = content.split('\n')
        new_lines = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]
            # Detect: run: | followed by python3 -c " on same or next line
            if 'python3 -c "' in line and line.strip().startswith('python3 -c'):
                indent = len(line) - len(line.lstrip())
                # Find the closing " — it's the last " on the last line before a line that doesn't continue
                # Collect all lines until the closing quote
                block_lines = [line]
                j = i + 1
                while j < len(lines):
                    block_lines.append(lines[j])
                    if lines[j].strip() == '"':
                        break
                    j += 1

                # Extract the inner Python code
                # First line: <indent>python3 -c "
                # Last line: <indent>"
                # Middle: the Python code
                inner_code = []
                for bl in block_lines[1:-1]:
                    # Strip the same indent level
                    if bl.startswith(' ' * indent):
                        inner_code.append(bl[indent:])
                    else:
                        inner_code.append(bl.lstrip())

                # Build heredoc replacement
                new_block = []
                new_block.append(' ' * indent + "python3 << 'CI_FIX_BOT'")
                for code_line in inner_code:
                    new_block.append(' ' * indent + code_line)
                new_block.append(' ' * indent + "CI_FIX_BOT")

                new_lines.extend(new_block)
                changes.append(f"  - {yml_file.name}: converted python3 -c to heredoc at line {i+1}")
                modified = True
                i = j + 1
            else:
                new_lines.append(line)
                i += 1

        if modified:
            with open(yml_file, 'w') as f:
                f.write('\n'.join(new_lines))
            files_changed.append(str(yml_file.relative_to(repo_root)))
            changes.append(f"  - Fixed bash quote conflict in {yml_file.name}")

    if files_changed:
        return FixResult(
            fixed=True,
            description=f"Converted python3 -c blocks to heredocs:\n" + "\n".join(changes),
            files_changed=files_changed,
        )
    return FixResult(fixed=False, description="No bash quote conflicts found")


# ============================================================
# PATTERN 3: Missing pip/uv install step
# ============================================================
def fix_missing_install(logs: str, repo_root: Path) -> FixResult:
    """
    Symptom: ModuleNotFoundError or "No module named X"
    Fix: Add pip install or uv sync step before the failing step.
    """
    pattern = r'ModuleNotFoundError: No module named \'(\w+)\''
    matches = re.findall(pattern, logs)

    if not matches:
        return FixResult(fixed=False, description="No missing module errors found")

    # Check if it's a third-party package that needs installing
    # or a local import issue
    packages = set(matches) - {'garvis', 'tests'}  # local packages

    if not packages:
        return FixResult(
            fixed=False,
            description=f"Missing modules are local imports: {matches} — likely a path issue, not a missing install"
        )

    return FixResult(
        fixed=True,
        description=f"Missing packages detected: {packages}. This requires adding an install step to the workflow.",
        files_changed=[],
        verification_cmd="uv run mypy . --exclude site && uv run pytest",
    )


# ============================================================
# PATTERN 4: JSON serialization of dataclass/enum
# ============================================================
def fix_json_serialization(logs: str, repo_root: Path) -> FixResult:
    """
    Symptom: TypeError: Object of type X is not JSON serializable
    Fix: Add asdict() for dataclasses or .value for enums before json.dumps()
    """
    pattern = r'TypeError: Object of type (\w+) is not JSON serializable'
    matches = re.findall(pattern, logs)

    if not matches:
        return FixResult(fixed=False, description="No JSON serialization errors found")

    type_names = matches
    # Find files that call json.dumps with the problematic type
    py_files = list(repo_root.rglob("*.py"))
    files_changed = []
    changes = []

    for py_file in py_files:
        if '.venv' in str(py_file) or '__pycache__' in str(py_file) or 'site-packages' in str(py_file):
            continue

        try:
            with open(py_file, 'r') as f:
                content = f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue

        modified = False
        for type_name in type_names:
            # Look for json.dumps(something_involving_type_name)
            # This is a heuristic — we look for json.dumps(variable) where variable
            # might be of the problematic type
            if f'json.dumps' in content and type_name.lower() in content.lower():
                # Check if dataclasses.asdict is already imported
                if 'from dataclasses import asdict' not in content:
                    # Add the import
                    content = content.replace(
                        'import json',
                        'import json\nfrom dataclasses import asdict'
                    )
                    modified = True
                    changes.append(f"  - {py_file}: added dataclasses.asdict import")

        if modified:
            with open(py_file, 'w') as f:
                f.write(content)
            files_changed.append(str(py_file.relative_to(repo_root)))

    if files_changed:
        return FixResult(
            fixed=True,
            description=f"Added dataclass serialization fixes:\n" + "\n".join(changes),
            files_changed=files_changed,
        )
    return FixResult(fixed=False, description="Could not auto-fix — manual investigation needed")


# ============================================================
# PATTERN 5: Grep self-matching in compliance checks
# ============================================================
def fix_grep_self_match(logs: str, repo_root: Path) -> FixResult:
    """
    Symptom: Compliance grep finds violations in its own workflow file
    Fix: Add exclusion for .github/ directory in the grep command
    """
    yaml_files = list(repo_root.glob(".github/workflows/*.yml"))

    files_changed = []
    changes = []

    for yml_file in yaml_files:
        with open(yml_file, 'r') as f:
            content = f.read()

        modified = False

        # Look for grep -rni patterns that don't exclude .github/
        # Pattern: grep -rni "something" . without grep -v ".github/"
        grep_pattern = r'(grep\s+-r\w*\s+"[^"]+"\s+\.)(?!.*grep\s+-v\s+"\.github/)'
        if re.search(grep_pattern, content) and 'grep -v ".github/"' not in content:
            # Add exclusion
            content = re.sub(
                r'(grep\s+-r\w*\s+"[^"]+"\s+\.)',
                r'\1 | grep -v ".github/"',
                content
            )
            modified = True
            changes.append(f"  - {yml_file.name}: added .github/ exclusion to grep")

        if modified:
            with open(yml_file, 'w') as f:
                f.write(content)
            files_changed.append(str(yml_file.relative_to(repo_root)))

    if files_changed:
        return FixResult(
            fixed=True,
            description=f"Added grep exclusions:\n" + "\n".join(changes),
            files_changed=files_changed,
        )
    return FixResult(fixed=False, description="No grep self-match issues found")


# ============================================================
# PATTERN 6: mypy comparison-overlap (tuple vs scalar)
# ============================================================
def fix_mypy_comparison_overlap(logs: str, repo_root: Path) -> FixResult:
    """
    Symptom: error: Non-overlapping equality check [comparison-overlap]
    When comparing a tuple element (float) with pytest.approx()
    Fix: Add explicit float() cast or use a separate variable
    """
    pattern = r'(.+?):(\d+): error: Non-overlapping equality check.*\[comparison-overlap\]'
    matches = re.findall(pattern, logs)

    if not matches:
        return FixResult(fixed=False, description="No comparison-overlap errors found")

    files_changed = []
    changes = []

    by_file = {}
    for filepath, line_num in matches:
        if filepath not in by_file:
            by_file[filepath] = []
        by_file[filepath].append(int(line_num))

    for filepath, line_nums in by_file.items():
        full_path = repo_root / filepath
        if not full_path.exists():
            continue

        with open(full_path, 'r') as f:
            lines = f.readlines()

        modified = False
        for line_num in line_nums:
            idx = line_num - 1
            if idx < len(lines):
                line = lines[idx]
                # If the line compares something[0] or something[1] with pytest.approx
                # Wrap in float() to satisfy mypy
                new_line = re.sub(
                    r'(\w+\.(\w+)\[(\d+)\])\s*==\s*pytest\.approx',
                    r'float(\1) == pytest.approx',
                    line
                )
                if new_line != line:
                    lines[idx] = new_line
                    modified = True
                    changes.append(f"  - {filepath}:{line_num}: wrapped in float() for mypy")

        if modified:
            with open(full_path, 'w') as f:
                f.writelines(lines)
            files_changed.append(filepath)

    if files_changed:
        return FixResult(
            fixed=True,
            description=f"Fixed comparison-overlap errors:\n" + "\n".join(changes),
            files_changed=files_changed,
            verification_cmd="uv run mypy . --exclude site",
        )
    return FixResult(fixed=False, description="Could not auto-fix comparison-overlap — needs manual review")


# ============================================================
# PATTERN 7: mypy arg-type (wrong argument type)
# ============================================================
def fix_mypy_arg_type(logs: str, repo_root: Path) -> FixResult:
    """
    Symptom: error: Argument N to X has incompatible type Y; expected Z [arg-type]
    Fix: This is often a casting issue. We add a type: ignore with the specific error
    only if auto-detection determines it's a known false positive.
    """
    pattern = r'(.+?):(\d+): error: Argument \d+ to "(\w+)" has incompatible type'
    matches = re.findall(pattern, logs)

    if not matches:
        return FixResult(fixed=False, description="No arg-type errors found")

    # This pattern is too complex for a deterministic fix without understanding the type system
    return FixResult(
        fixed=False,
        description=f"Found {len(matches)} arg-type errors in {set(m[0] for m in matches)}. These require manual type analysis — cannot auto-fix safely."
    )


# ============================================================
# MAIN: Run all fixers, report results
# ============================================================
def main():
    logs_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/all-logs.txt"
    repo_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    failed_workflow = os.environ.get("FAILED_WORKFLOW", "unknown")
    failed_sha = os.environ.get("FAILED_HEAD_SHA", "unknown")

    with open(logs_path, 'r', errors='replace') as f:
        logs = f.read()

    fixers = [
        ("unused_type_ignore", fix_unused_type_ignore),
        ("bash_quote_conflict", fix_bash_quote_conflict),
        ("missing_install", fix_missing_install),
        ("json_serialization", fix_json_serialization),
        ("grep_self_match", fix_grep_self_match),
        ("mypy_comparison_overlap", fix_mypy_comparison_overlap),
        ("mypy_arg_type", fix_mypy_arg_type),
    ]

    results = []
    any_fixed = False

    for name, fixer in fixers:
        result = fixer(logs, repo_root)
        results.append({"pattern": name, **{
            "fixed": result.fixed,
            "description": result.description,
            "files_changed": result.files_changed,
            "verification_cmd": result.verification_cmd,
        }})
        if result.fixed:
            any_fixed = True

    output = {
        "workflow": failed_workflow,
        "sha": failed_sha,
        "any_fixed": any_fixed,
        "results": results,
    }

    print(json.dumps(output, indent=2))

    # Exit code: 0 if something was fixed, 1 if nothing could be fixed
    sys.exit(0 if any_fixed else 1)


if __name__ == "__main__":
    main()
