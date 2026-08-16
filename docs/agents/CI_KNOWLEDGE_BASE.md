# GARVIS CI Knowledge Base

> This is GARVIS's accumulated knowledge about CI/CD failure patterns.
> The CI Doctor Agent pattern-matches errors against these entries to
> diagnose and auto-fix issues. Updated whenever a new pattern is discovered.

---

## Pattern 1: Bash Quote Conflict in YAML `python3 -c`

**Symptom:**
```
NameError: name 'X' is not defined
```
or
```
SyntaxError: invalid syntax
```

**Root Cause:**
YAML `run: |` blocks that wrap Python code with `python3 -c "..."`
break when the Python code contains inner double quotes. Bash interprets
the inner quotes as closing the outer string.

**Example (BROKEN):**
```yaml
run: |
  python3 -c "
  x = "hello"
  print(x)
  "
```

**Fix:**
Use a heredoc instead of `python3 -c`:
```yaml
run: |
  python3 << 'SCRIPT'
  x = "hello"
  print(x)
  SCRIPT
```

Or use single quotes inside:
```yaml
run: |
  python3 -c "
  x = 'hello'
  print(x)
  "
```

**Prevention:**
Never use `python3 -c "..."` with inner double quotes in YAML.
Always prefer heredocs for multi-line Python in GitHub Actions.

---

## Pattern 2: Missing pip install in Workflow

**Symptom:**
```
/opt/hostedtoolcache/Python/3.11.x/x64/bin/pytest: No module named pytest
```
or
```
ModuleNotFoundError: No module named 'X'
```

**Root Cause:**
A workflow step uses `pytest` or imports a package but the workflow
never ran `pip install` for that package.

**Fix:**
Add an install step before the test/run step:
```yaml
- name: Install dependencies
  run: pip install pytest
```

**Prevention:**
Every workflow that runs Python code must have an explicit
`pip install` step for all required packages.

---

## Pattern 3: ModelScore / Dataclass JSON Serialization

**Symptom:**
```
TypeError: Object of type ModelScore is not JSON serializable
```

**Root Cause:**
`json.dumps()` cannot serialize dataclass instances or enum values
directly. `brain.score()` returns a `ModelScore` dataclass with an
`EmpiricalStatus` enum field.

**Fix:**
```python
from dataclasses import asdict
d = asdict(score)
d['status'] = score.status.value  # Convert enum to string
json.dumps(d)
```

**Prevention:**
Always convert dataclasses with `asdict()` and enums with `.value`
before passing to `json.dumps()`.

---

## Pattern 4: Grep Self-Matching in Compliance Checks

**Symptom:**
A naming compliance check reports violations in its own workflow file.

**Root Cause:**
`grep -rni "the old name" .` matches the workflow file that contains the
word "JARVIS" in its own step names and error messages.

**Fix:**
Exclude the `.github/` directory from the grep:
```bash
grep -rni "pattern" . | grep -v ".git/" | grep -v ".github/" | wc -l
```

**Prevention:**
Compliance greps should always exclude their own workflow directory.
For strings that must be mentioned in workflow files (like checking for
an old name), construct them dynamically: `chr(74)+chr(65)+chr(82)+chr(86)+chr(73)+chr(83)`.

---

## Pattern 5: GitHub Release Without Tag

**Symptom:**
```
##[error]Error: Tag not found
```
in the `softprops/action-gh-release` step.

**Root Cause:**
`softprops/action-gh-release` requires a git tag to create a release.
When triggered via `workflow_dispatch` without a tag push, no tag exists.

**Fix:**
Auto-generate a tag in a prior step:
```yaml
- name: Determine tag
  id: tag
  run: |
    TAG="v0.$(date +%Y%m%d).$(git rev-parse --short HEAD)"
    echo "tag=$TAG" >> $GITHUB_OUTPUT
- uses: softprops/action-gh-release@v2
  with:
    tag_name: ${{ steps.tag.outputs.tag }}
```

**Prevention:**
Release workflows triggered by `workflow_dispatch` must always
generate or accept a tag name input.

---

## Pattern 6: Python Version Mismatch

**Symptom:**
```
SyntaxError: invalid syntax
```
on a line using Python 3.10+ features (match/case, type unions with `|`).

**Root Cause:**
Workflow uses `python-version: '3.9'` but code uses 3.10+ syntax.

**Fix:**
Bump the Python version in `actions/setup-python`:
```yaml
python-version: '3.11'
```

**Prevention:**
Match the CI Python version to the minimum supported version in
`setup.py` / `pyproject.toml`.

---

## Addendum: How to Add New Patterns

When the CI Doctor Agent encounters an unknown failure:
1. It creates a GitHub issue with the error, logs, and a diagnosis request
2. Once the root cause is identified, add a new pattern entry here
3. The CI Doctor will match it automatically on future occurrences

This knowledge base lives at `docs/agents/CI_KNOWLEDGE_BASE.md` and is
the GARVIS system's memory for CI/CD failure patterns.
