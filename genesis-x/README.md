# Genesis X Double-Slit Brain (`genesis-x-brain`)

A falsification-oriented Python package for comparing standard two-path quantum-mechanical visibility ($V_{\text{QM}}$) against the Genesis X / Canonical Lattice interference-discriminator hypothesis ($V_{\phi}$) using double-slit measurement data.

## Overview

The package provides a test harness and scoring brain to evaluate two competing visibility models:

1. **Standard Quantum Mechanics (QM)**:
   $$V_{\text{QM}} = \frac{2 O \sqrt{A B}}{A + B}$$

2. **Genesis X Discriminator Hypothesis**:
   $$V_{\phi} = \frac{O (A^{\alpha} B^{\beta} + A^{\beta} B^{\alpha})}{A + B}$$
   where $\phi = \frac{1 + \sqrt{5}}{2}$, $\alpha = \frac{1}{\phi}$, and $\beta = \frac{1}{\phi^2}$.

Symmetric path intensities ($A = B$) produce identical predictions between models ($V_{\text{QM}} = V_{\phi}$). Only asymmetric path intensities ($A \neq B$) allow model discrimination.

## Installation

Install locally in editable mode or as a package:

```bash
pip install -e .
```

Requirements: Python `>= 3.9`.

## Usage

### Running Self-Test

```python
from brain import self_test

results = self_test()
print(results)
```

### Evaluating CSV Data

```python
from brain import DoubleSlitBrain, load_csv

# Load dataset (expected columns: A, B, O, delta_theta, measured_intensity)
observations = load_csv("examples/sample_data.csv")

brain = DoubleSlitBrain()
for obs in observations:
    brain.observe(obs)

score = brain.score(asymmetric_only=True)
print("Empirical Status:", score.status)
print("QM MAE:", score.qm_mae)
print("Phi MAE:", score.phi_mae)
```

Command-line usage:

```bash
python brain.py examples/sample_data.csv
```

## Running Tests

Run unit tests using `pytest`:

```bash
pytest
```

or via Python test runner:

```bash
python -m unittest discover tests/
```

## Epistemic Boundary

* **Hypotheses Under Test**: Both the Canonical Lattice Law ($C = O^1 A^{1/\phi} B^{1/\phi^2}$) and the $V_\phi$ interference discriminator are strictly hypotheses under test.
* **Empirical Status**: Software tests on synthetic data are software validation only and do NOT constitute physical experimental evidence. Empirical evaluation requires authentic physical asymmetric-path laboratory data.
* **Scope Limits**: Mathematical consistency and software PASS do not establish physical truth, new physics, consciousness, or AGI.

## Credits & Metadata

* **Creator / Architect**: Adrien D. Thomas
* **Organization**: ProCityHub
* **System**: GARVIS / Genesis X
* **Model**: Double-Slit OAB Falsification Brain
