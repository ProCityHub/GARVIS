"""CORE identity alignment with Hypercube Heartbeat."""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "core_identity.py"
    spec = importlib.util.spec_from_file_location("core_identity", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_brain():
    path = Path(__file__).resolve().parents[1] / "brain.py"
    spec = importlib.util.spec_from_file_location("brain", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestCoreIdentity(unittest.TestCase):
    def test_all_pass(self):
        ci = _load()
        self.assertTrue(ci.all_identity_pass(), ci.identity_checks())

    def test_core_shape(self):
        ci = _load()
        self.assertEqual(len(ci.CORE), 11)
        self.assertEqual(ci.CORE[6], 666)

    def test_forbidden_retracted_differs(self):
        ci = _load()
        O = A = B = 2.0
        self.assertNotAlmostEqual(ci.lattice_C(O, A, B), (O * A * B) * ci.PHI)

    def test_phi_matches_brain(self):
        ci = _load()
        brain = _load_brain()
        self.assertTrue(math.isclose(brain.PHI, ci.PHI))
        self.assertTrue(math.isclose(brain.ALPHA, ci.INV_PHI))
        self.assertTrue(math.isclose(brain.BETA, ci.INV_PHI2))


if __name__ == "__main__":
    unittest.main()
