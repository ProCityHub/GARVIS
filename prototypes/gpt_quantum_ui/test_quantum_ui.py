from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import quantum_ui


class QuantumUITests(unittest.TestCase):
    def test_canonical_exponent_identity(self) -> None:
        self.assertTrue(
            math.isclose(
                quantum_ui.INV_PHI
                + quantum_ui.INV_PHI2,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_lattice_law_rejects_invalid_domain(self) -> None:
        with self.assertRaises(ValueError):
            quantum_ui.lattice_weight(
                1.0,
                0.0,
                1.0,
            )

    def test_epistemic_boundary_is_explicit(self) -> None:
        frame = quantum_ui.sample_frame()

        self.assertIn(
            "HYPOTHESIS UNDER TEST",
            frame.lattice_status,
        )
        self.assertIn(
            "NOT_SUPPORTED",
            frame.lattice_status,
        )
        self.assertFalse(frame.execution_enabled)
        self.assertTrue(frame.approval_required)

    def test_renderer_contains_oab_and_execution_gate(self) -> None:
        html = quantum_ui.render_html(
            quantum_ui.sample_frame()
        )

        self.assertIn("Observer", html)
        self.assertIn("OAB Bridge", html)
        self.assertIn("Actor", html)
        self.assertIn("EXECUTION_DISABLED", html)
        self.assertIn(
            "physical quantum coherence",
            html,
        )

    def test_script_payload_escapes_closing_markup(self) -> None:
        frame = quantum_ui.QuantumFrame(
            observer="</script><script>alert(1)</script>",
            bridge="verification",
            proposed_action="none",
            coherence=0.5,
            heartbeat=1,
        )

        html = quantum_ui.render_html(frame)

        self.assertNotIn(
            "</script><script>alert(1)</script>",
            html,
        )
        self.assertIn("\\u003c/script\\u003e", html)

    def test_prototype_uses_standard_library_only(self) -> None:
        source_path = Path(quantum_ui.__file__)
        tree = ast.parse(source_path.read_text())

        allowed_roots = {
            "__future__",
            "argparse",
            "json",
            "math",
            "dataclasses",
            "http",
            "typing",
        }

        imported_roots: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(
                        alias.name.split(".", 1)[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_roots.add(
                        node.module.split(".", 1)[0]
                    )

        self.assertEqual(
            imported_roots - allowed_roots,
            set(),
        )


if __name__ == "__main__":
    unittest.main()
