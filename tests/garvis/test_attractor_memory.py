"""Tests for the deterministic GARVIS attractor-memory prototype."""

import unittest

from garvis.attractor_memory import (
    HopfieldMemory,
    erase_bits,
    flip_bits,
)

PATTERN_A = (1, 1, 1, 1, -1, -1, -1, -1)
PATTERN_B = (1, 1, -1, -1, 1, 1, -1, -1)
PATTERN_C = (1, -1, 1, -1, 1, -1, 1, -1)

ORTHOGONAL_PATTERNS = (
    PATTERN_A,
    PATTERN_B,
    PATTERN_C,
)


class TestInputValidation(unittest.TestCase):
    def test_rejects_empty_pattern_collection(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            HopfieldMemory([])

    def test_rejects_empty_stored_pattern(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            HopfieldMemory([()])

    def test_rejects_inconsistent_dimensions(self):
        with self.assertRaisesRegex(ValueError, "same dimension"):
            HopfieldMemory([(1, -1), (1, -1, 1)])

    def test_rejects_invalid_stored_value(self):
        with self.assertRaisesRegex(ValueError, "state values"):
            HopfieldMemory([(1, 0, -1)])

    def test_rejects_invalid_cue_dimension(self):
        memory = HopfieldMemory([PATTERN_A])

        with self.assertRaisesRegex(ValueError, "cue dimension"):
            memory.recall((1, -1))

    def test_rejects_nonpositive_sweep_limit(self):
        memory = HopfieldMemory([PATTERN_A])

        with self.assertRaisesRegex(ValueError, "positive integer"):
            memory.recall(PATTERN_A, max_sweeps=0)


class TestWeightMatrix(unittest.TestCase):
    def test_weights_are_symmetric_with_zero_diagonal(self):
        memory = HopfieldMemory(ORTHOGONAL_PATTERNS)

        for row in range(memory.dimension):
            self.assertEqual(memory.weights[row][row], 0.0)

            for column in range(memory.dimension):
                self.assertAlmostEqual(
                    memory.weights[row][column],
                    memory.weights[column][row],
                )


class TestRecall(unittest.TestCase):
    def test_exact_recall_of_all_stored_patterns(self):
        memory = HopfieldMemory(ORTHOGONAL_PATTERNS)

        for pattern in ORTHOGONAL_PATTERNS:
            with self.subTest(pattern=pattern):
                result = memory.recall(pattern, target=pattern)

                self.assertTrue(result.converged)
                self.assertTrue(result.exact_match)
                self.assertEqual(result.hamming_distance, 0)
                self.assertEqual(result.final_state, pattern)
                self.assertEqual(result.warnings, ())

    def test_partial_cue_recall(self):
        memory = HopfieldMemory([PATTERN_A])
        cue = erase_bits(PATTERN_A, 3, seed=11)

        result = memory.recall(cue, target=PATTERN_A)

        self.assertIn(0, cue)
        self.assertTrue(result.converged)
        self.assertTrue(result.exact_match)
        self.assertEqual(result.final_state, PATTERN_A)
        self.assertNotIn(0, result.final_state)

    def test_noisy_cue_recall(self):
        memory = HopfieldMemory([PATTERN_A])
        cue = flip_bits(PATTERN_A, 3, seed=7)

        result = memory.recall(cue, target=PATTERN_A)

        self.assertNotEqual(cue, PATTERN_A)
        self.assertTrue(result.converged)
        self.assertTrue(result.exact_match)
        self.assertEqual(result.final_state, PATTERN_A)

    def test_energy_does_not_increase(self):
        memory = HopfieldMemory([PATTERN_A])
        cue = flip_bits(PATTERN_A, 3, seed=4)

        result = memory.recall(cue, target=PATTERN_A)

        self.assertIsNotNone(result.initial_energy)

        energies = (result.initial_energy,) + result.energy_trace

        for earlier, later in zip(energies, energies[1:]):
            self.assertLessEqual(later, earlier + 1e-12)

        self.assertNotIn(
            "energy_increased_during_asynchronous_settling",
            result.warnings,
        )

    def test_false_attractor_is_reported(self):
        memory = HopfieldMemory([PATTERN_A])
        inverted_pattern = tuple(-value for value in PATTERN_A)

        result = memory.recall(
            inverted_pattern,
            target=PATTERN_A,
        )

        self.assertTrue(result.converged)
        self.assertFalse(result.exact_match)
        self.assertGreater(result.hamming_distance, 0)
        self.assertIn(
            "converged_to_non_target_attractor",
            result.warnings,
        )

    def test_nonconvergence_is_bounded_and_reported(self):
        memory = HopfieldMemory([(1, 1)])

        memory.weights = (
            (0.0, -1.0),
            (1.0, 0.0),
        )

        result = memory.recall(
            (-1, -1),
            max_sweeps=4,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.sweeps, 4)
        self.assertIn(
            "maximum_sweeps_reached_without_convergence",
            result.warnings,
        )

    def test_multiple_pattern_interference_remains_visible(self):
        memory = HopfieldMemory(ORTHOGONAL_PATTERNS)

        noisy_cue = (
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
        )

        result = memory.recall(
            noisy_cue,
            target=PATTERN_A,
        )

        self.assertTrue(result.converged)
        self.assertFalse(result.exact_match)
        self.assertGreater(result.hamming_distance, 0)
        self.assertIn(
            "converged_to_non_target_attractor",
            result.warnings,
        )


class TestCorruptionUtilities(unittest.TestCase):
    def test_flip_bits_is_reproducible(self):
        first = flip_bits(PATTERN_A, 2, seed=123)
        second = flip_bits(PATTERN_A, 2, seed=123)

        self.assertEqual(first, second)

        changed = sum(original != corrupted for original, corrupted in zip(PATTERN_A, first))

        self.assertEqual(changed, 2)
        self.assertEqual(PATTERN_A, (1, 1, 1, 1, -1, -1, -1, -1))

    def test_erase_bits_is_reproducible(self):
        first = erase_bits(PATTERN_A, 3, seed=456)
        second = erase_bits(PATTERN_A, 3, seed=456)

        self.assertEqual(first, second)
        self.assertEqual(first.count(0), 3)
        self.assertEqual(PATTERN_A, (1, 1, 1, 1, -1, -1, -1, -1))


class TestBaselines(unittest.TestCase):
    def test_nearest_pattern_baseline(self):
        memory = HopfieldMemory(ORTHOGONAL_PATTERNS)

        cue = (
            1,
            1,
            0,
            0,
            -1,
            -1,
            -1,
            -1,
        )

        self.assertEqual(
            memory.nearest_pattern(cue),
            PATTERN_A,
        )

    def test_no_memory_baseline(self):
        memory = HopfieldMemory([PATTERN_A])

        cue = (
            1,
            0,
            -1,
            0,
            1,
            -1,
            0,
            -1,
        )

        result = memory.no_memory_baseline(cue)

        self.assertEqual(
            result,
            (1, 1, -1, 1, 1, -1, 1, -1),
        )


if __name__ == "__main__":
    unittest.main()
