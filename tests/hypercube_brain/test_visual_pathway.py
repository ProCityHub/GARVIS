import unittest

from hypercube_brain.visual_pathway import (
    VisualFeature,
    VisualPathway,
)


class VisualPathwayTests(unittest.TestCase):

    def setUp(self):
        self.pathway = VisualPathway()

    def test_left_visual_field_routes_right(self):
        percepts, _ = self.pathway.process([
            VisualFeature(
                "f1",
                "chair",
                0.20,
                0.60,
                0.95,
                "left",
            )
        ])

        self.assertEqual(percepts[0].hemisphere, "right")

    def test_right_visual_field_routes_left(self):
        percepts, _ = self.pathway.process([
            VisualFeature(
                "f1",
                "door",
                0.80,
                0.40,
                0.95,
                "right",
            )
        ])

        self.assertEqual(percepts[0].hemisphere, "left")

    def test_retinotopy_preserved(self):
        percepts, _ = self.pathway.process([
            VisualFeature(
                "f1",
                "table",
                0.23,
                0.71,
                0.90,
                "left",
                depth=1.8,
            )
        ])

        p = percepts[0]

        self.assertEqual(p.x, 0.23)
        self.assertEqual(p.y, 0.71)
        self.assertEqual(p.depth, 1.8)

    def test_parallel_channels_preserved(self):
        percepts, _ = self.pathway.process([
            VisualFeature(
                "f1",
                "vehicle",
                0.70,
                0.50,
                0.94,
                "right",
                motion=0.91,
                detail=0.62,
                color_signal=0.73,
            )
        ])

        p = percepts[0]

        self.assertEqual(p.motion_strength, 0.91)
        self.assertEqual(p.detail_strength, 0.62)
        self.assertEqual(p.color_strength, 0.73)

    def test_motion_drives_fast_attention_signal(self):
        _, collateral = self.pathway.process([
            VisualFeature(
                "f1",
                "moving_object",
                0.50,
                0.50,
                0.95,
                "central",
                motion=0.90,
            )
        ])

        self.assertTrue(collateral[0].orienting_attention)

    def test_bright_light_signal(self):
        _, collateral = self.pathway.process([
            VisualFeature(
                "f1",
                "bright_window",
                0.50,
                0.50,
                0.95,
                "central",
                intensity=0.95,
            )
        ])

        self.assertTrue(collateral[0].bright_light_response)

    def test_brain_observation_contains_spatial_state(self):
        percepts, _ = self.pathway.process([
            VisualFeature(
                "f1",
                "door",
                0.25,
                0.65,
                0.94,
                "left",
                motion=0.05,
                depth=2.2,
            )
        ])

        obs = self.pathway.to_brain_observation(percepts[0])

        self.assertIn("field=left", obs.content)
        self.assertIn("depth=2.20", obs.content)
        self.assertEqual(obs.independent_group, "vision")


if __name__ == "__main__":
    unittest.main()
