import unittest

from hypercube_brain.screen_retina import (
    AndroidScreenRetina,
    ScreenBounds,
    ScreenElement,
)

from hypercube_brain.sensory_nervous_system import (
    ThalamicRouter,
    activation_policy,
)


class ScreenRetinaTests(
    unittest.TestCase
):

    def setUp(self):
        self.retina = (
            AndroidScreenRetina()
        )

    def element(
        self,
        *,
        element_id: str = "send-1",
        label: str = "Send",
        bounds: ScreenBounds | None = None,
    ) -> ScreenElement:
        if bounds is None:
            bounds = ScreenBounds(
                810,
                2100,
                1050,
                2310,
            )

        return ScreenElement(
            element_id=element_id,
            label=label,
            role="button",
            bounds=bounds,
            screen_width=1080,
            screen_height=2400,
            confidence=0.98,
            motion=0.0,
            source="android_screen",
        )

    def test_maps_screen_element_to_visual_nerve(
        self,
    ):
        packet = self.retina.packet(
            self.element()
        )

        self.assertEqual(
            packet.label,
            "Send",
        )

        self.assertEqual(
            packet.visual_field,
            "right",
        )

        self.assertEqual(
            packet.source,
            "android_screen",
        )

        self.assertTrue(
            0.0
            <= packet.x
            <= 1.0
        )

        self.assertTrue(
            0.0
            <= packet.y
            <= 1.0
        )

    def test_one_input_produces_one_evidence_lineage(
        self,
    ):
        items = (
            self.element(
                element_id="left",
                label="Back",
                bounds=ScreenBounds(
                    0,
                    0,
                    120,
                    120,
                ),
            ),
            self.element(
                element_id="center",
                label="GARVIS",
                bounds=ScreenBounds(
                    420,
                    900,
                    660,
                    1200,
                ),
            ),
            self.element(),
        )

        packets = (
            self.retina.packets(
                items
            )
        )

        self.assertEqual(
            len(packets),
            len(items),
        )

        self.assertEqual(
            [
                packet.visual_field
                for packet
                in packets
            ],
            [
                "left",
                "central",
                "right",
            ],
        )

    def test_routes_as_vision_observation(
        self,
    ):
        cortex, observation = (
            ThalamicRouter.route(
                self.retina.packet(
                    self.element()
                )
            )
        )

        self.assertEqual(
            cortex,
            "visual_cortex",
        )

        self.assertEqual(
            observation.independent_group,
            "vision",
        )

        self.assertIn(
            "Send",
            observation.content,
        )

        self.assertTrue(
            observation.source.startswith(
                "visual_nerve:"
            )
        )

    def test_screen_activation_requires_approval(
        self,
    ):
        required = (
            "APPROVAL_REQUIRED_"
            "FOR_LIVE_SENSOR_ACTIVATION"
        )

        self.assertEqual(
            activation_policy(
                "screen"
            ),
            required,
        )

        self.assertEqual(
            activation_policy(
                "camera"
            ),
            required,
        )

        self.assertEqual(
            activation_policy(
                "microphone"
            ),
            required,
        )

        self.assertEqual(
            activation_policy(
                "unknown-sensor"
            ),
            "UNKNOWN_SENSOR_FAIL_CLOSED",
        )

    def test_retina_has_no_execution_methods(
        self,
    ):
        forbidden = {
            "click",
            "tap",
            "type",
            "send",
            "delete",
            "install",
            "deploy",
            "dispatch",
        }

        exposed = {
            name.lower()
            for name
            in dir(self.retina)
        }

        self.assertTrue(
            forbidden.isdisjoint(
                exposed
            )
        )

    def test_bounded_autocalibration_converges_without_persistence(
        self,
    ):
        items = (
            self.element(
                element_id="left",
                label="Back",
                bounds=ScreenBounds(
                    0,
                    0,
                    120,
                    120,
                ),
            ),
            self.element(
                element_id="center",
                label="GARVIS",
                bounds=ScreenBounds(
                    420,
                    900,
                    660,
                    1200,
                ),
            ),
            self.element(),
        )

        report = (
            self.retina.autocalibrate(
                items,
                max_rounds=9,
                stable_rounds=2,
            )
        )

        self.assertTrue(
            report.converged
        )

        self.assertTrue(
            report.provisional
        )

        self.assertEqual(
            report.packet_count,
            3,
        )

        self.assertEqual(
            report.unique_lineages,
            3,
        )

        self.assertLessEqual(
            report.max_roundtrip_error,
            1e-12,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
