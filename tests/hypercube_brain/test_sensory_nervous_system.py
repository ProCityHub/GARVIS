import unittest

from hypercube_brain.sensory_nervous_system import (
    AuditoryNervePacket,
    BinarySignal,
    BodyStatePacket,
    DigitalCochlea,
    GarvisEmbodiedBrain,
    LanguageCortexBridge,
    MultimodalAssociationCortex,
    VisualNervePacket,
    activation_policy,
)


class SensoryNervousSystemTests(unittest.TestCase):

    def test_binary_roundtrip(self):
        original = b"GARVIS"
        bits = BinarySignal.bits(original)
        self.assertEqual(BinarySignal.bytes_from_bits(bits), original)

    def test_visual_nerve_preserves_space(self):
        packet = VisualNervePacket(
            label="door",
            x=0.20,
            y=0.65,
            depth=1.8,
            motion=0.0,
            confidence=0.94,
            visual_field="left",
        )

        observation = packet.observation()

        self.assertIn("x=0.200", observation.content)
        self.assertIn("depth=1.80", observation.content)
        self.assertEqual(observation.independent_group, "vision")

    def test_digital_cochlea_produces_auditory_packet(self):
        samples = [
            -0.2, -0.1, 0.1, 0.2,
            0.1, -0.1, -0.2, 0.1,
        ] * 20

        packet = DigitalCochlea.encode(samples)

        self.assertIsInstance(packet, AuditoryNervePacket)
        self.assertGreater(packet.rms_energy, 0.0)
        self.assertGreaterEqual(packet.speech_probability, 0.0)
        self.assertLessEqual(packet.speech_probability, 1.0)

    def test_multimodal_cortex_preserves_modalities(self):
        cortex = MultimodalAssociationCortex()

        observations = cortex.integrate([
            VisualNervePacket(
                "person",
                0.50,
                0.50,
                2.0,
                0.1,
                0.93,
                "central",
            ),
            AuditoryNervePacket(
                0.2,
                0.1,
                "mid",
                0.8,
                0.85,
            ),
            BodyStatePacket(
                0.60,
                0.10,
                True,
                False,
                False,
            ),
        ])

        groups = {item.independent_group for item in observations}

        self.assertEqual(groups, {"vision", "audition", "body"})

    def test_language_bridge_contains_provenance(self):
        observations = MultimodalAssociationCortex().integrate([
            BodyStatePacket(
                0.50,
                0.10,
                True,
                False,
                False,
            )
        ])

        rendered = LanguageCortexBridge.render(observations)

        self.assertIn("source=phone_body", rendered)

    def test_embodied_brain_accepts_multimodal_input(self):
        brain = GarvisEmbodiedBrain()

        result = brain.perceive(
            claim="A speaking person may be ahead.",
            packets=[
                VisualNervePacket(
                    "person",
                    0.50,
                    0.50,
                    2.1,
                    0.05,
                    0.94,
                    "central",
                ),
                AuditoryNervePacket(
                    0.22,
                    0.12,
                    "mid",
                    0.86,
                    0.88,
                ),
            ],
        )

        self.assertGreater(result.support, 0.0)

    def test_live_sensor_activation_requires_approval(self):
        self.assertEqual(
            activation_policy("camera"),
            "APPROVAL_REQUIRED_FOR_LIVE_SENSOR_ACTIVATION",
        )

        self.assertEqual(
            activation_policy("microphone"),
            "APPROVAL_REQUIRED_FOR_LIVE_SENSOR_ACTIVATION",
        )


if __name__ == "__main__":
    unittest.main()
