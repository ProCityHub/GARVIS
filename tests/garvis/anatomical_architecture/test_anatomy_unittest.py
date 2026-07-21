import unittest

from garvis.anatomical_architecture import AnatomicalHeartbeat, OrganSystem
from garvis.anatomical_architecture.registry import SYSTEMS, list_systems


class AnatomicalArchitectureTests(unittest.TestCase):
    def test_all_eleven_systems_exist(self) -> None:
        self.assertEqual(len(SYSTEMS), 11)
        self.assertEqual(len(list_systems()), 11)

    def test_heartbeat_has_four_hypercube_phases(self) -> None:
        result = AnatomicalHeartbeat().run("hello")
        self.assertEqual(
            [state.phase for state in result.states],
            ["0.0", "0.6", "1.0", "1.6"],
        )

    def test_reproductive_system_is_controlled_generation(self) -> None:
        definition = SYSTEMS[OrganSystem.REPRODUCTIVE]
        self.assertIn("Controlled generation", definition.software_role)


if __name__ == "__main__":
    unittest.main()
