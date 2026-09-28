import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PIL import Image
except ImportError:
    raise unittest.SkipTest("Install requirements-local-eyes.txt to test optional vision")
from hypercube_brain.local_eyes import LocalEyes, main, read_image


class LocalEyesTests(unittest.TestCase):
    def test_real_pathway_and_brain_receive_pixels(self):
        eyes = LocalEyes()
        first = Image.new("RGB", (60, 60), "black")
        baseline = eyes.observe(first, source="test", synthetic=True)
        second = first.copy()
        second.paste("white", (0, 0, 10, 10))
        record = eyes.observe(second, source="test", synthetic=True)
        self.assertEqual(baseline["comparison_status"], "BASELINE")
        self.assertIsNone(baseline["cells"][0]["pixel_change"])
        self.assertEqual(record["cells"][0]["pixel_change"], 1)
        self.assertTrue(all(c["pixel_change"] == 0 for c in record["cells"][1:]))
        self.assertEqual(record["percepts"][0]["hemisphere"], "right")
        self.assertEqual(len(record["observations"]), 36)
        self.assertEqual({o["independent_group"] for o in record["observations"]}, {"vision"})
        self.assertEqual(record["brain_assessment"]["cycle_id"], 2)
        self.assertFalse(record["external_action_allowed"])
        self.assertNotEqual(record["brain_assessment"]["truth_state"], "VERIFIED")

    def test_unchanged_source_reset_and_size_reset(self):
        eyes = LocalEyes()
        image = Image.new("RGB", (12, 12), "red")
        first = eyes.observe(image, source="a")
        same = eyes.observe(image, source="a")
        self.assertEqual(first["frame_sha256"], same["frame_sha256"])
        self.assertTrue(all(c["pixel_change"] == 0 for c in same["cells"]))
        self.assertEqual(eyes.observe(image, source="b")["comparison_status"], "BASELINE")
        self.assertEqual(eyes.observe(image.resize((24, 24)), source="b")["comparison_status"], "BASELINE")

    def test_invalid_image_and_local_file_decoding(self):
        with self.assertRaises(ValueError):
            LocalEyes().observe(Image.new("RGB", (1, 1)), source="a")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (12, 12), "blue").save(path)
            self.assertEqual(read_image(path).getpixel((0, 0)), (0, 0, 255))
            path.write_text("not an image")
            with self.assertRaises(OSError):
                read_image(path)
        with self.assertRaises(ValueError):
            read_image("https://example.com/camera")

    def test_cli_demo_and_camera_gate(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--demo"]), 0)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["evidence_kind"] == "SYNTHETIC_TEST" for r in records))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--camera", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_history_bounded(self):
        eyes = LocalEyes()
        image = Image.new("RGB", (6, 6))
        for _ in range(100):
            eyes.observe(image, source="a")
        for history in (eyes.brain.working, eyes.brain.episodes, eyes.brain.events):
            self.assertLessEqual(len(history), 89)

    def test_camera_read_failure_releases_device(self):
        from unittest.mock import MagicMock
        cv2 = MagicMock()
        cv2.VideoCapture.return_value.read.return_value = (False, None)
        with patch.dict("sys.modules", {"cv2": cv2}):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--camera", "0", "--allow-camera"])
        cv2.VideoCapture.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
