import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from demo.saved_takes import SavedTakes


class SavedTakesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "takes"
        self.source = Path(temp.name) / "source.wav"
        sf.write(self.source, np.array([1, -2, 3], dtype=np.int16), 24000)
        self.takes = SavedTakes(self.root)

    def test_same_name_saves_survive_restart_with_original_wav(self):
        first = self.takes.save(self.source, "My take", ("Alice", "Carter"))
        second = self.takes.save(self.source, "My take", ("Alice",))
        reopened = SavedTakes(self.root)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual([take.id for take in reopened.list()], [second.id, first.id])
        self.assertEqual(reopened.lookup(first.id).wav.read_bytes(), self.source.read_bytes())
        self.assertEqual(reopened.lookup(first.id).voices, ("Alice", "Carter"))

    def test_bad_entries_and_failed_copy_leave_other_takes_playable(self):
        good = self.takes.save(self.source, "", ())
        bad = self.takes.save(self.source, "Bad", ())
        bad.wav.unlink()
        (self.root / ".incomplete").mkdir()
        with patch("demo.saved_takes.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.takes.save(self.source, "", ())
        self.assertEqual([take.id for take in self.takes.list()], [good.id])
        self.assertIsNone(self.takes.lookup("../source"))
        self.assertEqual(len(list(self.root.iterdir())), 3)
