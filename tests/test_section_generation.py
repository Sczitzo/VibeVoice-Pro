import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from demo.gradio_demo import check_speaker_script
from demo.section_audio import SectionJob, export_sections


class SectionGenerationTests(unittest.TestCase):
    def test_checked_sections_keep_original_slots_and_line_numbers(self):
        check = check_speaker_script("Speaker 0: Open.\n---\nSpeaker 1: Reply.\n---\nSpeaker 0: Close.", 2, True)
        self.assertEqual([part.voice_slots for part in check.sections], [(0,), (1,), (0,)])
        self.assertEqual(check.sections[1].formatted, "Speaker 0: Reply.")
        bad = check_speaker_script("Speaker 1: Open.\n---\nMissing label.", 2, True)
        self.assertEqual(bad.line, 3)
        self.assertIsNone(bad.previous_label)

    def test_retry_preserves_other_sections_and_export_is_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / f"{i}.wav" for i in range(3)]
            for i, path in enumerate(paths):
                sf.write(path, np.full(4, i + 1, dtype=np.int16), 24000)
            job = SectionJob(("Opening", "Reply", "Closing"), {"voice": "original"})
            attempted = []

            def generate(section, inputs):
                i = job.sections.index(section)
                self.assertEqual(inputs, {"voice": "original"})
                attempted.append(i)
                return None if i == 1 and attempted.count(1) == 1 else paths[i]

            list(job.run(range(3), generate, threading.Event()))
            self.assertEqual(job.status, ["Ready", "Failed", "Ready"])
            with self.assertRaises(ValueError):
                export_sections(job, root)
            list(job.run([1], generate, threading.Event()))
            self.assertEqual(attempted, [0, 1, 2, 1])
            audio, _ = sf.read(export_sections(job, root), dtype="int16")
            np.testing.assert_array_equal(audio, np.repeat([1, 2, 3], 4))
            paths[1].unlink()
            with self.assertRaisesRegex(ValueError, "Section 2"):
                export_sections(job, root)
            self.assertEqual(job.status, ["Ready", "Failed", "Ready"])
            sf.write(paths[1], np.full(4, 2, dtype=np.int16), 24000)
            list(job.run([1], generate, threading.Event()))
            repaired, _ = sf.read(export_sections(job, root), dtype="int16")
            np.testing.assert_array_equal(repaired, audio)

    def test_stop_retains_finished_sections_and_leaves_pending_retriable(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        ready = Path(temp.name) / "ready.wav"
        sf.write(ready, np.ones(4, dtype=np.int16), 24000)
        job = SectionJob((1, 2, 3), {})
        stop = threading.Event()

        def generate(section, inputs):
            if section == 2:
                stop.set()
            return ready

        list(job.run(range(3), generate, stop))
        self.assertEqual(job.status, ["Ready", "Stopped", "Stopped"])
        self.assertEqual(job.paths, [ready, None, None])
