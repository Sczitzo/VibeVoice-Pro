import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf

from demo.gradio_demo import VibeVoiceDemo


class VoiceRecorderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="vibevoice-recorder-test-")
        self.addCleanup(directory.cleanup)
        self.demo = VibeVoiceDemo.__new__(VibeVoiceDemo)
        self.demo.voices_dir = Path(directory.name)
        self.demo.voice_lock = threading.Lock()
        self.samples = np.array([0, 1200, -2400, 600], dtype=np.int16)
        self.existing = self.demo.voices_dir / "Existing.wav"
        sf.write(self.existing, self.samples, 24000, subtype="PCM_16")
        self.demo.available_voices = {"Existing": str(self.existing)}

    def test_remove_and_restore_preserve_the_recording_and_built_in_voice(self):
        name, path = self.demo.save_recorded_voice((24000, self.samples), "Studio sample")
        original = path.read_bytes()
        with self.assertRaisesRegex(gr.Error, "no longer available"):
            self.demo.change_saved_voice("Existing")
        self.assertTrue(self.existing.is_file())

        recovery = self.demo.change_saved_voice(name, revision=self.demo.saved_voice_catalog()[name])
        self.assertEqual(recovery.name, path.name)
        self.assertFalse(path.exists())
        self.assertEqual(recovery.read_bytes(), original)
        self.demo.setup_voice_presets()
        self.assertNotIn(name, self.demo.available_voices)

        restored = self.demo.change_saved_voice(name, restore_from=recovery)
        self.assertEqual(restored, path)
        self.assertEqual(restored.read_bytes(), original)
        self.assertIn(name, self.demo.saved_voice_catalog())

    def test_undo_never_overwrites_a_new_recording(self):
        name, _ = self.demo.save_recorded_voice((24000, self.samples), "Studio sample")
        recovery = self.demo.change_saved_voice(name, revision=self.demo.saved_voice_catalog()[name])
        _, replacement = self.demo.save_recorded_voice((24000, -self.samples), name)
        replacement_bytes = replacement.read_bytes()
        with self.assertRaisesRegex(gr.Error, "already exists"):
            self.demo.change_saved_voice(name, restore_from=recovery)
        self.assertEqual(replacement.read_bytes(), replacement_bytes)
        self.assertTrue(recovery.is_file())

    def test_recording_is_saved_as_pcm16_wav_and_loaded_as_a_voice(self):
        name, path = self.demo.save_recorded_voice((48000, self.samples), "My voice")
        self.assertEqual(name, "My voice")
        self.assertEqual(path, self.demo.voices_dir / "recorded" / "My voice.wav")
        info = sf.info(path)
        self.assertEqual((info.format, info.subtype, info.samplerate), ("WAV", "PCM_16", 48000))
        saved, _ = sf.read(path, dtype="int16")
        np.testing.assert_array_equal(saved, self.samples)
        self.assertEqual(self.demo.available_voices[name], str(path))
        reloaded = VibeVoiceDemo.__new__(VibeVoiceDemo)
        reloaded.voices_dir = self.demo.voices_dir
        reloaded.setup_voice_presets()
        self.assertEqual(reloaded.available_voices[name], str(path))

    def test_voice_library_refresh_is_case_insensitive_alphabetical(self):
        for name in ('zeta', 'beta', 'Alpha'):
            sf.write(self.demo.voices_dir / (name + '.wav'), self.samples, 24000, subtype='PCM_16')
        self.demo.setup_voice_presets()
        self.assertEqual(list(self.demo.available_voices), ['Alpha', 'beta', 'Existing', 'zeta'])
        self.demo.save_recorded_voice((24000, self.samples), 'delta')
        self.assertEqual(list(self.demo.available_voices), ['Alpha', 'beta', 'delta', 'Existing', 'zeta'])

    def test_stereo_recording_keeps_both_channels_and_amplitudes(self):
        stereo = np.column_stack((self.samples, -self.samples))
        _, path = self.demo.save_recorded_voice((24000, stereo), "Stereo voice")
        saved, _ = sf.read(path, dtype="int16")
        np.testing.assert_array_equal(saved, stereo)

    def test_uploaded_wav_keeps_its_samples_and_rate(self):
        _, path = self.demo.save_recorded_voice(str(self.existing), "Uploaded voice")
        saved, rate = sf.read(path, dtype="int16")
        self.assertEqual(rate, 24000)
        np.testing.assert_array_equal(saved, self.samples)

    def test_invalid_uploaded_wav_is_rejected_without_creating_a_voice(self):
        invalid = self.demo.voices_dir / "invalid.wav"
        invalid.write_bytes(b"This is not WAV audio")
        with self.assertRaises(gr.Error):
            self.demo.save_recorded_voice(str(invalid), "Invalid voice")
        disguised = self.demo.voices_dir / "disguised.wav"
        sf.write(disguised, self.samples, 24000, format="FLAC")
        with self.assertRaises(gr.Error):
            self.demo.save_recorded_voice(str(disguised), "Disguised voice")
        self.assertFalse((self.demo.voices_dir / "recorded").exists())

    def test_duplicate_voice_cannot_overwrite_a_preset(self):
        before = self.existing.read_bytes()
        with self.assertRaises(gr.Error):
            self.demo.save_recorded_voice((24000, self.samples), "existing")
        self.assertEqual(self.existing.read_bytes(), before)

    def test_invalid_names_and_audio_are_rejected(self):
        for name in ("", "../voice", "Cam/voice", "CON", "LPT1", "a" * 61):
            with self.subTest(name=name), self.assertRaises(gr.Error):
                self.demo.save_recorded_voice((24000, self.samples), name)
        for audio in (None, (0, self.samples), (24000, np.array([])),
                      (24000, np.array([np.nan])), (24000, np.zeros(4))):
            with self.subTest(audio=audio), self.assertRaises(gr.Error):
                self.demo.save_recorded_voice(audio, "My voice")

    def test_failed_write_does_not_leave_a_broken_voice(self):
        with patch("demo.gradio_demo.sf.write", side_effect=OSError("disk full")):
            with self.assertRaises(gr.Error):
                self.demo.save_recorded_voice((24000, self.samples), "My voice")
        self.assertFalse((self.demo.voices_dir / "recorded" / "My voice.wav").exists())
        self.assertNotIn("My voice", self.demo.available_voices)

    def test_simultaneous_duplicate_saves_keep_one_valid_wav(self):
        def save():
            try:
                self.demo.save_recorded_voice((24000, self.samples), "My voice")
                return True
            except gr.Error:
                return False

        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(lambda _index: save(), range(2)))
        self.assertEqual(sorted(results), [False, True])
        saved, _ = sf.read(self.demo.voices_dir / "recorded" / "My voice.wav", dtype="int16")
        np.testing.assert_array_equal(saved, self.samples)

    def test_import_saves_exact_range_and_leaves_source_unchanged(self):
        samples = np.arange(24000, dtype=np.int16)
        sf.write(self.existing, samples, 24000)
        original = self.existing.read_bytes()
        _, path = self.demo.save_recorded_voice(self.existing, "Trim", 0.25, 0.75)
        saved, rate = sf.read(path, dtype="int16")
        self.assertEqual(rate, 24000)
        np.testing.assert_array_equal(saved, samples[6000:18000])
        self.assertEqual(self.existing.read_bytes(), original)

    def test_import_rejects_limits_bounds_and_nonfinite_samples(self):
        for start, end in ((-1, 1), (0, 1), (0, 0), (float("nan"), 0.0001)):
            with self.subTest(start=start, end=end), self.assertRaises(gr.Error):
                self.demo.save_recorded_voice(self.existing, "Trim", start, end)
        with patch("demo.gradio_demo.MAX_IMPORT_BYTES", 8), self.assertRaises(gr.Error):
            self.demo.save_recorded_voice(self.existing, "Trim", 0, 0.0001)
        bad = self.demo.voices_dir / "Float.wav"
        sf.write(bad, np.array([0.1, np.nan]), 24000, subtype="FLOAT")
        with self.assertRaises(gr.Error):
            self.demo.save_recorded_voice(bad, "Trim", 0, 2 / 24000)
        self.assertFalse((self.demo.voices_dir / "recorded").exists())

    def test_shared_launch_excludes_and_cannot_save_private_voices(self):
        self.demo.save_recorded_voice((24000, self.samples), "Private voice")
        self.demo.share = True
        self.demo.setup_voice_presets()
        self.assertNotIn("Private voice", self.demo.available_voices)
        with self.assertRaises(gr.Error):
            self.demo.save_recorded_voice((24000, self.samples), "Shared voice")





if __name__ == "__main__":
    unittest.main()
