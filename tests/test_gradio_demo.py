import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from demo.gradio_demo import VibeVoiceDemo, create_demo_interface, merge_speaker_turns


class ScriptTurnTests(unittest.TestCase):
    def test_consecutive_paragraphs_keep_all_words_and_punctuation_in_one_turn(self):
        lines = [
            "Speaker 0: The opening paragraph.",
            "Speaker 0: A second paragraph, with punctuation: intact!",
            "Speaker 0: The final paragraph."
        ]
        self.assertEqual(merge_speaker_turns(lines), [
            "Speaker 0: The opening paragraph. A second paragraph, with punctuation: intact! The final paragraph."
        ])
        self.assertEqual(len(lines), 3)

    def test_speaker_changes_preserve_the_conversation_order(self):
        self.assertEqual(merge_speaker_turns([
            "Speaker 0: Hello.", "Speaker 1: First reply.",
            "Speaker 1: More from the same person.", "Speaker 0: Goodbye."
        ]), [
            "Speaker 0: Hello.", "Speaker 1: First reply. More from the same person.",
            "Speaker 0: Goodbye."
        ])

    def test_one_based_labels_and_empty_inputs(self):
        self.assertEqual(merge_speaker_turns(["Speaker 1: One.", "Speaker 1: Two."]),
                         ["Speaker 1: One. Two."])
        self.assertEqual(merge_speaker_turns([]), [])


class GenerationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.demo = VibeVoiceDemo.__new__(VibeVoiceDemo)
        self.demo.inference_steps = 10
        self.demo.loaded_adapter_root = None
        self.demo.is_generating = False
        self.demo.stop_generation = False
        self.demo.current_streamer = None
        self.demo.available_voices = {
            "en-Alice_woman": str(Path("demo/voices/en-Alice_woman.wav")),
            "en-Carter_man": str(Path("demo/voices/en-Carter_man.wav")),
            "en-Frank_man": str(Path("demo/voices/en-Frank_man.wav")),
            "en-Maya_woman": str(Path("demo/voices/en-Maya_woman.wav")),
        }

    def wrapper(self):
        interface = create_demo_interface(self.demo)
        self.addCleanup(interface.close)
        return next(
            block.fn for block in interface.fns.values()
            if getattr(block.fn, "__name__", None) == "generate_podcast_wrapper"
        )

    def test_invalid_script_restores_generate_button(self):
        results = list(self.wrapper()(
            1, "", "en-Alice_woman", None, None, None, 1.3, 10, 42, False
        ))
        self.assertIn("Please provide a script", results[-1][2])
        self.assertTrue(results[-1][4]["visible"])
        self.assertFalse(results[-1][5]["visible"])

    def test_empty_recording_does_not_put_speaker_controls_in_error_state(self):
        interface = create_demo_interface(self.demo)
        self.addCleanup(interface.close)
        save_recording = next(
            block.fn for block in interface.fns.values()
            if block.fn.__name__ == "save_recording"
        )
        result = save_recording(None, "My voice")
        self.assertIn("try Save Voice again", result[4])
        self.assertFalse(result[-1])
        for update in result[:4]:
            self.assertNotIn("value", update)
            self.assertNotIn("visible", update)
        self.assertFalse(result[5]["visible"])

    def test_saving_adds_a_library_voice_without_reassigning_speakers(self):
        interface = create_demo_interface(self.demo)
        self.addCleanup(interface.close)

        def save(_audio, name):
            self.demo.available_voices[name] = "recording.wav"
            return name, "recording.wav"

        self.demo.save_recorded_voice = save
        save_recording = next(
            block.fn for block in interface.fns.values()
            if block.fn.__name__ == "save_recording"
        )
        result = save_recording("upload.wav", "Library voice")
        self.assertTrue(all("value" not in update for update in result[:4]))
        self.assertTrue(all("Library voice" in update["choices"] for update in result[:4]))
        self.assertTrue(result[-1])

    def test_stop_during_voice_loading_preserves_result_contract(self):
        def stop_during_read(_path):
            self.demo.stop_generation = True
            return np.ones(24, dtype=np.float32)

        self.demo.read_audio = stop_during_read
        results = list(self.demo.generate_podcast_streaming(
            1, "Speaker 1: Hello.", speaker_1="en-Alice_woman"
        ))
        self.assertEqual(len(results[-1]), 4)
        self.assertIsNone(results[-1][1])
        self.assertIn("stopped", results[-1][2])
        self.assertFalse(self.demo.is_generating)

    def test_restoring_buttons_preserves_completed_audio(self):
        def completed_run(**_kwargs):
            yield None, "completed.wav", "Generation complete", {"visible": False}

        self.demo.generate_podcast_streaming = completed_run
        results = list(self.wrapper()(
            1, "Speaker 1: Hello.", "en-Alice_woman", None, None, None,
            1.3, 10, 42, False
        ))
        complete_audio = None
        for result in results:
            if "value" in result[1]:
                complete_audio = result[1]["value"]
        self.assertEqual(complete_audio, "completed.wav")
        self.assertTrue(results[-1][4]["visible"])
        self.assertFalse(results[-1][5]["visible"])

    def test_generation_worker_error_reaches_the_log(self):
        def processor(**_kwargs):
            return {}

        processor.tokenizer = None
        self.demo.processor = processor
        self.demo.device = "cpu"
        for failure in (RuntimeError("test generation worker failure"), RuntimeError()):
            with self.subTest(message=str(failure)):
                def fail_generation(**_kwargs):
                    raise failure

                self.demo.model = SimpleNamespace(
                    set_ddpm_inference_steps=lambda **_kwargs: None,
                    generate=fail_generation,
                )
                results = list(self.demo.generate_podcast_streaming(
                    1, "Speaker 1: Hello.", speaker_1="en-Alice_woman",
                    disable_voice_cloning=True
                ))
                self.assertIn("Generation failed", results[-1][2])
                self.assertIn("RuntimeError", results[-1][2])
                if str(failure):
                    self.assertIn(str(failure), results[-1][2])
                self.assertFalse(self.demo.is_generating)

    def test_generation_combines_single_speaker_paragraphs_and_retains_voice_conditioning(self):
        captured = {}
        reference = np.array([0.1, -0.1], dtype=np.float32)

        def processor(**kwargs):
            captured.update(kwargs)
            return {}

        def generate(**kwargs):
            streamer = kwargs["audio_streamer"]
            streamer.put(torch.tensor([[0.1, -0.1, 0.1]]), torch.tensor([0]))
            streamer.end()

        processor.tokenizer = None
        self.demo.processor = processor
        self.demo.device = "cpu"
        self.demo.read_audio = lambda _path: reference
        self.demo.model = SimpleNamespace(
            set_ddpm_inference_steps=lambda **_kwargs: None, generate=generate
        )
        with patch("demo.gradio_demo.time.sleep"), patch("demo.gradio_demo.sf.write"):
            results = list(self.demo.generate_podcast_streaming(
                1, "First paragraph.\nSecond paragraph: keep this.\nLast paragraph.",
                speaker_1="en-Alice_woman"
            ))
        self.assertEqual(captured["text"], [
            "Speaker 0: First paragraph. Second paragraph: keep this. Last paragraph."
        ])
        self.assertIs(captured["voice_samples"][0][0], reference)
        self.assertIn("Formatted script with 1 turns", results[-1][2])


if __name__ == "__main__":
    unittest.main()
