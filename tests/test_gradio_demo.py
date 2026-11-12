import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gradio.state_holder import SessionState

from demo import gradio_demo
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

    def test_unmarked_continuation_reports_original_line_and_previous_speaker(self):
        script = "Speaker 1: Opening.\r\n\r\n  A continuation.\r\nSpeaker 2: Reply."
        check = gradio_demo.check_speaker_script(script, 2)
        self.assertEqual(check.line, 3)
        self.assertEqual(check.previous_label, 1)
        self.assertFalse(check.formatted)

    def test_numbering_and_sparse_speakers_preserve_selected_voice_slots(self):
        cases = (
            ("Speaker 2: Reply.", 2, "Speaker 0: Reply.", (1,)),
            ("Speaker 0: Opening.\nSpeaker 1: Reply.", 2, "Speaker 0: Opening.\nSpeaker 1: Reply.", (0, 1)),
            ("Solo opening.\nA second paragraph: intact!", 1, "Speaker 0: Solo opening. A second paragraph: intact!", (0,)),
        )
        for script, count, formatted, slots in cases:
            with self.subTest(script=script):
                check = gradio_demo.check_speaker_script(script, count)
                self.assertFalse(check.message)
                self.assertEqual(check.formatted, formatted)
                self.assertEqual(check.voice_slots, slots)

    def test_invalid_labels_and_unmarked_opening_do_not_offer_a_guess(self):
        for line in ("Speaker -1: No.", "Speaker - 1: No.", "Speaker 3: No.", "Speaker X: No.", "Speaker 1:", "Opening."):
            with self.subTest(line=line):
                check = gradio_demo.check_speaker_script(f"\n{line}\nSpeaker 2: Reply.", 2)
                self.assertEqual(check.line, 2)
                self.assertTrue(check.message)
                self.assertIsNone(check.previous_label)

    def test_empty_zero_label_does_not_invalidate_an_earlier_speaker(self):
        check = gradio_demo.check_speaker_script("Speaker 2: Reply.\nSpeaker 0:", 2)
        self.assertEqual(check.line, 2)
        self.assertIn("needs text", check.message)

    def test_solo_prose_and_existing_unicode_speaker_numbers_are_accepted(self):
        solo = gradio_demo.check_speaker_script("Speaker notes: these are ordinary words.", 1)
        self.assertEqual(solo.formatted, "Speaker 0: Speaker notes: these are ordinary words.")
        numbered = gradio_demo.check_speaker_script("Speaker ２: Reply.", 2)
        self.assertEqual(numbered.voice_slots, (1,))


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

    def test_ambiguous_generate_does_not_replace_previous_audio(self):
        with patch.object(self.demo, "generate_podcast_streaming") as generate:
            results = list(self.wrapper()(
                2, "Speaker 1: Opening.\nUnmarked continuation.\nSpeaker 2: Reply.",
                "en-Alice_woman", "en-Carter_man", None, None, 1.3, 10, 42, False,
            ))
        generate.assert_not_called()
        self.assertTrue(results)
        self.assertTrue(all("value" not in result[1] for result in results))
        self.assertIn("Line 2", results[-1][2])

    def test_rejected_generation_finishes_gradio_stream_without_error(self):
        async def reject():
            interface = create_demo_interface(self.demo)
            self.addCleanup(interface.close)
            fn = next(block for block in interface.fns.values()
                      if getattr(block.fn, "__name__", "") == "generate_podcast_wrapper")
            state = SessionState(interface)
            response = await interface.process_api(fn, [
                2, "Speaker 1: Opening.\nUnmarked continuation.",
                "en-Alice_woman", "en-Carter_man", None, None, 1.3, 10, 42, False,
                False,
            ], state=state, session_hash="invalid-script")
            self.assertTrue(response["is_generating"])
            finished = await interface.process_api(fn, [], state=state, session_hash="invalid-script",
                                                   iterator=response["iterator"])
            self.assertFalse(finished["is_generating"])
        asyncio.run(reject())

    def test_speaker_two_only_receives_the_second_voice_reference(self):
        captured = {}
        carter = np.array([0.4, -0.4], dtype=np.float32)

        def processor(**kwargs):
            captured.update(kwargs)
            return {}

        def generate(**kwargs):
            kwargs["audio_streamer"].end()

        processor.tokenizer = None
        self.demo.processor = processor
        self.demo.device = "cpu"
        self.demo.read_audio = lambda path: carter if "Carter" in path else np.ones(2)
        self.demo.model = SimpleNamespace(
            set_ddpm_inference_steps=lambda **_kwargs: None, generate=generate,
        )
        list(self.demo.generate_podcast_streaming(
            2, "Speaker 2: Reply.", "en-Alice_woman", "en-Carter_man",
        ))
        self.assertEqual(captured["text"], ["Speaker 0: Reply."])
        self.assertEqual(len(captured["voice_samples"][0]), 1)
        self.assertIs(captured["voice_samples"][0][0], carter)

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
        self.assertIsNone(result[5]["value"])

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
        self.assertEqual(result[5]["value"], "recording.wav")
        self.assertNotIn("visible", result[5])
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

    def test_section_that_hits_model_limit_does_not_return_partial_wav(self):
        def processor(**_kwargs):
            return {}

        def cap(**kwargs):
            streamer = kwargs["audio_streamer"]
            streamer.put(torch.tensor([[0.1, -0.1]]), torch.tensor([0]))
            streamer.end()
            return SimpleNamespace(reach_max_step_sample=torch.tensor([True]))

        processor.tokenizer = None
        self.demo.processor = processor
        self.demo.device = "cpu"
        self.demo.model = SimpleNamespace(set_ddpm_inference_steps=lambda **_kwargs: None, generate=cap)
        with patch("demo.gradio_demo.time.sleep"):
            result = list(self.demo.generate_podcast_streaming(
                1, "Speaker 1: Hello.", speaker_1="en-Alice_woman", disable_voice_cloning=True,
                require_complete=True,
            ))
        self.assertFalse(any(output[1] for output in result))
        self.assertIn("reached its limit", result[-1][2])



if __name__ == "__main__":
    unittest.main()
