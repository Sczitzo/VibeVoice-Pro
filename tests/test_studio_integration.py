"""Integration contracts for the compact composer and local studio tools."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from demo import gradio_demo as app


class StudioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vibevoice-integration-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.demo = app.VibeVoiceDemo.__new__(app.VibeVoiceDemo)
        self.demo.inference_steps = 10
        self.demo.loaded_adapter_root = None
        self.demo.is_generating = False
        self.demo.stop_generation = False
        self.demo.current_streamer = None
        self.demo.share = False
        self.demo.media_root = self.root
        self.demo.available_voices = {'Alice': 'unused.wav', 'Carter': 'unused2.wav'}
        self.wav = self.root / 'completed.wav'
        sf.write(self.wav, np.arange(240, dtype=np.int16), 24000, subtype='PCM_16')
        self.location = patch.object(app, 'takes_directory', return_value=self.root / 'takes')
        self.location.start()
        self.addCleanup(self.location.stop)

    def interface(self):
        interface = app.create_demo_interface(self.demo)
        self.addCleanup(interface.close)
        return interface

    def callbacks(self, interface):
        return {getattr(fn.fn, '__name__', ''): fn.fn for fn in interface.fns.values()}

    def args(self, script='Speaker 1: Hello.'):
        return [1, script, 'Alice', None, None, None, 1.3, 10, 42, False]

    def test_normal_generation_can_be_saved_and_reopened(self):
        def generate(**kwargs):
            yield None, str(self.wav), 'Complete', {}
        self.demo.generate_podcast_streaming = generate
        calls = self.callbacks(self.interface())
        result = list(calls['generate_podcast_wrapper'](*self.args()))[-1]
        completed = result[9]
        _, status = calls['save_completed_take'](completed, 'Demo take')
        self.assertIn('Saved', status)
        take = app.SavedTakes(self.root / 'takes').list()[0]
        copied = calls['open_saved_take'](take.id)
        self.assertTrue(Path(copied).is_file())
        self.assertEqual(Path(copied).read_bytes(), self.wav.read_bytes())
        self.assertIsNone(calls['open_saved_take']('../completed.wav'))

    def test_invalid_and_stopped_requests_do_not_clear_saved_completed_state(self):
        calls = self.callbacks(self.interface())
        invalid = list(calls['generate_podcast_wrapper'](*self.args('')))
        self.assertTrue(all('value' not in row[1] for row in invalid))
        self.assertTrue(all('value' not in row[9] for row in invalid))
        def stopped(**kwargs):
            kwargs['cancelled'].set()
            yield None, None, 'Stopped', {}
        self.demo.generate_podcast_streaming = stopped
        stopped_rows = list(calls['generate_podcast_wrapper'](*self.args()))
        self.assertTrue(all('value' not in row[1] for row in stopped_rows))
        self.assertTrue(all('value' not in row[9] for row in stopped_rows))

    def test_section_retry_and_export_preserve_order(self):
        attempted = []
        def generate(**kwargs):
            text = kwargs['checked'].formatted
            attempted.append(text)
            if 'Middle' in text and attempted.count(text) == 1:
                yield None, None, 'Failed', {}
                return
            index = 2 if 'Middle' in text else 1
            wav = self.root / f'section-{index}.wav'
            sf.write(wav, np.full(24, index, dtype=np.int16), 24000, subtype='PCM_16')
            yield None, str(wav), 'Complete', {}
        self.demo.generate_podcast_streaming = generate
        calls = self.callbacks(self.interface())
        rows = list(calls['generate_podcast_wrapper'](*self.args('Speaker 1: Open.\n---\nSpeaker 1: Middle.'), True))
        job = rows[-1][10]
        self.assertEqual(job.status, ['Ready', 'Failed'])
        list(calls['section_request'](job.id + ':1', job))
        exported = list(calls['section_request'](job.id + ':export', job))[-1]
        values, _ = sf.read(exported[9][0], dtype='int16')
        np.testing.assert_array_equal(values, np.repeat([1, 2], 24))

    def test_only_generation_owner_can_stop_active_job(self):
        def generate(**kwargs):
            yield np.zeros(24, dtype=np.int16), None, 'Live', {}
        self.demo.generate_podcast_streaming = generate
        calls = self.callbacks(self.interface())
        job = calls['generate_podcast_wrapper'](*self.args(), request=SimpleNamespace(session_hash='owner'))
        next(job)
        calls['stop_generation_handler'](SimpleNamespace(session_hash='other'))
        self.assertFalse(self.demo.stop_generation)
        calls['stop_generation_handler'](SimpleNamespace(session_hash='owner'))
        self.assertTrue(self.demo.stop_generation)
        job.close()

    def test_shared_interface_omits_private_operations(self):
        self.demo.share = True
        names = self.callbacks(self.interface())
        for name in ('save_recording', 'save_completed_take', 'open_saved_take', 'section_request'):
            self.assertNotIn(name, names)

    def test_compact_interface_keeps_studio_controls_outside_main_layout(self):
        interface = self.interface()
        by_id = {getattr(component, 'elem_id', None): component for component in interface.blocks.values()}
        self.assertIn('composer-transport', by_id)
        self.assertIn('studio-library-fields', by_id)
        self.assertIn('generation-sections', by_id)
        fields = by_id['studio-library-fields']
        descendants = []
        def visit(node):
            descendants.append(getattr(node, 'elem_id', None))
            for child in getattr(node, 'children', []): visit(child)
        visit(fields)
        self.assertIn('take-save-controls', descendants)
        self.assertIn('generation-sections', descendants)
        self.assertNotIn('composer-transport', descendants)


if __name__ == '__main__': unittest.main()
