"""Integration contracts for the compact composer and local studio tools."""
import json
import ast
import asyncio
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from gradio.state_holder import SessionState
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
        rendered, status = calls['save_completed_take'](completed, 'Demo take')
        self.assertIn('Saved', status)
        self.assertIn('Demo take', rendered)
        self.assertIn('Demo take', calls['render_takes']())
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
        progress = [json.loads(row[-1])['progress'] for row in rows]
        self.assertEqual([entry['ready'] for entry in progress if entry], [0, 1, 1, 1])
        self.assertTrue(all(entry['total'] == 2 for entry in progress if entry))
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

    def test_status_only_runs_never_publish_empty_live_playlists(self):
        def generate(**kwargs):
            yield None, str(self.wav), 'Complete', {}
        self.demo.generate_podcast_streaming = generate
        async def run():
            interface = self.interface()
            fn = next(fn for fn in interface.fns.values()
                      if getattr(fn.fn, '__name__', '') == 'generate_podcast_wrapper')
            for name, script, sectioned in [('invalid', '', False),
                    ('sections', 'Speaker 1: Hello.\n---\nSpeaker 1: Again.', True), ('file-only', 'Speaker 1: Hello.', False)]:
                state = SessionState(interface)
                response = await interface.process_api(fn, self.args(script) + [sectioned],
                    state=state, session_hash=name)
                while True:
                    self.assertNotIn('is_stream', response['data'][0])
                    if not response['is_generating']: break
                    response = await interface.process_api(fn, [], state=state, session_hash=name,
                        iterator=response['iterator'])
                self.assertTrue(all(not streams for streams in interface.pending_streams[name].values()))
        asyncio.run(run())

    def test_real_live_segments_still_finalize(self):
        def generate(**kwargs):
            yield (24000, np.zeros(24000, dtype=np.float32)), None, 'Live', {}
            yield None, str(self.wav), 'Complete', {}
        self.demo.generate_podcast_streaming = generate
        async def run():
            interface = self.interface()
            fn = next(fn for fn in interface.fns.values()
                      if getattr(fn.fn, '__name__', '') == 'generate_podcast_wrapper')
            state = SessionState(interface)
            response = await interface.process_api(fn, self.args() + [False], state=state, session_hash='live')
            while response['is_generating']:
                response = await interface.process_api(fn, [], state=state, session_hash='live',
                    iterator=response['iterator'])
            streams = [stream for run in interface.pending_streams['live'].values() for stream in run.values()]
            self.assertEqual(len(streams), 1)
            self.assertTrue(streams[0].ended)
            self.assertGreater(streams[0].max_duration, 0)
        asyncio.run(run())

    def test_stop_works_with_windows_cp1252_console(self):
        output = io.BytesIO()
        console = io.TextIOWrapper(output, encoding='cp1252', errors='strict', write_through=True)
        ended = []
        self.demo.current_streamer = SimpleNamespace(end=lambda: ended.append(True))
        with patch('sys.stdout', console):
            self.demo.stop_audio_generation()
        self.assertTrue(self.demo.stop_generation)
        self.assertEqual(ended, [True])
        self.assertIn(b'Audio generation stop requested', output.getvalue())
        console.detach()

    def test_console_message_literals_are_ascii(self):
        tree = ast.parse(Path(app.__file__).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        self.assertTrue(child.value.isascii(), repr(child.value))

    def test_voice_choices_are_sorted_initially_and_after_save_without_reassignment(self):
        self.demo.available_voices = {'zeta': 'z.wav', 'Carter': 'c.wav', 'alice': 'a.wav'}
        interface = self.interface()
        dropdowns = [component for component in interface.blocks.values()
                     if isinstance(component, app.gr.Dropdown)]
        voices = [component for component in dropdowns if ('zeta', 'zeta') in component.choices]
        self.assertEqual(len(voices), 4)
        for component in voices:
            self.assertEqual([choice[0] for choice in component.choices], ['alice', 'Carter', 'zeta'])
        def save(_audio, name, **kwargs):
            self.demo.available_voices[name] = str(self.wav)
            return name, str(self.wav)
        self.demo.save_recorded_voice = save
        updates = self.callbacks(interface)['save_recording']('fake.wav', 'beta')
        for update in updates[:4]:
            self.assertEqual(update['choices'], ['alice', 'beta', 'Carter', 'zeta'])
            self.assertNotIn('value', update)
        refresh = next(fn for fn in interface.fns.values()
                       if getattr(fn.fn, '__name__', '') == 'refresh_voice_choices')
        self.assertEqual({event for _, event in refresh.targets}, {'load', 'click'})
        self.assertEqual(refresh.fn(['alice', 'Carter', 'zeta'])[:5], updates[:5])
        self.assertEqual(refresh.fn(updates[4])[:5], [app.gr.skip()] * 5)
        cleared = refresh.fn(['alice', 'Carter', 'zeta', 'Removed'], 'Removed', 'Carter')
        self.assertIsNone(cleared[0]['value'])
        self.assertNotIn('value', cleared[1])

    def test_stale_voice_removal_preserves_same_name_replacement(self):
        self.demo.voices_dir = self.root
        self.demo.voice_lock = threading.Lock()
        sf.write(self.root / 'Alice.wav', np.ones(240), 24000)
        self.demo.setup_voice_presets()
        name, path = self.demo.save_recorded_voice((24000, np.ones(240)), 'Sample')
        original = path.read_bytes()
        calls = self.callbacks(self.interface())
        # The first tab's trash action was opened for the original recording.
        revision = self.demo.saved_voice_catalog()[name]
        command = f'delete:{revision}:{name}'
        removed = calls['apply_voice_action'](command, None, name, 'Alice', None, None)
        self.assertIsNone(removed[0]['value'])
        self.assertEqual(removed[-2][1].read_bytes(), original)

        _, path = self.demo.save_recorded_voice((24000, -np.ones(240)), name)
        replacement = path.read_bytes()
        self.assertNotEqual(replacement, original)
        stale = calls['apply_voice_action'](command, None, name, 'Alice', None, None)
        self.assertIn('Review the updated list', stale[-1])
        self.assertEqual(path.read_bytes(), replacement)
        self.assertIsNone(stale[-2].get('value'))
        self.assertIn(self.demo.saved_voice_catalog()[name], stale[5])

    def test_generate_listener_rejects_repeat_submissions_while_pending(self):
        interface = self.interface()
        generate = next(fn for fn in interface.fns.values()
                        if getattr(fn.fn, '__name__', '') == 'generate_podcast_wrapper')
        self.assertEqual(generate.trigger_mode, 'once')
        self.assertEqual(generate.concurrency_limit, 1)

    def test_generation_emits_explicit_terminal_outcome_with_file(self):
        def generate(**kwargs):
            yield (24000, np.zeros(36000, dtype=np.int16)), None, 'Live', {}
            yield (24000, np.zeros(36000, dtype=np.int16)), None, 'Live', {}
            yield None, str(self.wav), 'Complete', {}
        self.demo.generate_podcast_streaming = generate
        calls = self.callbacks(self.interface())
        clock = SimpleNamespace(monotonic=iter([100, 103, 106, 107, 108]).__next__)
        with patch.object(app, 'time', clock):
            results = list(calls['generate_podcast_wrapper'](*self.args()))
            next_run = calls['generate_podcast_wrapper'](*self.args())
            try:
                next_progress = json.loads(next(next_run)[-1])['progress']
            finally:
                next_run.close()
        self.assertTrue(json.loads(results[0][-1])['running'])
        progress = [json.loads(row[-1])['progress'] for row in results]
        self.assertEqual(progress[0], {'percent': 0, 'eta': None})
        self.assertEqual([entry['seconds'] for entry in progress[1:3]], [1, 3])
        self.assertEqual(progress[1]['percent'], 75)
        self.assertEqual(progress[1]['eta'], 1)
        self.assertEqual(progress[2]['percent'], 95)
        self.assertIsNone(progress[2]['eta'])
        self.assertIsNone(progress[-1])
        self.assertEqual(next_progress['eta'], 7)
        self.assertEqual(json.loads(results[-1][-1])['completed'], True)
        self.assertFalse(json.loads(results[-1][-1])['running'])
        self.assertEqual(results[-1][1]['value'], str(self.wav))
        rejected = list(calls['generate_podcast_wrapper'](*self.args('')))[-1]
        self.assertFalse(json.loads(rejected[-1])['running'])
        self.assertFalse(json.loads(rejected[-1])['completed'])
        self.assertNotEqual(json.loads(rejected[-1])['id'], json.loads(results[-1][-1])['id'])
        self.assertNotIn('value', rejected[1])

    def test_valid_generation_resets_previous_live_stream_but_retains_completed_take(self):
        def generate(**kwargs):
            yield None, str(self.wav), 'Complete', {}
        self.demo.generate_podcast_streaming = generate
        calls = self.callbacks(self.interface())
        for _ in range(2):
            results = list(calls['generate_podcast_wrapper'](*self.args()))
            initial = results[0]
            self.assertEqual(initial[0], {'__type__': 'update', 'value': None})
            self.assertNotIn('value', initial[1])
            self.assertNotIn('value', initial[9])
            self.assertEqual(results[-1][1]['value'], str(self.wav))
        rejected = list(calls['generate_podcast_wrapper'](*self.args('')))[0]
        self.assertIsNone(rejected[0])
        self.assertNotIn('value', rejected[1])

    def test_random_examples_cycle_without_overwriting_editor_on_server(self):
        self.demo.example_scripts = [[1, 'One'], [2, 'Two'], [3, 'Three']]
        interface = self.interface()
        choose = self.callbacks(interface)['load_random_example']
        remaining = []
        seen = []
        for _ in range(3):
            example, remaining = choose(remaining)
            example = json.loads(example)
            seen.append(example['script'])
            self.assertIn('id', example)
        self.assertEqual(set(seen), {'One', 'Two', 'Three'})
        self.assertEqual(remaining, [])
        fn = next(fn for fn in interface.fns.values()
                  if getattr(fn.fn, '__name__', '') == 'load_random_example')
        self.assertFalse(any(getattr(output, 'elem_id', None) == 'composer-editor' for output in fn.outputs))
        self.assertEqual(fn.trigger_mode, 'once')

    def test_browser_control_messages_serialize_as_parseable_json_text(self):
        async def run():
            interface = self.interface()
            state = SessionState(interface)
            funcs = {getattr(fn.fn, '__name__', ''): fn for fn in interface.fns.values()}
            response = await interface.process_api(funcs['load_random_example'], [None],
                                                  state=state, session_hash='transport')
            self.assertIn('script', json.loads(response['data'][0]))
            response = await interface.process_api(funcs['generate_podcast_wrapper'], self.args('') + [False],
                                                  state=state, session_hash='transport')
            self.assertFalse(json.loads(response['data'][-1])['completed'])
        asyncio.run(run())

    def test_shared_interface_omits_private_operations(self):
        self.demo.share = True
        names = self.callbacks(self.interface())
        for name in ('save_recording', 'apply_voice_action', 'save_completed_take', 'open_saved_take', 'section_request'):
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
