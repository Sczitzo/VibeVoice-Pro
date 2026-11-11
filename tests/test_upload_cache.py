import os
import tempfile
import unittest
from pathlib import Path

from demo import gradio_demo


class UploadCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="vibevoice-upload-test-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_duplicate_upload_preserves_cached_file_and_timestamp(self):
        source = self.directory / "upload.tmp"
        destination = self.directory / "audio.wav"
        source.write_bytes(b"recorded audio")
        destination.write_bytes(source.read_bytes())
        os.utime(destination, (1, 1))
        timestamp = destination.stat().st_mtime_ns
        gradio_demo.cache_uploaded_files([str(source)], [str(destination)])
        self.assertEqual(destination.read_bytes(), b"recorded audio")
        self.assertEqual(destination.stat().st_mtime_ns, timestamp)
        self.assertFalse(source.exists())

    def test_first_upload_is_moved_to_the_cache(self):
        source = self.directory / "upload.tmp"
        destination = self.directory / "audio.wav"
        source.write_bytes(b"new recording")
        gradio_demo.cache_uploaded_files([str(source)], [str(destination)])
        self.assertEqual(destination.read_bytes(), b"new recording")
        self.assertFalse(source.exists())

    def test_source_already_in_cache_is_preserved(self):
        path = self.directory / "audio.wav"
        path.write_bytes(b"existing recording")
        gradio_demo.cache_uploaded_files([str(path)], [str(path)])
        self.assertEqual(path.read_bytes(), b"existing recording")

    def test_different_cached_content_uses_the_original_move(self):
        source = self.directory / "upload.tmp"
        destination = self.directory / "audio.wav"
        source.write_bytes(b"replacement recording")
        destination.write_bytes(b"different cached content")
        gradio_demo.cache_uploaded_files([str(source)], [str(destination)])
        self.assertEqual(destination.read_bytes(), b"replacement recording")
        self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
