import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import soundfile as sf


def takes_directory() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
    root = Path(os.environ.get("VIBEVOICE_PRO_TAKES_DIR") or base / "VibeVoicePro" / "Takes").resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or root.is_relative_to(repo) or repo.is_relative_to(root):
        raise ValueError("The saved takes folder must be outside the app repository.")
    return root


@dataclass(frozen=True)
class Take:
    id: str
    name: str
    voices: tuple[str, ...]
    saved_at: datetime
    duration: float
    wav: Path


def wav_duration(path: Path) -> float:
    info = sf.info(path)
    if info.format != "WAV" or info.subtype != "PCM_16" or info.frames <= 0:
        raise ValueError("The completed WAV is missing or invalid.")
    return info.frames / info.samplerate


class SavedTakes:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def save(self, source: Path, name: str, voices: tuple[str, ...]) -> Take:
        name = name.strip() or "Latest generation"
        if len(name) > 80 or any(ord(ch) < 32 for ch in name):
            raise ValueError("Use a take name with up to 80 characters.")
        self.root.mkdir(parents=True, exist_ok=True)
        take_id = uuid4().hex
        saved_at = datetime.now().astimezone()
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=self.root) as pending:
            stage = Path(pending)
            wav = stage / "audio.wav"
            shutil.copyfile(source, wav)
            duration = wav_duration(wav)
            (stage / "take.json").write_text(json.dumps({
                "version": 1, "id": take_id, "name": name, "voices": voices,
                "saved_at": saved_at.isoformat(), "duration": duration,
            }), encoding="utf-8")
            stage.rename(self.root / take_id)
        return Take(take_id, name, voices, saved_at, duration, self.root / take_id / "audio.wav")

    def lookup(self, take_id: str) -> Take | None:
        if not isinstance(take_id, str) or not re.fullmatch(r"[0-9a-f]{32}", take_id):
            return None
        folder = self.root / take_id
        wav, metadata = folder / "audio.wav", folder / "take.json"
        try:
            if wav.resolve() != wav or metadata.resolve() != metadata or metadata.stat().st_size > 8192:
                return None
            data = json.loads(metadata.read_text(encoding="utf-8"))
            name, voices = data["name"], data["voices"]
            saved_at = datetime.fromisoformat(data["saved_at"])
            duration = wav_duration(wav)
            if (data["version"] != 1 or data["id"] != take_id or saved_at.tzinfo is None
                    or data["duration"] != duration or not isinstance(name, str) or not 0 < len(name) <= 80
                    or not isinstance(voices, list) or not all(isinstance(voice, str) for voice in voices)):
                return None
            return Take(take_id, name, tuple(voices), saved_at, duration, wav)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return None

    def list(self) -> list[Take]:
        try:
            takes = (self.lookup(path.name) for path in self.root.iterdir())
            return sorted((take for take in takes if take is not None),
                          key=lambda take: (take.saved_at, take.id), reverse=True)
        except OSError:
            return []
