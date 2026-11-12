import tempfile
from pathlib import Path
from uuid import uuid4

import soundfile as sf


def section_frames(path: Path) -> int:
    info = sf.info(path)
    if (info.samplerate, info.channels, info.format, info.subtype) != (24000, 1, "WAV", "PCM_16") or not info.frames:
        raise ValueError("Invalid section WAV")
    return info.frames


class SectionJob:
    def __init__(self, sections, inputs):
        self.id = uuid4().hex
        self.sections = tuple(sections)
        self.inputs = dict(inputs)
        self.paths: list[Path | None] = [None] * len(sections)
        self.frames = [0] * len(sections)
        self.status = ["Waiting"] * len(sections)
        self.details = [""] * len(sections)

    def run(self, indices, synthesize, cancelled):
        for i in indices:
            if cancelled.is_set():
                self.status[i] = "Stopped"
                continue
            self.status[i] = "Generating"
            self.details[i] = ""
            yield
            try:
                result = synthesize(self.sections[i], self.inputs)
                if result is not None and not cancelled.is_set():
                    self.frames[i] = section_frames(result)
                    self.paths[i] = Path(result)
            except (OSError, RuntimeError, ValueError) as error:
                self.paths[i] = None
                self.details[i] = str(error)
            self.status[i] = "Ready" if self.paths[i] else "Stopped" if cancelled.is_set() else "Failed"
            yield


def export_sections(job: SectionJob, media_root: Path | None) -> Path:
    if not job.paths or any(path is None for path in job.paths):
        raise ValueError("Finish all sections before exporting.")
    folder = Path(tempfile.mkdtemp(prefix="section-export-", dir=media_root))
    pending, final = folder / "writing.wav", folder / "full-script.wav"
    try:
        with sf.SoundFile(pending, "w", samplerate=24000, channels=1, subtype="PCM_16", format="WAV") as output:
            for index, path in enumerate(job.paths, 1):
                try:
                    with sf.SoundFile(path) as part:
                        if (part.samplerate, part.channels, part.format, part.subtype) != (24000, 1, "WAV", "PCM_16") or part.frames != job.frames[index - 1]:
                            raise ValueError("Invalid section WAV")
                        while part.tell() < part.frames:
                            samples = part.read(65536, dtype="int16")
                            if not samples.size:
                                raise ValueError("Incomplete section WAV")
                            try:
                                output.write(samples)
                            except (OSError, RuntimeError) as error:
                                raise ExportWriteError("Could not write the full WAV.") from error
                except (OSError, RuntimeError, ValueError) as error:
                    job.paths[index - 1] = None
                    job.frames[index - 1] = 0
                    job.status[index - 1] = "Failed"
                    job.details[index - 1] = "The section audio is missing or unreadable."
                    raise ValueError(f"Section {index} is missing or unreadable. Retry that section.") from error
        pending.replace(final)
        return final
    finally:
        pending.unlink(missing_ok=True)
        if not final.is_file():
            folder.rmdir()


class ExportWriteError(Exception):
    pass
