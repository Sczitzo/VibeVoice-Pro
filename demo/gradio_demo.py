"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import filecmp
import json
import math
import shutil
import os
import sys
import tempfile
import time
from uuid import uuid4
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import List, Dict, Any, Iterator, NamedTuple, Optional
from unicodedata import decimal
from datetime import datetime
import threading
import numpy as np
import gradio as gr
from gradio import route_utils, routes
import librosa
import soundfile as sf
import torch
import os
import traceback
import re

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

if __package__:
    from .saved_takes import SavedTakes, takes_directory
    from .section_audio import ExportWriteError, SectionJob, export_sections
else:
    from saved_takes import SavedTakes, takes_directory
    from section_audio import ExportWriteError, SectionJob, export_sections

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
MAX_IMPORT_BYTES = 32 * 1024 * 1024
MAX_IMPORT_SECONDS = 60


def cache_uploaded_files(files, destinations):
    for source, destination in zip(files, destinations):
        if os.path.normcase(os.path.abspath(source)) == os.path.normcase(os.path.abspath(destination)):
            continue
        if os.path.isfile(destination) and filecmp.cmp(source, destination, shallow=False):
            os.unlink(source)
        else:
            route_utils.move_uploaded_files_to_cache([source], [destination])


def merge_speaker_turns(lines: List[str]) -> List[str]:
    """Keep consecutive paragraphs from the same speaker in one speaking turn."""
    merged = []
    previous_speaker = None
    for line in lines:
        speaker, _, text = line.partition(":")
        if speaker == previous_speaker:
            merged[-1] += " " + text.strip()
        else:
            merged.append(line)
            previous_speaker = speaker
    return merged


class ScriptCheck(NamedTuple):
    formatted: str = ""
    voice_slots: tuple[int, ...] = ()
    message: str = ""
    line: int | None = None
    previous_label: int | None = None
    sections: tuple["ScriptCheck", ...] = ()


def check_speaker_script(script: str, num_speakers: int, sectioned: bool = False) -> ScriptCheck:
    """Validate labels and return the text and reference slots the model needs."""
    if not isinstance(script, str) or not script.strip():
        return ScriptCheck(message="Please provide a script.")
    if isinstance(num_speakers, bool) or num_speakers not in (1, 2, 3, 4):
        return ScriptCheck(message="Choose between 1 and 4 speakers.")

    label_pattern = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
    lines = []
    for i, text in enumerate(script.splitlines(), 1):
        text = text.strip()
        if text:
            match = label_pattern.fullmatch(text)
            label = ("".join(str(decimal(digit)) for digit in match[1]).lstrip("0") or "0") if match else None
            lines.append((i, text, label, match[2] if match else text))
    first = 0 if any(label == "0" and body for _, _, label, body in lines) else 1
    labels = {str(first + slot): slot for slot in range(int(num_speakers))}
    turns = []
    groups = []
    previous = None
    for line, text, label, body in lines:
        if sectioned and text == "---":
            if not turns:
                return ScriptCheck(message=f"Line {line} starts an empty section.", line=line)
            groups.append(turns)
            turns = []
            previous = None
            continue
        if label is not None:
            if not body:
                return ScriptCheck(message=f"Line {line} needs text after the speaker label.", line=line)
            slot = labels.get(label)
            if slot is None:
                return ScriptCheck(message=f"Line {line} has a speaker outside your selection.", line=line)
            previous = slot + first
            turns.append((slot, body))
        elif (re.match(r"^Speaker(?:\s*[+-]?\s*\d|\s*:|\s+[A-Za-z]\s*:)", text, re.IGNORECASE)
              or (num_speakers > 1 and re.match(r"^Speaker\s+[^:\s]+\s*:", text, re.IGNORECASE))):
            return ScriptCheck(message=f"Line {line} has an invalid speaker label.", line=line)
        elif num_speakers == 1:
            turns.append((0, text))
        else:
            return ScriptCheck(message=f"Line {line} has no speaker label", line=line,
                               previous_label=previous)

    def format_turns(group):
        slots = tuple(sorted({slot for slot, _ in group}))
        model_ids = {slot: i for i, slot in enumerate(slots)}
        formatted = merge_speaker_turns([f"Speaker {model_ids[slot]}: {text}" for slot, text in group])
        return ScriptCheck(formatted="\n".join(formatted), voice_slots=slots)

    if sectioned:
        if not groups:
            return ScriptCheck(message="Put --- on its own line between sections.")
        if not turns:
            return ScriptCheck(message="Add text after the last section break.", line=lines[-1][0])
        groups.append(turns)
        return ScriptCheck(sections=tuple(format_turns(group) for group in groups))
    return format_turns(turns)


def spoken_units(script: ScriptCheck) -> int:
    text = " ".join(line.partition(":")[2] for line in script.formatted.splitlines())
    # Count unspaced Chinese and Japanese text by character and other text by word.
    return max(1, len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]|[^\W_\u3040-\u30ff\u3400-\u9fff]+", text)))


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5, adapter_path: Optional[str] = None, share: bool = False):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.share = share
        self.voices_dir = Path(__file__).resolve().parent / "voices"
        self.voice_lock = threading.Lock()
        self.device = device
        self.inference_steps = inference_steps
        self.adapter_path = adapter_path
        self.loaded_adapter_root: Optional[str] = None
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        self.generation_error: Optional[str] = None
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts

    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        self.loaded_adapter_root = None
        # Normalize potential 'mpx'
        if self.device.lower() == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        print(f"Using device: {self.device}")
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        # Decide dtype & attention
        if self.device == "mps":
            # float16, not float32: fp32 doubles the 1.5B model to ~11 GB,
            # which pushes 16 GB Apple Silicon machines into swap (measured
            # 92 s/step on an M1 Pro vs ~5 it/s at fp16, clean output).
            load_dtype = torch.float16
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        print(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl_primary,
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl_primary,
                )
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print(f"[ERROR] : {type(e).__name__}: {e}")
                print(traceback.format_exc())
                fallback_attn = "sdpa"
                print(f"Falling back to attention implementation: {fallback_attn}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation=fallback_attn,
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e
        if self.adapter_path:
            print(f"Loading fine-tuned assets from {self.adapter_path}")
            report = load_lora_assets(self.model, self.adapter_path)
            loaded_components = [
                name for name, loaded in (
                    ("language LoRA", report.language_model),
                    ("diffusion head LoRA", report.diffusion_head_lora),
                    ("diffusion head weights", report.diffusion_head_full),
                    ("acoustic connector", report.acoustic_connector),
                    ("semantic connector", report.semantic_connector),
                )
                if loaded
            ]
            if loaded_components:
                print(f"Loaded components: {', '.join(loaded_components)}")
            else:
                print("Warning: no adapter components were loaded; check the checkpoint path.")
            if report.adapter_root is not None:
                self.loaded_adapter_root = str(report.adapter_root)
                print(f"Adapter assets resolved to: {self.loaded_adapter_root}")
            else:
                self.loaded_adapter_root = self.adapter_path

        self.model.eval()

        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = self.voices_dir

        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        # Scan for all WAV files in the voices directory
        self.voice_presets = {}

        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir)
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        recorded_dir = voices_dir / "recorded"
        if getattr(self, "share", False):
            private_root = recorded_dir.resolve()
            wav_files = [name for name in wav_files if not (voices_dir / name).resolve().is_relative_to(private_root)]
        elif recorded_dir.is_dir():
            wav_files.extend(str(path.relative_to(voices_dir)) for path in sorted(recorded_dir.glob("*.wav")) if path.is_file())

        # Create dictionary with filename (without extension) as key
        seen_names = set()
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(os.path.basename(wav_file))[0]
            if name.casefold() in seen_names:
                continue
            seen_names.add(name.casefold())
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path

        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items(), key=lambda item: item[0].casefold()))

        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to the demo/voices directory.")

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    def saved_voice_names(self):
        recorded = getattr(self, "voices_dir", Path()) / "recorded"
        return [name for name, path in self.available_voices.items()
                if Path(path).parent == recorded and Path(path).suffix.lower() == ".wav"]

    def change_saved_voice(self, name, restore_from=None):
        if getattr(self, "share", False):
            raise gr.Error("Saved voices can only be removed on this computer.")
        recorded = self.voices_dir.resolve() / "recorded"
        trash = recorded / ".trash"
        with self.voice_lock:
            if recorded.resolve() != recorded or trash.resolve() != trash:
                raise gr.Error("The saved voice folder could not be verified.")
            if restore_from is None:
                if name not in self.saved_voice_names():
                    raise gr.Error("That saved voice is no longer available.")
                source = Path(self.available_voices[name]).absolute()
                destination = trash / uuid4().hex / source.name
            else:
                source = restore_from
                destination = recorded / f"{name}.wav"
                if destination.parent != recorded or destination.exists() or destination.is_symlink() or any(
                    voice.casefold() == name.casefold() for voice in self.available_voices
                ):
                    raise gr.Error("A voice with that name already exists. The removed recording is still in recovery.")
            source_root = source.parent if restore_from is None else source.parent.parent
            if source_root != (recorded if restore_from is None else trash) or source.resolve() != source:
                raise gr.Error("That recording could not be verified.")
            try:
                if restore_from is None:
                    destination.parent.mkdir(parents=True)
                source.rename(destination)
            except OSError as error:
                raise gr.Error("Could not move the recording. Try again.") from error
            # Update the index only after the move succeeds, without rescanning the library.
            updated = dict(self.available_voices)
            if restore_from is None:
                del updated[name]
            else:
                updated[name] = str(destination)
            self.available_voices = updated
            self.voice_presets = updated.copy()
            return destination

    def save_recorded_voice(self, audio, voice_name: str, start=None, end=None):
        if getattr(self, "share", False):
            raise gr.Error("Create Voice is available on this computer only.")
        if audio is None:
            raise gr.Error("Record your voice and stop recording before saving.")
        name = voice_name.strip() if isinstance(voice_name, str) else ""
        if not name:
            raise gr.Error("Enter a name for your voice.")
        if not re.fullmatch(r"[\w -]{1,60}", name):
            raise gr.Error("Use 1-60 letters, numbers, spaces, dashes, or underscores for the voice name.")
        if name.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", name, re.IGNORECASE):
            raise gr.Error("That name is reserved by Windows. Choose another voice name.")
        if isinstance(audio, (str, Path)):
            try:
                is_import = start is not None or end is not None
                if is_import and Path(audio).stat().st_size > MAX_IMPORT_BYTES:
                    raise gr.Error("Choose a WAV file no larger than 32 MiB.")
                with sf.SoundFile(audio) as source:
                    if source.format not in ("WAV", "WAVEX"):
                        raise ValueError("The uploaded recording must contain WAV audio.")
                    frames = source.frames
                    if is_import:
                        if (source.channels not in (1, 2) or source.samplerate <= 0
                                or not 0 < frames <= MAX_IMPORT_SECONDS * source.samplerate):
                            raise gr.Error("Choose a mono or stereo WAV up to 60 seconds long.")
                        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                                or not math.isfinite(value) for value in (start, end))
                                or not 0 <= start < end <= frames / source.samplerate + 1e-9):
                            raise gr.Error("Choose a start and end within the WAV.")
                        first = math.floor(start * source.samplerate + 0.5)
                        last = min(frames, math.floor(end * source.samplerate + 0.5))
                        if last <= first:
                            raise gr.Error("Choose a longer range.")
                        source.seek(first)
                        frames = last - first
                    samples = source.read(frames, dtype="float32")
                    if len(samples) != frames:
                        raise ValueError("The WAV is incomplete.")
                    audio = (source.samplerate, samples)
            except (OSError, RuntimeError, ValueError) as error:
                raise gr.Error("The WAV could not be read. Your recording is still available. Try Save Voice again.") from error
        if not isinstance(audio, (tuple, list)) or len(audio) != 2:
            raise gr.Error("The recording could not be read. Please record it again.")
        sample_rate, samples = audio
        samples = np.asarray(samples)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise gr.Error("The recording has an invalid sample rate. Please record it again.")
        if samples.ndim not in (1, 2) or not samples.size or samples.dtype.kind not in "fiu" or not np.isfinite(samples).all():
            raise gr.Error("The recording is empty or invalid. Please record it again.")
        if not np.any(samples):
            raise gr.Error("The recording is silent. Check your microphone and record again.")

        with self.voice_lock:
            if name.casefold() in {voice.casefold() for voice in self.available_voices}:
                raise gr.Error("A voice with that name already exists. Choose another name.")
            recorded_dir = self.voices_dir / "recorded"
            path = recorded_dir / f"{name}.wav"
            created = False
            try:
                recorded_dir.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as recording:
                    created = True
                    sf.write(recording, samples, int(sample_rate), format="WAV", subtype="PCM_16")
            except Exception as error:
                if created:
                    path.unlink(missing_ok=True)
                raise gr.Error(f"Could not save the voice recording: {error}") from error
            self.setup_voice_presets()
            return name, path

    def generate_podcast_streaming(self,
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.3,
                                 inference_steps: Optional[int] = None,
                                 seed: Optional[int] = None,
                                 disable_voice_cloning: bool = False,
                                 checked: ScriptCheck | None = None,
                                 cancelled: threading.Event | None = None,
                                 require_complete: bool = False,
                                 reset_stop: bool = True) -> Iterator[tuple]:
        generation_thread = None
        audio_streamer = None

        def stopped():
            return self.stop_generation or (cancelled is not None and cancelled.is_set())

        try:

            # Reset stop flag and set generating state
            if reset_stop:
                self.stop_generation = False
            self.is_generating = True
            self.generation_error = None

            check = checked if checked is not None else check_speaker_script(script, num_speakers)
            if check.message:
                self.is_generating = False
                raise gr.Error(check.message)

            # The processor pairs consecutive model speaker IDs with these references.
            choices = [speaker_1, speaker_2, speaker_3, speaker_4]
            selected_speakers = [choices[slot] for slot in check.voice_slots]

            voice_cloning_enabled = not disable_voice_cloning

            # Resolve per-run parameters
            resolved_inference_steps = int(inference_steps) if inference_steps is not None else int(self.inference_steps)
            resolved_seed: Optional[int]
            if seed is None:
                resolved_seed = None
            else:
                # Gradio Number may come through as float
                resolved_seed = int(seed)
                if resolved_seed < 0:
                    resolved_seed = None

            def _sanitize_filename_part(value: str, max_len: int = 60) -> str:
                value = (value or "").strip()
                value = re.sub(r"\s+", "_", value)
                value = re.sub(r"[^A-Za-z0-9_\-\+\.]+", "", value)
                return value[:max_len] if len(value) > max_len else value

            def _write_complete_wav(audio_int16: np.ndarray, sample_rate: int = 24000) -> str:
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                speakers_part = _sanitize_filename_part("-".join(selected_speakers) if selected_speakers else "speaker")
                cfg_part = _sanitize_filename_part(f"{cfg_scale:.2f}")
                seed_part = str(resolved_seed) if resolved_seed is not None else "rand"
                # Requested pattern: yyyyMMddhhmmss_[speaker]_[cfg_scale]_[inference_step][seed].wav
                basename = f"{ts}_{speakers_part}_{cfg_part}_{resolved_inference_steps}_{seed_part}.wav"
                out_path = os.path.join(tempfile.mkdtemp(prefix="result-", dir=getattr(self, "media_root", None)), basename)
                sf.write(out_path, np.asarray(audio_int16, dtype=np.int16), sample_rate, subtype="PCM_16")
                return out_path

            # Build initial log
            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={resolved_inference_steps}, Seed={(resolved_seed if resolved_seed is not None else 'random')}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            log += f"🔊 Voice cloning: {'Enabled' if voice_cloning_enabled else 'Disabled'}\n"
            if self.loaded_adapter_root:
                log += f"🧩 LoRA: {self.loaded_adapter_root}\n"

            # Check for stop signal
            if stopped():
                self.is_generating = False
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Keep recordings in place while this run reads them.
            voice_samples = [] if voice_cloning_enabled else None
            with self.voice_lock:
                for slot, speaker in zip(check.voice_slots, selected_speakers):
                    audio_path = self.available_voices.get(speaker)
                    if not audio_path:
                        raise gr.Error(f"Choose an available voice for Speaker {slot + 1}.")
                    if voice_samples is not None:
                        audio_data = self.read_audio(audio_path)
                        if len(audio_data) == 0:
                            raise gr.Error(f"Error: Failed to load audio for {speaker}")
                        voice_samples.append(audio_data)

            # Check for stop signal
            if stopped():
                self.is_generating = False
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            formatted_script = check.formatted.replace("’", "'")
            log += f"📝 Formatted script with {len(formatted_script.splitlines())} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"

            # Check for stop signal before processing
            if stopped():
                self.is_generating = False
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            start_time = time.time()

            processor_kwargs = {
                "text": [formatted_script],
                "padding": True,
                "return_tensors": "pt",
                "return_attention_mask": True,
            }
            processor_kwargs["voice_samples"] = [voice_samples] if voice_samples is not None else None

            inputs = self.processor(**processor_kwargs)
            # Move tensors to device
            target_device = self.device if self.device in ("cuda", "mps") else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)

            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )

            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer

            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer, voice_cloning_enabled, resolved_inference_steps, resolved_seed, target_device, stopped, require_complete)
            )
            generation_thread.start()

            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if stopped():
                audio_streamer.end()
                generation_thread.join()
                self.is_generating = False
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15 # Yield every 15 seconds
            min_chunk_size = sample_rate * 2 # At least 2 seconds of audio

            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)

            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all

            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if stopped():
                    audio_streamer.end()
                    break

                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk

                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)

                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()

                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)

                # Store for final statistics
                all_audio_chunks.append(audio_16bit)

                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)

                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time

                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True

                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate

                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"

                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)

                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time

            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio

            generation_thread.join()

            # Clean up
            self.current_streamer = None
            self.is_generating = False

            generation_time = time.time() - start_time

            # Check if stopped by user
            if stopped():
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            if self.generation_error:
                yield None, None, log + f"\n❌ Generation failed: {self.generation_error}", gr.update(visible=False)
                return

            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")

            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate

                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."

                # Yield the complete audio as a filepath so download uses our filename
                complete_wav_path = _write_complete_wav(complete_audio, sample_rate=sample_rate)
                yield None, complete_wav_path, final_log, gr.update(visible=False)
                return

            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                yield None, None, error_log, gr.update(visible=False)
                return

            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate

                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."

                # Final yield: Clear streaming audio and provide complete audio as filepath
                complete_wav_path = _write_complete_wav(complete_audio, sample_rate=sample_rate)
                yield None, complete_wav_path, final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)

        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
        finally:
            if generation_thread is not None and generation_thread.is_alive():
                self.stop_generation = True
                audio_streamer.end()
                generation_thread.join()
            self.current_streamer = None
            self.is_generating = False

    def _generate_with_streamer(
        self,
        inputs,
        cfg_scale,
        audio_streamer,
        voice_cloning_enabled: bool,
        inference_steps: int,
        seed: Optional[int],
        target_device: str,
        stopped,
        require_complete: bool,
    ):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if stopped():
                audio_streamer.end()
                return

            # Apply per-run DDPM steps
            try:
                self.model.set_ddpm_inference_steps(num_steps=int(inference_steps))
            except Exception as e:
                print(f"Warning: failed to set inference steps ({inference_steps}): {e}")

            generator = None
            if seed is not None:
                try:
                    if target_device == "cuda":
                        generator = torch.Generator(device="cuda")
                    else:
                        generator = torch.Generator()
                    generator.manual_seed(int(seed))
                except Exception as e:
                    print(f"Warning: failed to create seeded generator (seed={seed}, device={target_device}): {e}")
                    generator = None

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                generator=generator,
                audio_streamer=audio_streamer,
                stop_check_fn=stopped,
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
                is_prefill=voice_cloning_enabled,
            )
            capped = getattr(outputs, "reach_max_step_sample", None)
            if require_complete and capped is not None and capped.any().item():
                self.generation_error = "The model reached its limit before finishing this section."

        except Exception as e:
            self.generation_error = f"{type(e).__name__}: {e}"
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()

    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("Audio generation stop requested")

    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []

        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return

        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir)
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])

        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)

            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()

                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())

                if not script_content:
                    continue

                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)

                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")

            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")

        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")

    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()

        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)

        # If no speakers found, default to 1
        if not speakers:
            return 1

        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)

        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)


class AudioStreamBlocks(gr.Blocks):
    """Avoid Gradio 5.50 empty HLS playlists for status-only generator updates.

    None is not an audio chunk. Gradio otherwise publishes a zero-duration
    playlist even for rejected scripts or section jobs with no live playback.
    Preserve normal segment delivery and normal finalization once a stream exists.
    """

    async def handle_streaming_outputs(self, block_fn, data, session_hash, run,
                                       root_path=None, final=False):
        if session_hash is not None and run is not None:
            existing = self.pending_streams.get(session_hash, {}).get(run, {})
            data = list(data)
            for index, component in enumerate(block_fn.outputs):
                if (isinstance(component, gr.Audio) and component.streaming
                        and data[index] is None and component._id not in existing):
                    data[index] = gr.skip()
        return await super().handle_streaming_outputs(
            block_fn, data, session_hash, run, root_path=root_path, final=final)


def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""

    demo_dir = Path(__file__).parent
    recorder_js = (demo_dir / "voice_recorder.js").read_text(encoding="utf-8")
    browser_js = "\n".join((recorder_js, *((demo_dir / name).read_text(encoding="utf-8")
                                         for name in ("composer.js", "voice_menu.js", "script_check.js", "generated_audio.js", "studio_audio.js"))))
    css = (demo_dir / "eucalyptus.css").read_text(encoding="utf-8")
    theme = gr.themes.Base(primary_hue=gr.themes.colors.teal, font=["Inter", "Segoe UI", "sans-serif"])
    private_features = not getattr(demo_instance, "share", False)
    takes = SavedTakes(takes_directory()) if private_features else None
    recorder_setup = "initVoiceRecorder();initVoiceMenu();" if private_features else "document.getElementById('open-voice-recorder').hidden = true;"

    def render_saved_voice_names():
        names = escape(json.dumps(demo_instance.saved_voice_names()), quote=True)
        return f'<span data-saved-voice-names="{names}"></span>'

    with AudioStreamBlocks(title="VibeVoice", theme=theme, css=css,
                   js="() => {" + browser_js + "\n" + recorder_setup + "initComposer();initScriptCheck();initGeneratedAudio();initStudioAudio();}") as interface:
        with gr.Column(elem_id="guided-composer"):
            gr.HTML("""
                <header class="composer-header">
                    <a class="composer-skip" href="#composer-editor">Skip to Script</a>
                    <div class="composer-brand">
                        <span class="composer-brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
                        <div><h1>VibeVoice</h1><span>Guided Composer</span></div>
                    </div>
                    <div class="composer-header-actions">
                        <button type="button" id="show-speakers" class="composer-button mobile-only"
                                aria-controls="composer-speakers" aria-expanded="false">Speakers</button>
                        <button type="button" id="open-studio-library" class="composer-button" aria-haspopup="dialog">Takes &amp; Sections</button>
                        <button type="button" id="open-voice-recorder" class="composer-button"
                                aria-haspopup="dialog">Create Voice</button>
                    </div>
                </header>
            """, elem_id="composer-header", padding=False)

            with gr.Row(elem_id="composer-grid"):
                with gr.Column(elem_id="composer-script"):
                    with gr.Row(elem_id="script-heading"):
                        gr.HTML("<h2>Script</h2>", elem_id="composer-script-label", padding=False)
                        with gr.Row(elem_classes="script-tools"):
                            random_example_btn = gr.Button("Random Example", elem_id="random-example")
                            gr.HTML('<button type="button" id="undo-script" class="composer-button" hidden title="Undo last script change (Ctrl+Z)">Undo</button>', padding=False)
                            check_script_btn = gr.Button("Check Script", elem_id="check-script")
                            gr.HTML('<span id="word-count">0 words</span>', padding=False)
                    with gr.Column(elem_id="editor-frame"):
                        gr.HTML("""
                            <div id="speaker-chips" aria-label="Insert a speaker label">
                                <span>Insert speaker</span>
                                <button type="button" data-speaker="1" aria-label="Insert Speaker 1 label">Speaker 1</button>
                                <button type="button" data-speaker="2" aria-label="Insert Speaker 2 label">Speaker 2</button>
                                <button type="button" data-speaker="3" aria-label="Insert Speaker 3 label" hidden>Speaker 3</button>
                                <button type="button" data-speaker="4" aria-label="Insert Speaker 4 label" hidden>Speaker 4</button>
                            </div>
                        """, padding=False)
                        script_input = gr.Textbox(
                            label="Script", show_label=False, elem_id="composer-editor",
                            placeholder="Speaker 1: Welcome to our podcast…",
                            lines=5, container=False,
                        )
                        gr.HTML('<p id="composer-script-help">Start each turn with a speaker label.</p>', padding=False)
                        script_feedback = gr.HTML("", elem_id="script-check-feedback", padding=False)
                        gr.HTML('<button type="button" class="composer-button" id="keep-speaker" hidden>Keep with Speaker</button>',
                                padding=False, elem_id="keep-speaker-host")
                        generation_feedback = gr.Markdown(visible=False, elem_id="generation-feedback")
                    checked_script = gr.State(False)

                with gr.Column(elem_id="composer-speakers"):
                    with gr.Row(elem_id="speaker-heading"):
                        gr.HTML("""
                            <div class="speaker-heading-label"><h2>Speakers</h2>
                                <span class="composer-help">
                                    <button type="button" aria-label="About speaker assignments" aria-describedby="speaker-tooltip">?</button>
                                    <span id="speaker-tooltip" class="composer-tooltip" role="tooltip" hidden>Choose a voice for each speaker label in your script.</span>
                                </span>
                            </div>
                        """, padding=False)
                        num_speakers = gr.Radio([1, 2, 3, 4], value=2, label="Number of Speakers", elem_id="speaker-count")
                    with gr.Column(elem_id="voice-assignments"):
                        available_speaker_names = sorted(demo_instance.available_voices, key=str.casefold)
                        default_speakers = ["en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman"]
                        speaker_selections = []
                        for index, default_voice in enumerate(default_speakers):
                            fallback = available_speaker_names[index % len(available_speaker_names)] if available_speaker_names else None
                            speaker_selections.append(gr.Dropdown(
                                choices=available_speaker_names,
                                value=default_voice if default_voice in available_speaker_names else fallback,
                                label=f"Speaker {index + 1}", visible=index < 2,
                                elem_classes=["speaker-item", f"tone-{index + 1}"],
                            ))
                    gr.HTML("""
                        <button type="button" class="composer-button" id="open-advanced-settings" aria-haspopup="dialog">Advanced Settings</button>
                        <button type="button" class="composer-button mobile-only" id="back-to-script">Back to Script</button>
                    """, padding=False, elem_id="speaker-actions")
                    if private_features:
                        gr.HTML("""
                            <div id="voice-menu-feedback" hidden>
                                <span role="status"></span>
                                <button type="button" class="composer-button" hidden>Undo</button>
                            </div>
                            <span id="voice-menu-help" hidden>Use the arrow keys to highlight a saved voice.
                                Press Shift+F10 to reveal Delete, then Enter to remove it or Escape to cancel.</span>
                            <span id="voice-menu-announcement" class="voice-menu-sr-only" role="status"></span>
                        """, padding=False, elem_id="voice-menu-status")
                        voice_catalog = gr.HTML(render_saved_voice_names(), visible="hidden", elem_id="voice-menu-catalog")
                        voice_command = gr.Textbox(visible="hidden", elem_id="voice-menu-command")
                        voice_action = gr.Button(visible="hidden", elem_id="voice-menu-submit")
                        voice_result = gr.HTML(visible="hidden", elem_id="voice-menu-result")
                        undo_voice = gr.State()
                        refresh_voices = gr.Button(visible="hidden", elem_id="refresh-voices")
                        known_voices = gr.State(available_speaker_names)

            with gr.Row(elem_id="composer-transport"):
                gr.HTML("""
                    <div id="generated-audio-player">
                        <audio hidden preload="metadata" aria-label="Generated audio"></audio>
                        <button type="button" class="composer-button" id="playback-toggle" disabled>Play</button>
                        <div class="transport-track">
                            <div class="track-meta">
                                <span id="playback-title" role="status">No audio yet</span>
                                <span class="take-controls">
                                    <button type="button" data-replay-audio disabled>Replay</button>
                                    <button type="button" data-clear-audio disabled>Clear</button>
                                </span>
                                <span class="track-time"><span id="playback-position-time"><span id="playback-elapsed">0:00</span> / </span><span id="playback-duration">0:00</span></span>
                            </div>
                            <div class="transport-wave transport-waveform">
                                <progress id="generation-progress" aria-label="Audio generation progress" hidden></progress>
                                <canvas id="playback-waveform" width="480" height="32" aria-hidden="true"></canvas>
                                <input type="range" id="playback-seek" min="0" max="1000" step="1" value="0" aria-label="Playback position" disabled>
                            </div>
                        </div>
                        <a class="composer-button" id="download-wav" download aria-disabled="true">Download WAV</a>
                    </div>
                """, elem_id="playback-transport", padding=False)
                with gr.Column(elem_id="composer-actions"):
                    generate_btn = gr.Button("Generate Audio", variant="primary", elem_id="generate-audio")
                    stop_btn = gr.Button("Stop", visible=False, elem_id="stop-generation")

        audio_output = gr.Audio(label="Live Audio", type="numpy", streaming=True, autoplay=False,
                                show_download_button=False, elem_id="live-generated-audio")
        complete_audio_output = gr.File(type="filepath", interactive=False, visible="hidden")
        generation_outcome = gr.Textbox(visible="hidden")
        completed_take = gr.State(None)
        section_job = gr.State(None)
        streaming_status = gr.Markdown("Generating audio…", visible=False, elem_id="streaming-status")
        gr.HTML("""
            <dialog id="composer-settings" aria-labelledby="composer-settings-title">
                <header class="recorder-header">
                    <h2 id="composer-settings-title">Advanced Settings</h2>
                    <button type="button" class="composer-button" data-close-settings>Done</button>
                </header>
                <div id="settings-fields-host"></div>
            </dialog>
        """, padding=False, elem_id="composer-settings-host")
        with gr.Column(elem_id="composer-settings-fields"):
            gr.HTML("""
                <span class="composer-help" id="guidance-help">
                    <button type="button" aria-label="About guidance strength" aria-describedby="guidance-tooltip">?</button>
                    <span id="guidance-tooltip" class="composer-tooltip" role="tooltip" hidden>Change how strongly VibeVoice follows the selected voice and script. Start with 1.3.</span>
                </span>
            """, padding=False, elem_id="guidance-help-source")
            cfg_scale = gr.Slider(1.0, 12.0, value=1.3, step=0.05, label="Guidance Strength",
                                  elem_id="guidance-strength")
            inference_steps = gr.Slider(1, 50, value=int(getattr(demo_instance, "inference_steps", 10)),
                                        step=1, label="Inference Steps", elem_id="inference-steps")
            seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)", elem_id="generation-seed")
            disable_voice_cloning = gr.Checkbox(value=False, label="Disable voice cloning",
                                               info="Generate voices without using the selected voice recordings.")
            section_mode = gr.Checkbox(label="Generate in sections (preview)", value=False, visible=private_features, elem_id="section-mode",
                info="Put --- on its own line between sections. Review pacing before sharing.")
            with gr.Accordion("Technical Details", open=False):
                log_output = gr.Textbox(label="Generation Log", lines=8, max_lines=15, interactive=False)
            with gr.Accordion("About & Limitations", open=False):
                gr.Markdown("VibeVoice can produce inaccurate speech. Check your output before sharing. "
                            "Use voices with permission and disclose AI-generated audio when you share it.")

        gr.HTML("""
            <dialog id="studio-library" aria-labelledby="studio-library-title">
                <header class="recorder-header"><h2 id="studio-library-title">Takes &amp; Sections</h2>
                <button type="button" class="composer-button" data-close-studio>Done</button></header>
                <div id="studio-library-fields-host"></div>
            </dialog>
        """, padding=False)
        with gr.Column(elem_id="studio-library-fields", visible=private_features):
            section_panel = gr.HTML("<p>Section results appear here when you generate in sections.</p>", padding=False, elem_id="generation-sections")
            section_command = gr.Textbox(visible="hidden", elem_id="section-command")
            section_action = gr.Button(visible="hidden", elem_id="section-action")
            if takes is not None:
                with gr.Row(elem_id="take-save-controls"):
                    take_name = gr.Textbox(label="Take name", placeholder="Latest generation", lines=1, max_lines=1)
                    save_take = gr.Button("Save Take", variant="primary")
                take_status = gr.Markdown()
                take_list = gr.HTML(padding=False)
                refresh_takes = gr.Button(visible="hidden", elem_id="refresh-takes")
                open_folder = gr.Button("Open takes folder", visible=sys.platform == "win32")
                take_id = gr.Textbox(visible=False)
                take_file = gr.File(visible=False)
                open_take = gr.Button(visible=False)

        gr.HTML("""
            <dialog id="voice-recorder" aria-labelledby="voice-recorder-title">
                <header class="recorder-header">
                    <h2 id="voice-recorder-title">Create Voice</h2>
                    <button type="button" data-close>Close</button>
                </header>
                <div class="recorder-actions voice-modes" role="group" aria-label="Voice source">
                    <button type="button" data-voice-mode="record" aria-pressed="true">Record</button>
                    <button type="button" data-voice-mode="import" aria-pressed="false">Import audio</button>
                </div>
                <ol class="recorder-steps" aria-label="Recording steps">
                    <li data-step="microphone" aria-current="step">Microphone</li>
                    <li data-step="record">Record</li>
                    <li data-step="listen">Listen</li>
                    <li data-step="save">Save</li>
                </ol>
                <section class="recorder-reading" aria-labelledby="voice-reading-title">
                    <h3 id="voice-reading-title" tabindex="-1">Read Aloud</h3>
                    <p class="recorder-help">Read it like you’re telling a friend what happened.</p>
                    <p class="recorder-passage">
                        I lucked out at the market and got the last basket of peaches. I picked up fresh bread
                        and a jar of jam, too.
                    </p>
                    <p class="recorder-passage">
                        I was enjoying my usual walk home when it started pouring. By the time I got to my
                        front door, I was soaked. I looked in my bag for my keys. Nothing.
                    </p>
                    <p class="recorder-passage">
                        Then I checked my jacket pocket, and there they were.
                    </p>
                </section>
                <section data-pane="microphone">
                    <div class="recorder-mic">
                        <label for="voice-microphone">Microphone</label>
                        <select id="voice-microphone" data-microphone aria-describedby="voice-microphone-info">
                            <option value="">System default microphone</option>
                        </select>
                        <p class="recorder-help" id="voice-microphone-info" data-microphone-info>Choose a microphone.</p>
                    </div>
                    <div class="recorder-actions">
                        <button type="button" data-refresh>Refresh Microphones</button>
                        <button type="button" data-record class="primary">Start Recording</button>
                    </div>
                    <button type="button" data-cancel class="recorder-next" hidden>Cancel Request</button>
                    <button type="button" data-return class="recorder-next" hidden>Back to Recording</button>
                </section>
                <section data-pane="record" hidden>
                    <h3 data-capture-state>Recording</h3>
                    <div class="recorder-timer" data-elapsed aria-label="Elapsed time">0:00</div>
                    <div class="recorder-bars" aria-hidden="true">
                        <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
                    </div>
                    <meter data-level min="0" max="100" value="0" hidden></meter>
                    <p data-level-message role="status">Listening for sound…</p>
                    <p class="recorder-help" data-active-device></p>
                    <div class="recorder-actions">
                        <button type="button" data-pause>Pause</button>
                        <button type="button" data-stop class="primary" disabled>Stop Recording</button>
                    </div>
                </section>
                <section data-pane="listen" hidden>
                    <h3>Listen to Your Recording</h3>
                    <audio controls hidden preload="metadata" aria-label="Your voice recording"></audio>
                    <div class="recorder-actions">
                        <button type="button" data-retake>Record Again</button>
                        <button type="button" data-name class="primary">Name Voice</button>
                    </div>
                </section>
                <section data-pane="import" hidden>
                    <label for="voice-import-file">Choose a WAV file</label>
                    <input id="voice-import-file" type="file" accept=".wav,audio/wav">
                    <p class="recorder-help">Mono or stereo, up to 60 seconds and 32 MiB.</p>
                    <div data-import-range hidden>
                        <p data-import-filename></p>
                        <canvas width="420" height="72" role="img" aria-label="Waveform with selected range"></canvas>
                        <p class="recorder-help" data-import-duration></p>
                        <div class="recorder-actions">
                            <label>Start (seconds) <input data-import-start type="number" min="0" step="any"></label>
                            <label>End (seconds) <input data-import-end type="number" min="0" step="any"></label>
                        </div>
                        <div class="recorder-actions">
                            <button type="button" data-import-preview>Preview range</button>
                            <button type="button" class="primary" data-import-name>Name Voice</button>
                        </div>
                    </div>
                </section>
                <section data-pane="save" hidden>
                    <div data-save-controls></div>
                    <button type="button" data-back>Back to Listen</button>
                    <div class="recorder-actions recorder-saved-actions">
                        <a data-saved-download download hidden>Download WAV</a>
                        <button type="button" data-done class="primary">Done</button>
                    </div>
                </section>
                <p data-recording-status role="status"></p>
            </dialog>
        """, elem_id="composer-recorder-host", padding=False)
        with gr.Column(elem_id="voice-save-panel"):
            voice_recording = gr.File(type="filepath", file_types=[".wav"], visible=False)
            import_start = gr.Number(visible=False)
            import_end = gr.Number(visible=False)
            voice_name = gr.Textbox(label="Voice Name", placeholder="My voice…", elem_id="voice-name")
            save_voice_btn = gr.Button("Save Voice", variant="primary", elem_id="save-voice-btn")
            voice_save_status = gr.Markdown(elem_id="voice-save-status")
            saved_voice_file = gr.File(interactive=False, visible="hidden")
            voice_saved = gr.Checkbox(value=False, visible=False)

        voice_output_count = len(speaker_selections) + 2

        def refresh_voice_choices(previous=None, *selections):
            choices = sorted(demo_instance.available_voices, key=str.casefold)
            if choices == previous:
                return [gr.skip()] * voice_output_count
            updates = [gr.update(choices=choices) for _ in speaker_selections]
            for index, selected in enumerate(selections):
                if selected and selected not in choices:
                    updates[index]['value'] = None
            return updates + [choices, render_saved_voice_names()]

        def save_recording(audio, name, start=None, end=None):
            unchanged = [gr.skip()] * voice_output_count
            empty_file = gr.update(value=None)
            if audio is None:
                return unchanged + [
                    "Voice not saved. Check the recording message and try Save Voice again.", empty_file, False
                ]
            try:
                if start is None and end is None:
                    saved_name, saved_path = demo_instance.save_recorded_voice(audio, name)
                else:
                    saved_name, saved_path = demo_instance.save_recorded_voice(audio, name, start, end)
            except gr.Error as error:
                return unchanged + [error.message, empty_file, False]
            return refresh_voice_choices() + [
                f"Saved **{saved_name}.wav**. Choose it from any speaker dropdown.",
                gr.update(value=str(saved_path)), True
            ]

        if private_features:
            voice_outputs = speaker_selections + [known_voices, voice_catalog]
            gr.on(
                triggers=[interface.load, refresh_voices.click],
                fn=refresh_voice_choices, inputs=[known_voices] + speaker_selections, outputs=voice_outputs,
                preprocess=False,
                queue=False, api_name=False, show_progress="hidden",
            )

            def apply_voice_action(command, undo, *selections):
                unchanged = [gr.skip()] * voice_output_count
                def render_result(result):
                    payload = escape(json.dumps(result), quote=True)
                    return f'<span data-voice-result="{payload}" data-update="{uuid4().hex}"></span>'

                try:
                    if command == "undo" and undo:
                        name, path = undo
                        demo_instance.change_saved_voice(name, restore_from=path)
                        result = {"message": f"{name} restored.", "undo": False, "changed": True}
                        next_undo = None
                    elif isinstance(command, str) and command.startswith("delete:"):
                        name = command.removeprefix("delete:")
                        path = demo_instance.change_saved_voice(name)
                        slots = [str(i) for i, choice in enumerate(selections, 1) if choice == name]
                        result = {"message": f"{name} removed.", "undo": True, "changed": True, "slots": slots}
                        next_undo = (name, path)
                    else:
                        raise gr.Error("There is no voice removal to undo.")
                except gr.Error as error:
                    result = {"message": error.message, "undo": bool(undo), "error": True}
                    return unchanged + [gr.skip(), render_result(result)]
                return refresh_voice_choices(None, *selections) + [next_undo, render_result(result)]

            voice_event = voice_action.click(
                apply_voice_action, inputs=[voice_command, undo_voice] + speaker_selections,
                outputs=voice_outputs + [undo_voice, voice_result], preprocess=False,
                queue=False, api_name=False, show_progress="hidden",
            )
            voice_event.failure(fn=None, js="() => window.vibevoiceVoiceMenu.finish()", queue=False)
            save_event = save_voice_btn.click(
                fn=save_recording,
                inputs=[voice_recording, voice_name, import_start, import_end],
                outputs=voice_outputs + [voice_save_status, saved_voice_file, voice_saved],
                js="async (...inputs) => await window.vibevoiceRecorder.prepareSave(inputs)",
                queue=False,
            )
            save_event.success(
                fn=None, inputs=[voice_saved, saved_voice_file],
                js="(saved, file) => window.vibevoiceRecorder.finishSave(saved, false, file)",
                queue=False,
            )
            save_event.failure(
                fn=None, js="() => window.vibevoiceRecorder.finishSave(false, true)", queue=False,
            )

        def update_speaker_visibility(num_speakers):
            updates = []
            for i in range(4):
                updates.append(gr.update(visible=(i < num_speakers)))
            return updates

        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections
        )

        if takes is not None:
            take_media_lock = threading.Lock()

            def render_takes():
                rows = []
                for take in takes.list():
                    timestamp = take.saved_at.astimezone().strftime("%b %d, %Y at %I:%M %p")
                    seconds = int(take.duration)
                    summary = escape(f"{seconds // 60}:{seconds % 60:02d} · {', '.join(take.voices)} · {timestamp}")
                    rows.append(f"""<li class="studio-audio-row">
                        <div><strong>{escape(take.name)}</strong><small>{summary}</small></div>
                        <div class="studio-audio-actions">
                            <button type="button" class="composer-button" data-take-id="{take.id}" data-take-action="play">Play</button>
                            <button type="button" class="composer-button" data-take-id="{take.id}" data-take-action="download">Download WAV</button>
                        </div>
                        <audio controls hidden aria-label="{escape(take.name)}"></audio></li>""")
                content = f"<ul>{''.join(rows)}</ul>" if rows else "<p>No saved takes yet.</p>"
                return f'<section class="studio-audio-list"><h3>Saved takes</h3>{content}<p id="take-action-status" role="status"></p></section>'

            def save_completed_take(completed, name):
                if completed is None:
                    return gr.skip(), "Generate or export a complete WAV first."
                try:
                    source, voices = completed
                    take = takes.save(Path(source), name or "", voices)
                except (OSError, RuntimeError, ValueError) as error:
                    return gr.skip(), f"Could not save this take. {escape(str(error))}"
                return render_takes(), f"Saved **{escape(take.name)}**."

            def open_saved_take(identifier):
                take = takes.lookup(identifier)
                if take is None:
                    return None
                basename = re.sub(r"[^\w -]", "", take.name).strip(" .") or "Take"
                target = Path(take_file.GRADIO_CACHE) / "saved-takes" / take.id / f"{basename}-{take.id[:8]}.wav"
                with take_media_lock:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.is_file():
                        temporary = target.with_suffix(".tmp")
                        try:
                            shutil.copyfile(take.wav, temporary)
                            temporary.replace(target)
                        finally:
                            temporary.unlink(missing_ok=True)
                return str(target)

            def open_takes_folder():
                try:
                    takes.root.mkdir(parents=True, exist_ok=True)
                    os.startfile(takes.root)
                    return "Opened the saved takes folder."
                except OSError:
                    return "Could not open the saved takes folder."

            gr.on(
                triggers=[interface.load, refresh_takes.click], fn=render_takes, outputs=take_list,
                api_name=False, queue=False, show_progress="hidden",
            )
            save_take.click(save_completed_take, [completed_take, take_name], [take_list, take_status], api_name=False, queue=False)
            open_folder.click(open_takes_folder, outputs=take_status, api_name=False, queue=False)
            open_take.click(open_saved_take, take_id, take_file, api_name="open_saved_take", show_api=False, queue=False)

        generation_outcome.change(
            fn=None,
            inputs=[generation_outcome, complete_audio_output],
            js="(outcome, file) => { window.vibevoicePlayback?.finish(outcome, file); return []; }",
            queue=False,
        )

        def script_feedback_updates(script, count, check, sectioned=False):
            suggestion = check.previous_label
            snapshot = sha256(f"{count}\n{int(sectioned)}\n{script}".encode("utf-8")).hexdigest()
            if check.message:
                detail = ""
                if check.line is not None:
                    excerpt = script.splitlines()[check.line - 1].strip()
                    excerpt = excerpt[:157] + "…" if len(excerpt) > 160 else excerpt
                    detail = f'<p class="script-check-line">{escape(excerpt)}</p>'
                if suggestion is not None:
                    detail += f"<p>Keep it with Speaker {suggestion} or add a different speaker label.</p>"
                message = f'<strong>{escape(check.message)}</strong>{detail}'
            else:
                message = "Script ready."
            style = "script-check-issue" if check.message else "script-check-ready"
            line = check.line if check.line is not None else ""
            label = suggestion if suggestion is not None else ""
            html = (f'<div class="{style}" role="status" hidden data-snapshot="{snapshot}" '
                    f'data-line="{line}" data-label="{label}">{message}</div>')
            return html, True

        def check_script_inputs(script, count, sectioned=False):
            sectioned = bool(sectioned and private_features)
            return script_feedback_updates(script, count, check_speaker_script(script, count, sectioned), sectioned)

        def refresh_script_check(script, count, has_checked, sectioned=False):
            if not has_checked:
                return gr.skip(), gr.skip()
            return check_script_inputs(script, count, sectioned)

        script_check_outputs = [script_feedback, checked_script]
        check_script_btn.click(
            fn=check_script_inputs, inputs=[script_input, num_speakers, section_mode], outputs=script_check_outputs,
            queue=False, api_name=False, show_progress="hidden",
        )
        gr.on(
            triggers=[script_input.change, num_speakers.change, section_mode.change],
            fn=refresh_script_check, inputs=[script_input, num_speakers, checked_script, section_mode],
            outputs=script_check_outputs, queue=False, api_name=False, show_progress="hidden",
        )

        active_lock = threading.Lock()
        active = None
        last_timing = None

        def claim_model(request):
            nonlocal active
            with active_lock:
                if active is not None:
                    return None
                cancelled = threading.Event()
                active = (getattr(request, "session_hash", None), cancelled)
                return cancelled

        def release_model():
            nonlocal active
            with active_lock:
                active = None

        def generation_update(log, *, running=False, live=None, streaming=False,
                              feedback=None, warning=None, completed=None, job=None, panel=None,
                              progress=None):
            return (
                live, gr.update(value=completed[0]) if completed else gr.skip(), log,
                gr.update(visible=streaming), gr.update(visible=not running),
                gr.update(visible=running, interactive=True, value="Stop"),
                gr.update(value=warning, visible=bool(warning)),
                *(feedback if feedback is not None else (gr.skip(), gr.skip())),
                completed if completed else gr.skip(), job if job is not None else gr.skip(),
                panel if panel is not None else gr.skip(),
                json.dumps({"id": uuid4().hex, "running": running, "completed": bool(completed),
                            "progress": progress}),
            )

        def render_sections(job, running=False, note=""):
            rows = []
            for i, status in enumerate(job.status):
                command = f"{job.id}:{i}"
                if status == "Ready":
                    button = (f'<button type="button" class="composer-button" '
                              f'data-section-preview="{escape(str(job.paths[i]))}">Preview</button>')
                elif status in ("Failed", "Stopped"):
                    disabled = " disabled" if running else ""
                    button = (f'<button type="button" class="composer-button" data-section-command="{command}"'
                              f'{disabled}>Retry section</button>')
                else:
                    button = ""
                rows.append(f'<li class="studio-audio-row" data-status="{status}">'
                            f'<strong>Section {i + 1}</strong><div class="studio-audio-actions">'
                            f'<span>{status}</span>{button}</div>'
                            f'{f"<small>{escape(job.details[i])}</small>" if job.details[i] else ""}'
                            f'<audio controls hidden aria-label="Section {i + 1} preview"></audio></li>')
            ready = job.status.count("Ready")
            disabled = " disabled" if ready != len(rows) or running else ""
            return (f'<section class="studio-audio-list"><h3>Generation sections</h3>'
                    f'<p role="status">{ready} of {len(rows)} sections ready. {escape(note)}</p>'
                    f'<ul>{"".join(rows)}</ul><button type="button" class="composer-button" '
                    f'data-section-command="{job.id}:export"{disabled}>Export full WAV</button></section>')

        def run_sections(job, indices, cancelled, feedback=None):
            def synthesize(section, inputs):
                completed = None
                last_log = ""
                for _, path, last_log, _ in demo_instance.generate_podcast_streaming(
                    **inputs, checked=section, cancelled=cancelled, require_complete=True
                ):
                    if path:
                        completed = path
                if completed is None and "The model reached its limit" in last_log:
                    raise ValueError("The model reached its limit before finishing this section.")
                return completed

            started = time.monotonic()
            work = {index: spoken_units(job.sections[index]) for index in indices}
            for _ in job.run(work, synthesize, cancelled):
                finished = sum(units for index, units in work.items() if job.status[index] == "Ready")
                remaining = sum(units for index, units in work.items()
                                if job.status[index] in ("Waiting", "Generating"))
                eta = math.ceil((time.monotonic() - started) * remaining / finished) if finished else None
                yield generation_update("Generating sections…", running=True, job=job,
                                        panel=render_sections(job, running=True), feedback=feedback,
                                        progress={"ready": job.status.count("Ready"), "total": len(job.sections),
                                                  "eta": eta})
                feedback = None
            note = "Stopped. Retry any unfinished section." if cancelled.is_set() else ""
            yield generation_update("Section results are ready to review.", job=job,
                                    panel=render_sections(job, note=note))

        def generate_podcast_wrapper(num_speakers, script, speaker_1, speaker_2, speaker_3, speaker_4,
                                     cfg_scale, inference_steps, seed, disable_voice_cloning,
                                     sectioned=False, request: gr.Request = None):
            nonlocal last_timing
            sectioned = bool(sectioned and private_features)
            check = check_speaker_script(script, num_speakers, sectioned)
            feedback = script_feedback_updates(script, num_speakers, check, sectioned)
            if check.message:
                # Status-only updates must not create an empty live stream.
                yield generation_update(check.message, feedback=feedback)
                return
            choices = (speaker_1, speaker_2, speaker_3, speaker_4)
            slots = {slot for part in check.sections for slot in part.voice_slots} if sectioned else check.voice_slots
            for slot in slots:
                if choices[slot] not in demo_instance.available_voices:
                    message = f"Choose an available voice for Speaker {slot + 1}."
                    yield generation_update(message, warning=message)
                    return
            inputs = dict(num_speakers=int(num_speakers), script=script,
                          speaker_1=speaker_1, speaker_2=speaker_2, speaker_3=speaker_3, speaker_4=speaker_4,
                          cfg_scale=cfg_scale, inference_steps=inference_steps, seed=seed,
                          disable_voice_cloning=disable_voice_cloning)
            cancelled = claim_model(request)
            if cancelled is None:
                yield generation_update("The model is busy.", warning="Wait for the current generation to finish.")
                return
            try:
                if sectioned:
                    job = SectionJob(check.sections, inputs)
                    yield from run_sections(job, range(len(job.sections)), cancelled, feedback=feedback)
                    return
                timing_key = (tuple(choices[slot] for slot in check.voice_slots),
                              inference_steps, cfg_scale, disable_voice_cloning)
                units = spoken_units(check)
                # shortcut: script length only approximates spoken duration; use measured
                # alignment if the model later reports which words it has finished.
                expected_seconds = max(2, units / 2.25)
                initial_eta = None
                if last_timing and last_timing[0] == timing_key:
                    _, seconds_per_unit, compute_per_second = last_timing
                    expected_seconds = max(2, units * seconds_per_unit)
                    initial_eta = math.ceil(expected_seconds * compute_per_second)
                started = time.monotonic()
                yield generation_update("Starting generation…", running=True, feedback=feedback,
                                        progress={"percent": 0, "eta": initial_eta})
                final_log = "Starting generation…"
                completed = None
                generated_seconds = 0.0
                for streaming_audio, complete_audio, log, _ in demo_instance.generate_podcast_streaming(
                    **inputs, checked=check, cancelled=cancelled
                ):
                    final_log = log
                    if complete_audio and not cancelled.is_set():
                        completed = (complete_audio, tuple(choices[i] for i in check.voice_slots))
                    if streaming_audio is not None and not cancelled.is_set():
                        sample_rate, samples = streaming_audio
                        generated_seconds += len(samples) / sample_rate
                        remaining = expected_seconds - generated_seconds
                        elapsed = time.monotonic() - started
                        eta = math.ceil(remaining * elapsed / generated_seconds) if remaining > 0 and elapsed > 0 else None
                        yield generation_update(log, running=True, live=streaming_audio, streaming=True,
                                                progress={"seconds": int(generated_seconds),
                                                          "percent": min(95, int(100 * generated_seconds / expected_seconds)),
                                                          "eta": eta})
                if completed and generated_seconds >= 2:
                    last_timing = (timing_key, generated_seconds / units,
                                   (time.monotonic() - started) / generated_seconds)
                warning = None
                if not completed and not cancelled.is_set():
                    warning = "Audio wasn’t generated. Check Technical Details and try again."
                yield generation_update(final_log, completed=completed, warning=warning)
            except Exception as error:
                traceback.print_exc()
                yield generation_update(f"Generation failed: {error}", warning="Audio wasn’t generated. Try again.")
            finally:
                release_model()

        def section_request(command, job, request: gr.Request = None):
            if job is None or not isinstance(command, str) or not command.startswith(f"{job.id}:"):
                yield generation_update("This section job is no longer active.")
                return
            action = command.split(":", 1)[1]
            cancelled = claim_model(request)
            if cancelled is None:
                yield generation_update("The model is busy.", warning="Wait for the current generation to finish.")
                return
            try:
                if action == "export":
                    try:
                        path = export_sections(job, getattr(demo_instance, "media_root", None))
                        slots = sorted({slot for section in job.sections for slot in section.voice_slots})
                        voices = tuple(job.inputs[f"speaker_{slot + 1}"] for slot in slots)
                        yield generation_update("Full WAV ready.", completed=(str(path), voices), job=job,
                            panel=render_sections(job, note="Full WAV ready in Generated Audio."))
                    except (ExportWriteError, OSError, RuntimeError, ValueError) as error:
                        yield generation_update("Export unavailable.", job=job, panel=render_sections(job, note=str(error)))
                elif action.isascii() and action.isdecimal() and len(action) <= 6:
                    index = int(action)
                    if index < len(job.sections) and job.status[index] in ("Failed", "Stopped"):
                        yield from run_sections(job, [index], cancelled)
                        return
                    yield generation_update("This section does not need a retry.", job=job, panel=render_sections(job))
                else:
                    yield generation_update("Choose a section to retry.", job=job, panel=render_sections(job))
            finally:
                release_model()

        def stop_generation_handler(request: gr.Request):
            with active_lock:
                if active is None or active[0] != getattr(request, "session_hash", None):
                    return gr.skip(), gr.skip(), gr.skip()
                active[1].set()
                demo_instance.stop_audio_generation()
            return "Stopping generation…", gr.update(visible=False), gr.update(value="Stopping…", interactive=False)

        generation_outputs = [audio_output, complete_audio_output, log_output, streaming_status, generate_btn,
                              stop_btn, generation_feedback, *script_check_outputs, completed_take, section_job, section_panel, generation_outcome]
        generation_event = generate_btn.click(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, inference_steps, seed, disable_voice_cloning, section_mode],
            outputs=generation_outputs,
            js="(...inputs) => { window.vibevoicePlayback?.start(); return inputs; }",
            queue=True, trigger_mode="once", show_progress="hidden", concurrency_id="vibevoice-model", concurrency_limit=1,
        )
        generation_event.failure(
            fn=None, js="() => { window.vibevoicePlayback?.finish({running:false, completed:false}, null); }",
            queue=False,
        )
        if private_features:
            section_mode.change(lambda enabled: gr.update(value="Generate in sections" if enabled else "Generate Audio"),
                                section_mode, generate_btn, queue=False)
            section_event = section_action.click(section_request, [section_command, section_job], generation_outputs,
                                 queue=True, concurrency_id="vibevoice-model", concurrency_limit=1, api_name=False)
            section_event.failure(
                fn=None, js="() => { window.vibevoicePlayback?.finish({running:false, completed:false}, null); }",
                queue=False,
            )

        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, stop_btn],
            js="() => { window.vibevoicePlayback?.stop(); return []; }",
            queue=False, api_name=False,
        )

        example_cycle = gr.State([])
        example_result = gr.Textbox(visible="hidden")

        def load_random_example(remaining):
            import random
            examples = getattr(demo_instance, "example_scripts", None) or [
                [2, "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
            ]
            remaining = [index for index in (remaining or []) if 0 <= index < len(examples)]
            if not remaining:
                remaining = list(range(len(examples)))
                random.shuffle(remaining)
            selected = examples[remaining.pop()]
            return json.dumps({"id": uuid4().hex, "speakers": selected[0], "script": selected[1]}), remaining

        random_example_btn.click(
            fn=load_random_example, inputs=[example_cycle], outputs=[example_result, example_cycle],
            queue=False, trigger_mode="once",
        )
        example_result.change(
            fn=None, inputs=[example_result],
            js="(example) => { window.vibevoiceComposer?.applyExample(example); return []; }",
            queue=False,
        )

    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))

    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/vibevoice-model",
        help="Path to the VibeVoice model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for DDPM (not exposed to users)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the demo on",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint directory containing LoRA adapters (optional)",
    )

    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    legacy_media = os.environ.get("GRADIO_TEMP_DIR") or str(Path(tempfile.gettempdir()) / "gradio")
    media = tempfile.TemporaryDirectory(prefix="vibevoice-media-", ignore_cleanup_errors=True)
    os.environ["GRADIO_TEMP_DIR"] = media.name

    # shortcut: Gradio 5.50 re-copies Windows uploads during playback; remove when upstream fixes it.
    if os.name == "nt" and gr.__version__ == "5.50.0":
        routes.move_uploaded_files_to_cache = cache_uploaded_files

    set_seed(42)  # Set a fixed seed for reproducibility

    print("Initializing VibeVoice Demo with Streaming Support...")

    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps,
        adapter_path=args.checkpoint_path,
        share=args.share,
    )
    demo_instance.media_root = Path(media.name)

    # Create interface
    interface = create_demo_interface(demo_instance)

    print(f"Launching demo on port {args.port}")
    print(f"Model path: {args.model_path}")
    print(f"Available voices: {len(demo_instance.available_voices)}")
    print(f"Streaming mode: ENABLED")
    print(f"Session isolation: ENABLED")

    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=args.share,
            inbrowser=True,
            server_port=args.port,
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            blocked_paths=[
                str((demo_instance.voices_dir / "recorded").resolve()),
                str(Path(legacy_media).resolve()),
                str(Path(".gradio/cached_examples").resolve()),
                str(takes_directory()),
            ] if args.share else [str(takes_directory())],
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Server error: {e}")
        raise
    finally:
        media.cleanup()


if __name__ == "__main__":
    main()
