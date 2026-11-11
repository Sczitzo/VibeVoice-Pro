"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import filecmp
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
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

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


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


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5, adapter_path: Optional[str] = None):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
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
        if recorded_dir.is_dir():
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
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
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

    def save_recorded_voice(self, audio, voice_name: str):
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
                if sf.info(audio).format != "WAV":
                    raise ValueError("The uploaded recording must contain WAV audio.")
                samples, sample_rate = sf.read(audio, dtype="int16")
                audio = (sample_rate, samples)
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
                                 disable_voice_cloning: bool = False) -> Iterator[tuple]:
        try:
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            self.generation_error = None
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
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
                out_path = os.path.join(tempfile.gettempdir(), basename)
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
            if self.stop_generation:
                self.is_generating = False
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples when voice cloning is enabled
            voice_samples = None
            if voice_cloning_enabled:
                voice_samples = []
                for speaker_name in selected_speakers:
                    audio_path = self.available_voices[speaker_name]
                    audio_data = self.read_audio(audio_path)
                    if len(audio_data) == 0:
                        self.is_generating = False
                        raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                    voice_samples.append(audio_data)
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script_lines = merge_speaker_turns(formatted_script_lines)
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
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
                args=(inputs, cfg_scale, audio_streamer, voice_cloning_enabled, resolved_inference_steps, resolved_seed, target_device)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
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
            min_chunk_size = sample_rate * 30 # At least 2 seconds of audio
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
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
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
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
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(
        self,
        inputs,
        cfg_scale,
        audio_streamer,
        voice_cloning_enabled: bool,
        inference_steps: int,
        seed: Optional[int],
        target_device: str,
    ):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return

            # Apply per-run DDPM steps
            try:
                self.model.set_ddpm_inference_steps(num_steps=int(inference_steps))
            except Exception as e:
                print(f"Warning: failed to set inference steps ({inference_steps}): {e}")
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation

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
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
                is_prefill=voice_cloning_enabled,
            )
            
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
        print("🛑 Audio generation stop requested")
    
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
    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    demo_dir = Path(__file__).parent
    recorder_js = (demo_dir / "voice_recorder.js").read_text(encoding="utf-8")
    css = (demo_dir / "eucalyptus.css").read_text(encoding="utf-8")
    theme = gr.themes.Base(primary_hue=gr.themes.colors.teal, font=["Inter", "Segoe UI", "sans-serif"])
    with gr.Blocks(title="VibeVoice", theme=theme, css=css, js="() => {" + recorder_js + "\ninitVoiceRecorder();}") as interface:
        with gr.Column(elem_id="guided-composer"):
            gr.HTML("""
                <header class="composer-header">
                    <div class="composer-brand">
                        <span class="composer-brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
                        <h1>VibeVoice</h1>
                    </div>
                    <button type="button" id="open-voice-recorder" class="composer-button"
                            aria-haspopup="dialog">Create Voice</button>
                </header>
            """, padding=False)

            with gr.Row(elem_id="composer-grid"):
                with gr.Column(elem_id="composer-script"):
                    with gr.Row(elem_id="script-heading"):
                        gr.Markdown("## Script", elem_id="composer-script-label")
                        random_example_btn = gr.Button("Random Example", elem_id="random-example")
                    gr.HTML('<p id="composer-script-help">Start each turn with a speaker label, like <strong>Speaker 1:</strong></p>', padding=False)
                    script_input = gr.Textbox(
                        label="Script", show_label=False, elem_id="composer-editor",
                        placeholder="Speaker 1: Welcome to our podcast…",
                        lines=10, max_lines=16, container=False,
                    )

                with gr.Column(elem_id="composer-speakers"):
                    gr.HTML("<h2>Speakers</h2>", elem_id="speaker-heading", padding=False)
                    num_speakers = gr.Radio([1, 2, 3, 4], value=2, label="Number of Speakers",
                                            elem_id="speaker-count")
                    available_speaker_names = list(demo_instance.available_voices)
                    default_speakers = ["en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman"]
                    speaker_selections = []
                    for index, default_voice in enumerate(default_speakers):
                        fallback = available_speaker_names[index % len(available_speaker_names)] if available_speaker_names else None
                        speaker_selections.append(gr.Dropdown(
                            choices=available_speaker_names,
                            value=default_voice if default_voice in available_speaker_names else fallback,
                            label=f"Speaker {index + 1}", visible=index < 2, elem_classes="speaker-item",
                        ))

                    with gr.Accordion("Advanced Settings", open=False):
                        cfg_scale = gr.Slider(1.0, 12.0, value=1.3, step=0.05, label="Guidance Strength")
                        inference_steps = gr.Slider(1, 50, value=int(getattr(demo_instance, "inference_steps", 10)),
                                                    step=1, label="Inference Steps")
                        seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
                        disable_voice_cloning = gr.Checkbox(
                            value=False, label="Disable voice cloning",
                            info="Generate voices without using the selected voice recordings.",
                        )

            with gr.Row(elem_id="composer-actions"):
                streaming_status = gr.Markdown("Generating audio…", visible=False, elem_id="streaming-status")
                generate_btn = gr.Button("Generate Audio", variant="primary", elem_id="generate-audio")
                stop_btn = gr.Button("Stop Generation", visible=False, elem_id="stop-generation")
            generation_feedback = gr.Markdown(visible=False, elem_id="generation-feedback")

            with gr.Column(elem_id="composer-output"):
                gr.Markdown("## Generated Audio", elem_id="composer-output-label")
                audio_output = gr.Audio(label="Live Audio", type="numpy", streaming=True, autoplay=True,
                                        show_download_button=False, elem_id="live-generated-audio")
                complete_audio_output = gr.File(type="filepath", interactive=False, visible="hidden")
                gr.HTML("""
                    <div id="generated-audio-player" hidden>
                        <audio controls controlslist="nodownload" preload="metadata" aria-label="Generated audio"></audio>
                        <a class="composer-button" download>Download WAV</a>
                    </div>
                """, padding=False)
                with gr.Accordion("Technical Details", open=False):
                    log_output = gr.Textbox(label="Generation Log", lines=8, max_lines=15, interactive=False)

            with gr.Accordion("About & Limitations", open=False, elem_id="composer-footer"):
                gr.Markdown("""
VibeVoice can produce unexpected, biased, or inaccurate speech. Check your output before sharing.
Use recordings you have permission to use. Do not use synthetic speech to impersonate, deceive, or commit fraud.
Disclose AI-generated audio when you share it.
                """)

        gr.HTML("""
            <dialog id="voice-recorder" aria-labelledby="voice-recorder-title">
                <header class="recorder-header">
                    <h2 id="voice-recorder-title">Create Voice</h2>
                    <button type="button" data-close>Close</button>
                </header>
                <ol class="recorder-steps" aria-label="Recording steps">
                    <li data-step="microphone" aria-current="step">Microphone</li>
                    <li data-step="record">Record</li>
                    <li data-step="listen">Listen</li>
                    <li data-step="save">Save</li>
                </ol>
                <section class="recorder-reading" aria-labelledby="voice-reading-title">
                    <h3 id="voice-reading-title" tabindex="-1">Read Aloud</h3>
                    <p class="recorder-help">Read the whole passage in your usual speaking voice.</p>
                    <p class="recorder-passage">
                        Twinkle, twinkle, little star.<br>
                        Jack and Jill went up the hill.
                    </p>
                    <p class="recorder-passage">
                        On a warm June day, a boy in a beige jacket took a blue pail of water outside and
                        picked three fresh peaches. “Would you like to go with me?” he asked.
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
                    <p class="recorder-help recorder-next">Stop to listen to your take.</p>
                </section>
                <section data-pane="listen" hidden>
                    <h3>Listen to Your Recording</h3>
                    <audio controls hidden preload="metadata" aria-label="Your voice recording"></audio>
                    <div class="recorder-actions">
                        <button type="button" data-retake>Record Again</button>
                        <button type="button" data-name class="primary">Name Voice</button>
                    </div>
                </section>
                <section data-pane="save" hidden>
                    <div data-save-controls></div>
                    <button type="button" data-back>Back to Listen</button>
                </section>
                <p data-recording-status role="status"></p>
            </dialog>
        """, elem_id="composer-recorder-host", padding=False)
        with gr.Column(elem_id="voice-save-panel"):
            voice_recording = gr.File(type="filepath", file_types=[".wav"], visible=False)
            voice_name = gr.Textbox(label="Voice Name", placeholder="My voice…", elem_id="voice-name")
            save_voice_btn = gr.Button("Save Voice", variant="primary", elem_id="save-voice-btn")
            voice_save_status = gr.Markdown(elem_id="voice-save-status")
            saved_voice_file = gr.File(label="Saved WAV", interactive=False, visible=False, elem_id="saved-voice-download")
            voice_saved = gr.Checkbox(value=False, visible=False)

        def save_recording(audio, name):
            unchanged = [gr.skip() for _ in speaker_selections]
            hidden = gr.update(value=None, visible=False)
            if audio is None:
                return unchanged + [
                    "Voice not saved. Check the recording message and try Save Voice again.", hidden, False
                ]
            try:
                saved_name, saved_path = demo_instance.save_recorded_voice(audio, name)
            except gr.Error as error:
                return unchanged + [error.message, hidden, False]
            choices = list(demo_instance.available_voices)
            updates = [gr.update(choices=choices) for _ in speaker_selections]
            return updates + [
                f"Saved **{saved_name}.wav**. Choose it from any speaker dropdown.",
                gr.update(value=str(saved_path), visible=True), True
            ]

        save_event = save_voice_btn.click(
            fn=save_recording,
            inputs=[voice_recording, voice_name],
            outputs=speaker_selections + [voice_save_status, saved_voice_file, voice_saved],
            js="async (audio, name) => await window.vibevoiceRecorder.prepareSave([audio, name])",
            queue=False,
        )
        save_event.success(
            fn=None, inputs=[voice_saved], js="(saved) => window.vibevoiceRecorder.finishSave(saved)",
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

        complete_audio_output.change(
            fn=None,
            inputs=[complete_audio_output],
            js="""(file) => {
                const player = document.getElementById("generated-audio-player");
                const audio = player.querySelector("audio");
                const download = player.querySelector("a");
                player.hidden = !file?.url;
                if (file?.url) {
                    audio.src = file.url;
                    download.href = file.url;
                    download.download = file.orig_name || "generated-audio.wav";
                } else {
                    audio.removeAttribute("src");
                    audio.load();
                    download.removeAttribute("href");
                }
                return [];
            }""",
            queue=False,
        )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(num_speakers, script, speaker_1, speaker_2, speaker_3, speaker_4, cfg_scale, inference_steps, seed, disable_voice_cloning):
            """Wrapper function to handle the streaming generation call."""
            try:
                speakers = [speaker_1, speaker_2, speaker_3, speaker_4]

                # Clear outputs and reset visibility at start
                hidden = gr.update(value=None, visible=False)
                yield None, gr.update(value=None), "Starting generation…", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), hidden

                # The generator will yield multiple times
                final_log = "Starting generation…"
                completed = False

                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps,
                    seed=seed,
                    disable_voice_cloning=disable_voice_cloning
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        completed = True
                        yield None, gr.update(value=complete_audio), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.skip()
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.skip(), log, streaming_visible, gr.update(visible=False), gr.update(visible=True), gr.skip()
                        else:
                            # No new audio, just update status
                            yield None, gr.skip(), log, streaming_visible, gr.update(visible=False), gr.update(visible=True), gr.skip()

                feedback = hidden
                if not completed and not demo_instance.stop_generation and "Generation stopped" not in final_log:
                    message = "Please provide a script." if not (script or "").strip() else "Audio wasn’t generated. Check Technical Details and try again."
                    feedback = gr.update(value=message, visible=True)
                yield (
                    gr.skip(), gr.skip(), final_log, gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=False), feedback
                )

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value="Audio wasn’t generated. Check Technical Details and try again.", visible=True)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(  # Immediate UI update to hide Generate, show Stop (non-queued)
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[generate_btn, stop_btn],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, inference_steps, seed, disable_voice_cloning],
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn, generation_feedback],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example(replace):
            if replace != "replace":
                return gr.skip(), gr.skip()
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[script_input],
            outputs=[num_speakers, script_input],
            js="""(script) => [!(script ?? "").trim() || window.confirm("Replace your script with a random example?") ? "replace" : ""]""",
            queue=False
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
        default=7860,
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

    # shortcut: Gradio 5.50 re-copies Windows uploads during playback; remove when upstream fixes it.
    if os.name == "nt" and gr.__version__ == "5.50.0":
        routes.move_uploaded_files_to_cache = cache_uploaded_files
    
    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")
    
    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps,
        adapter_path=args.checkpoint_path,
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Model path: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=args.share,
            inbrowser=True,
            # server_port=args.port,
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
