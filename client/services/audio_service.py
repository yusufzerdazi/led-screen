"""
Audio service for TTS and STT.

Handles text-to-speech (Piper) and speech-to-text (Whisper) functionality.
"""

import os
import sys
import numpy as np
import sounddevice as sd
from threading import Lock
from typing import Optional, Callable
from scipy import signal

# TTS imports
from piper import PiperVoice

# STT imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speech_to_text import SpeechToText
from gpu_utils import get_gpu_detector


class AudioService:
    """Manages TTS and STT functionality."""
    
    def __init__(self, tts_output_device_index=None):
        """Initialize audio service.
        
        Args:
            tts_output_device_index: Optional output device index for TTS playback (None = use default)
        """
        self.tts_voice: Optional[PiperVoice] = None
        self.tts_lock = Lock()
        self.tts_playing = False
        self.tts_output_device_index = tts_output_device_index
        
        self.speech_to_text: Optional[SpeechToText] = None
        self.stt_enabled = False
        self.stt_input_device_index = None
        
        self._voice_command_callback: Optional[Callable[[str], None]] = None
        
        # Set up logging
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from logger import get_logger
        self.logger_tts = get_logger("Audio (TTS)")
        self.logger_stt = get_logger("Audio (STT)")
        
        # List available audio devices
        self._list_audio_devices()
    
    def _list_audio_devices(self):
        """List all available audio output devices"""
        try:
            devices = sd.query_devices()
            self.logger_tts.info("Available audio output devices:")
            default_output = sd.query_devices(kind='output')
            self.logger_tts.info(f"  Default output device: {default_output['name']} (index {default_output['index']})")
            
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    default_marker = " [DEFAULT]" if i == default_output['index'] else ""
                    self.logger_tts.info(f"  [{i}] {device['name']} - {device['max_output_channels']} channels @ {device['default_samplerate']}Hz{default_marker}")
            
            if self.tts_output_device_index is not None:
                try:
                    selected_device = sd.query_devices(self.tts_output_device_index)
                    self.logger_tts.info(f"Using TTS output device [{self.tts_output_device_index}]: {selected_device['name']}")
                except Exception as e:
                    self.logger_tts.warning(f"Invalid TTS output device index {self.tts_output_device_index}: {e}")
                    self.logger_tts.info("Falling back to default output device")
                    self.tts_output_device_index = None
            else:
                self.logger_tts.info(f"Using default TTS output device: {default_output['name']}")
        except Exception as e:
            self.logger_tts.warning(f"Could not list audio devices: {e}")
    
    def _apply_reverb(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply reverb effect to audio
        
        Args:
            audio: Audio array (float32, normalized to -1.0 to 1.0)
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio array with reverb applied
        """
        # Simple reverb using impulse response simulation
        # Create a simple reverb impulse response
        reverb_time = 0.5  # Reverb decay time in seconds
        reverb_decay = 0.3  # Decay factor
        
        # Generate impulse response with multiple delays
        impulse_length = int(reverb_time * sample_rate)
        impulse = np.zeros(impulse_length)
        
        # Create multiple delayed reflections
        delays = [int(sample_rate * 0.03),  # 30ms delay
                 int(sample_rate * 0.05),   # 50ms delay
                 int(sample_rate * 0.08),   # 80ms delay
                 int(sample_rate * 0.12)]   # 120ms delay
        
        gains = [0.4, 0.3, 0.2, 0.15]  # Decreasing gains
        
        for delay, gain in zip(delays, gains):
            if delay < impulse_length:
                impulse[delay] = gain
        
        # Add exponential decay
        decay = np.exp(-np.arange(impulse_length) / (reverb_time * sample_rate * reverb_decay))
        impulse = impulse * decay
        
        # Normalize impulse response
        impulse = impulse / np.max(np.abs(impulse)) * 0.5
        
        # Apply convolution (reverb)
        reverb_audio = signal.convolve(audio, impulse, mode='same')
        
        # Mix original with reverb (70% original, 30% reverb)
        mixed = audio * 0.7 + reverb_audio * 0.3
        
        return mixed
    
    def initialize_tts(self) -> bool:
        """Initialize Piper TTS voice.
        
        Returns:
            True if initialized successfully
        """
        # Search for voice models in common locations
        search_dirs = [
            os.path.expanduser("~/.local/share/piper/voices"),
            os.path.join(os.path.dirname(__file__), "..", "..", "voices"),
        ]
        
        model_path = None
        config_path = None
        
        # Search for .onnx files recursively
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.onnx'):
                        model_path = os.path.join(root, file)
                        # Look for corresponding .json config file
                        config_path = model_path + ".json"
                        if not os.path.exists(config_path):
                            config_path = model_path.replace(".onnx", ".json")
                        if not os.path.exists(config_path):
                            base_name = os.path.splitext(file)[0]
                            for json_file in files:
                                if json_file.endswith('.json') and base_name in json_file:
                                    config_path = os.path.join(root, json_file)
                                    break
                        break
                if model_path:
                    break
            if model_path:
                break
        
        if model_path and os.path.exists(model_path):
            # Try to use GPU acceleration for ONNX Runtime if available
            gpu_detector = get_gpu_detector()
            onnx_providers = gpu_detector.get_onnx_providers()
            
            if config_path and os.path.exists(config_path):
                # PiperVoice.load doesn't directly support providers, but ONNX Runtime
                # will use them automatically if available
                self.tts_voice = PiperVoice.load(model_path, config_path)
                if len(onnx_providers) > 1:  # More than just CPU
                    self.logger_tts.info(f"Piper TTS initialized with GPU acceleration: {onnx_providers[0]}")
                else:
                    self.logger_tts.info(f"Piper TTS initialized with model: {model_path} and config: {config_path}")
            else:
                self.tts_voice = PiperVoice.load(model_path)
                if len(onnx_providers) > 1:
                    self.logger_tts.info(f"Piper TTS initialized with GPU acceleration: {onnx_providers[0]}")
                else:
                    self.logger_tts.info(f"Piper TTS initialized with model: {model_path} (auto-detected config)")
            return True
        else:
            voices_dir = os.path.expanduser("~/.local/share/piper/voices")
            self.logger_tts.warning(f"Piper TTS model not found in {voices_dir}")
            return False
    
    def initialize_stt(self, input_device_index=None, voice_command_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Initialize speech-to-text.
        
        Args:
            input_device_index: Audio input device index
            voice_command_callback: Callback function for voice commands
            
        Returns:
            True if initialized successfully
        """
        self.stt_input_device_index = input_device_index
        self._voice_command_callback = voice_command_callback
        
        self.logger_stt.info("Initializing speech-to-text...")
        self.speech_to_text = SpeechToText(
            model_size="tiny",
            language="en",
            use_whisper=True,
            input_device_index=input_device_index
        )
        
        success = self.speech_to_text.start_listening(on_text_callback=self._handle_voice_command)
        if success:
            self.logger_stt.info("Speech-to-text enabled - listening for voice commands")
            self.stt_enabled = True
            return True
        else:
            self.logger_stt.error("Failed to start speech-to-text")
            self.stt_enabled = False
            self.speech_to_text = None
            return False
    
    def _handle_voice_command(self, text: str) -> None:
        """Handle voice command and forward to callback."""
        if self._voice_command_callback:
            self._voice_command_callback(text)
    
    def speak(self, text: str) -> None:
        """Generate and play TTS audio in a separate thread.
        
        Args:
            text: Text to speak
        """
        if not self.tts_voice:
            self.logger_tts.warning(f"TTS voice not initialized, cannot speak: {text}")
            return
        
        self.logger_tts.info(f"Speaking: {text}")
        
        def play_audio():
            with self.tts_lock:
                if self.tts_playing:
                    self.logger_tts.debug("TTS already playing, skipping")
                    return
                self.tts_playing = True
            
            try:
                self.logger_tts.debug("Starting TTS synthesis")
                # Synthesize speech
                audio_generator = self.tts_voice.synthesize(text)
                
                # Consume the generator to get audio bytes
                audio_chunks = []
                for audio_chunk in audio_generator:
                    if hasattr(audio_chunk, 'audio_int16_bytes'):
                        audio_chunks.append(audio_chunk.audio_int16_bytes)
                    elif hasattr(audio_chunk, 'audio_bytes'):
                        audio_chunks.append(audio_chunk.audio_bytes)
                    elif isinstance(audio_chunk, bytes):
                        audio_chunks.append(audio_chunk)
                    else:
                        audio_chunks.append(bytes(audio_chunk))
                
                # Combine all chunks
                audio_data = b''.join(audio_chunks)
                
                # Get sample rate
                sample_rate = self.tts_voice.config.sample_rate if hasattr(self.tts_voice.config, 'sample_rate') else 22050
                
                self.logger_tts.debug(f"TTS synthesized {len(audio_data)} bytes at {sample_rate}Hz")
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Apply reverb effect
                audio_array = self._apply_reverb(audio_array, sample_rate)
                
                # Convert back to int16
                audio_array = (np.clip(audio_array, -1.0, 1.0) * 32767.0).astype(np.int16)
                
                # Play audio
                self.logger_tts.info(f"Playing TTS audio: {text}")
                try:
                    if self.tts_output_device_index is not None:
                        sd.play(audio_array, samplerate=sample_rate, device=self.tts_output_device_index)
                        self.logger_tts.debug(f"Playing on device {self.tts_output_device_index}")
                    else:
                        sd.play(audio_array, samplerate=sample_rate)
                        self.logger_tts.debug("Playing on default output device")
                    sd.wait()  # Wait until playback is finished
                    self.logger_tts.info(f"Finished playing TTS: {text}")
                except Exception as e:
                    self.logger_tts.error(f"Error playing audio: {e}")
                    # Try with default device if custom device failed
                    if self.tts_output_device_index is not None:
                        self.logger_tts.info("Retrying with default output device")
                        try:
                            sd.play(audio_array, samplerate=sample_rate)
                            sd.wait()
                            self.logger_tts.info(f"Finished playing TTS (default device): {text}")
                        except Exception as e2:
                            self.logger_tts.error(f"Error playing on default device: {e2}")
                    raise
                
            except Exception as e:
                self.logger_tts.error(f"Error in TTS playback: {e}")
                import traceback
                traceback.print_exc()
            finally:
                with self.tts_lock:
                    self.tts_playing = False
        
        # Play in background thread
        import threading
        thread = threading.Thread(target=play_audio, daemon=True)
        thread.start()
    
    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self.speech_to_text:
            self.speech_to_text.stop()
            self.logger_stt.info("Speech-to-text stopped")
            self.speech_to_text = None

