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

# TTS imports
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# STT imports
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from speech_to_text import SpeechToText
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False


class AudioService:
    """Manages TTS and STT functionality."""
    
    def __init__(self):
        """Initialize audio service."""
        self.tts_voice: Optional[PiperVoice] = None
        self.tts_lock = Lock()
        self.tts_playing = False
        
        self.speech_to_text: Optional[SpeechToText] = None
        self.stt_enabled = False
        self.stt_input_device_index = None
        
        self._voice_command_callback: Optional[Callable[[str], None]] = None
        
        # Set up logging
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from logger import get_logger
            self.logger_tts = get_logger("Audio (TTS)")
            self.logger_stt = get_logger("Audio (STT)")
        except ImportError:
            # Logger not available, use print fallback
            self.logger_tts = None
            self.logger_stt = None
    
    def initialize_tts(self) -> bool:
        """Initialize Piper TTS voice.
        
        Returns:
            True if initialized successfully
        """
        if not PIPER_AVAILABLE:
            if self.logger_tts:
                self.logger_tts.warning("Piper TTS not available")
            else:
                print("Warning: Piper TTS not available")
            return False
        
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
            try:
                if config_path and os.path.exists(config_path):
                    self.tts_voice = PiperVoice.load(model_path, config_path)
                    if self.logger_tts:
                        self.logger_tts.info(f"Piper TTS initialized with model: {model_path} and config: {config_path}")
                    else:
                        print(f"Piper TTS initialized with model: {model_path} and config: {config_path}")
                else:
                    self.tts_voice = PiperVoice.load(model_path)
                    if self.logger_tts:
                        self.logger_tts.info(f"Piper TTS initialized with model: {model_path} (auto-detected config)")
                    else:
                        print(f"Piper TTS initialized with model: {model_path} (auto-detected config)")
                return True
            except Exception as e:
                if self.logger_tts:
                    self.logger_tts.error(f"Failed to load Piper TTS model: {e}")
                else:
                    print(f"Failed to load Piper TTS model: {e}")
                return False
        else:
            voices_dir = os.path.expanduser("~/.local/share/piper/voices")
            if self.logger_tts:
                self.logger_tts.warning(f"Piper TTS model not found in {voices_dir}")
            else:
                print(f"Piper TTS model not found in {voices_dir}")
            return False
    
    def initialize_stt(self, input_device_index=None, voice_command_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Initialize speech-to-text.
        
        Args:
            input_device_index: Audio input device index
            voice_command_callback: Callback function for voice commands
            
        Returns:
            True if initialized successfully
        """
        if not STT_AVAILABLE:
            if self.logger_stt:
                self.logger_stt.warning("speech_to_text not available - voice commands will be disabled")
            else:
                print("Warning: speech_to_text not available - voice commands will be disabled")
            return False
        
        self.stt_input_device_index = input_device_index
        self._voice_command_callback = voice_command_callback
        
        try:
            if self.logger_stt:
                self.logger_stt.info("Initializing speech-to-text...")
            else:
                print("[STT] Initializing speech-to-text...")
            self.speech_to_text = SpeechToText(
                model_size="tiny",
                language="en",
                use_whisper=True,
                input_device_index=input_device_index
            )
            
            success = self.speech_to_text.start_listening(on_text_callback=self._handle_voice_command)
            if success:
                if self.logger_stt:
                    self.logger_stt.info("Speech-to-text enabled - listening for voice commands")
                else:
                    print("[STT] Speech-to-text enabled - listening for voice commands")
                self.stt_enabled = True
                return True
            else:
                if self.logger_stt:
                    self.logger_stt.error("Failed to start speech-to-text")
                else:
                    print("[STT] Failed to start speech-to-text")
                self.stt_enabled = False
                self.speech_to_text = None
                return False
        except Exception as e:
            if self.logger_stt:
                self.logger_stt.error(f"Error initializing speech-to-text: {e}")
            else:
                print(f"[STT] Error initializing speech-to-text: {e}")
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
            return
        
        def play_audio():
            try:
                with self.tts_lock:
                    if self.tts_playing:
                        return
                    self.tts_playing = True
                
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
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Play audio
                sd.play(audio_array, samplerate=sample_rate)
                sd.wait()  # Wait until playback is finished
                
            except Exception as e:
                if self.logger_tts:
                    self.logger_tts.error(f"Error playing TTS audio: {e}")
                else:
                    print(f"Error playing TTS audio: {e}")
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
            try:
                self.speech_to_text.stop()
                if self.logger_stt:
                    self.logger_stt.info("Speech-to-text stopped")
                else:
                    print("[STT] Speech-to-text stopped")
            except Exception as e:
                if self.logger_stt:
                    self.logger_stt.error(f"Error stopping speech-to-text: {e}")
                else:
                    print(f"[STT] Error stopping speech-to-text: {e}")
            self.speech_to_text = None

