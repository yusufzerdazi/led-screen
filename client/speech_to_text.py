"""
Speech-to-Text module for continuous microphone listening.

Listens to microphone input and transcribes speech to text, logging the results.
Can be integrated into modes to drive state changes based on voice commands.
"""

import pyaudio
import wave
import tempfile
import os
import threading
import time
import numpy as np

# Try to import speech recognition libraries
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: speech_recognition not installed. Install with: pip install SpeechRecognition")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not installed. Install with: pip install openai-whisper")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: pyaudio not available - speech-to-text will be disabled")


class SpeechToText:
    """Continuous speech-to-text recognition from microphone"""
    
    def __init__(self, model_size="base", language="en", use_whisper=True, input_device_index=None):
        """
        Initialize speech-to-text recognizer
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            language: Language code (default: "en")
            use_whisper: If True, use local Whisper model; if False, use speech_recognition with online services
            input_device_index: Optional microphone device index (None = use default)
        """
        self.input_device_index = input_device_index
        self.model_size = model_size
        self.language = language
        self.use_whisper = use_whisper and WHISPER_AVAILABLE
        self.is_listening = False
        self.audio_thread = None
        self.recognition_thread = None
        
        # Audio configuration
        self.CHUNK = 2048  # Increased buffer size to reduce overflow
        if PYAUDIO_AVAILABLE:
            self.FORMAT = pyaudio.paInt16
        else:
            self.FORMAT = None
        self.CHANNELS = 1
        self.RATE = 8000  # Lower sample rate for better performance (Whisper can handle this)
        
        # Initialize recognizer
        if self.use_whisper:
            print(f"[STT] Loading Whisper model: {model_size}")
            try:
                self.whisper_model = whisper.load_model(model_size)
                print(f"[STT] Whisper model loaded successfully")
            except Exception as e:
                print(f"[STT] Error loading Whisper model: {e}")
                print("[STT] Falling back to speech_recognition")
                self.use_whisper = False
        
        if not self.use_whisper:
            if not SPEECH_RECOGNITION_AVAILABLE:
                raise ImportError("Neither whisper nor speech_recognition available. Install one of them.")
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            print("[STT] Using speech_recognition library")
        
        # Audio buffer for continuous listening
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.audio = None
        self.stream = None
        
        # Lock for Whisper model (will be created after model loads)
        self.whisper_lock = None
        
        # Recognition settings
        self.phrase_timeout = 3.0  # Seconds of silence before processing (increased to reduce processing frequency)
        self.min_audio_length = 1.0  # Minimum audio length to process (increased to reduce short clips)
        self.last_audio_time = time.time()
        self.last_silence_check = time.time()
        self.silence_threshold = 500  # Amplitude threshold for silence detection (adjust based on mic sensitivity)
        self.max_audio_length = 5.0  # Maximum audio length to process (prevent very long processing)
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        # Only log non-zero status if it's a real error (not just overflow warnings)
        # Status 2 = paInputOverflow (can happen occasionally, not critical)
        # Status 4 = paInputUnderflow (can happen occasionally, not critical)
        if status and status not in [2, 4]:  # Ignore overflow/underflow warnings
            print(f"[STT] Audio callback status: {status}")
        
        # Check audio level to detect silence
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        audio_level = np.abs(audio_array).mean()
        is_silent = audio_level < self.silence_threshold
        
        current_time = time.time()
        
        # Add audio data to buffer (non-blocking, drop if buffer is full)
        try:
            with self.buffer_lock:
                self.audio_buffer.append(in_data)
                
                # Only update last_audio_time if there's actual sound (not silence)
                if not is_silent:
                    self.last_audio_time = current_time
                
                # Limit buffer size to prevent memory issues (keep last 10 seconds)
                max_chunks = int((10.0 * self.RATE) / self.CHUNK)  # ~10 seconds of audio
                if len(self.audio_buffer) > max_chunks:
                    # Drop oldest chunks if buffer is too large
                    self.audio_buffer = self.audio_buffer[-max_chunks:]
        except:
            # If buffer lock fails, just continue (don't block audio callback)
            pass
        
        return (None, pyaudio.paContinue)
    
    def _audio_capture_loop(self):
        """Background thread for capturing audio"""
        print("[STT] Audio capture started")
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Use provided device index or default to device 0
            input_device_index = self.input_device_index if self.input_device_index is not None else 0
            
            try:
                device_info = self.audio.get_device_info_by_index(input_device_index)
                print(f"[STT] Using microphone device {input_device_index}: {device_info['name']}")
                
                # Get device's default sample rate or use our preferred rate
                device_sample_rate = int(device_info.get('defaultSampleRate', self.RATE))
                
                # Try to use device's default sample rate, or fall back to common rates
                sample_rates_to_try = [
                    device_sample_rate,  # Device's default
                    44100,  # Common rate
                    48000,  # Common rate
                    16000,  # Our preferred (for Whisper)
                    22050,  # Common rate
                    32000,  # Common rate
                ]
                
                # Remove duplicates while preserving order
                sample_rates_to_try = list(dict.fromkeys(sample_rates_to_try))
                
                actual_rate = None
                for rate in sample_rates_to_try:
                    try:
                        # Test if this rate is supported by opening a test stream
                        test_stream = self.audio.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=rate,
                            input=True,
                            frames_per_buffer=self.CHUNK,
                            input_device_index=input_device_index
                        )
                        test_stream.stop_stream()
                        test_stream.close()
                        actual_rate = rate
                        print(f"[STT] Using sample rate: {actual_rate}Hz")
                        break
                    except:
                        continue
                
                if actual_rate is None:
                    raise Exception("No supported sample rate found for device")
                
                # Update rate for this device
                self.RATE = actual_rate
                
            except Exception as e:
                print(f"[STT] Error with device {input_device_index}: {e}")
                print("[STT] Falling back to default device...")
                try:
                    default_info = self.audio.get_default_input_device_info()
                    input_device_index = default_info['index']
                    actual_rate = int(default_info.get('defaultSampleRate', self.RATE))
                    self.RATE = actual_rate
                    print(f"[STT] Using default device: {default_info['name']} (index {input_device_index}, rate {actual_rate}Hz)")
                except:
                    input_device_index = 0
                    print("[STT] Using device index 0 with default rate")
            
            # Open audio stream with the determined rate
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._audio_callback,
                input_device_index=input_device_index
            )
            
            self.stream.start_stream()
            print(f"[STT] Audio stream started on device {input_device_index} at {self.RATE}Hz")
            
            # Keep thread alive
            while self.is_listening and self.stream.is_active():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"[STT] Error in audio capture: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
            print("[STT] Audio capture stopped")
    
    def _process_audio_with_whisper(self, audio_data):
        """Process audio data using Whisper model"""
        try:
            # Convert audio buffer to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Validate audio data
            if len(audio_np) == 0:
                return None
            
            # Check for invalid values
            if np.any(np.isnan(audio_np)) or np.any(np.isinf(audio_np)):
                return None
            
            # Check if audio is all zeros (silence)
            if np.all(audio_np == 0):
                return None
            
            # Check audio level
            audio_level = np.abs(audio_np).mean()
            if audio_level < 0.001:  # Very quiet, likely noise
                return None
            
            # Whisper expects 16kHz, so resample if needed
            if self.RATE != 16000:
                try:
                    import librosa
                    audio_np = librosa.resample(audio_np, orig_sr=self.RATE, target_sr=16000)
                except ImportError:
                    # librosa not available, try with original rate
                    pass
                except Exception as e:
                    # Resampling failed, try with original rate
                    pass
            
            # Ensure audio is not too short (Whisper needs at least ~0.5s)
            if len(audio_np) < 8000:  # Less than 0.5s at 16kHz
                return None
            
            # Transcribe using Whisper (with lock for thread safety)
            if self.whisper_lock:
                with self.whisper_lock:
                    result = self.whisper_model.transcribe(
                        audio_np,
                        language=self.language,
                        task="transcribe",
                        fp16=False  # Use FP32 to avoid NaN issues
                    )
            else:
                result = self.whisper_model.transcribe(
                    audio_np,
                    language=self.language,
                    task="transcribe",
                    fp16=False  # Use FP32 to avoid NaN issues
                )
            
            text = result["text"].strip()
            
            if text:
                print(f"[STT] Transcribed: {text}")
                return text
            return None
            
        except Exception as e:
            print(f"[STT] Error processing audio with Whisper: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_audio_with_sr(self, audio_data):
        """Process audio data using speech_recognition"""
        try:
            # Create AudioData object
            sample_size = 2  # 16-bit = 2 bytes
            if PYAUDIO_AVAILABLE:
                sample_size = pyaudio.get_sample_size(self.FORMAT)
            
            audio_source = sr.AudioData(
                audio_data,
                self.RATE,
                sample_size
            )
            
            # Recognize speech (using Google's free API as fallback)
            # Note: This requires internet connection
            text = self.recognizer.recognize_google(audio_source, language=self.language)
            return text.strip()
            
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[STT] Error with speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"[STT] Error processing audio with speech_recognition: {e}")
            return None
    
    def _recognition_loop(self):
        """Background thread for processing audio and recognizing speech"""
        print("[STT] Recognition loop started")
        
        last_debug_time = time.time()
        debug_interval = 5.0  # Print debug info every 5 seconds
        
        while self.is_listening:
            try:
                # Check if we have enough audio and enough silence
                current_time = time.time()
                time_since_audio = current_time - self.last_audio_time
                
                with self.buffer_lock:
                    buffer_size = len(self.audio_buffer)
                
                # Debug logging periodically (disabled by default - uncomment if needed)
                # if current_time - last_debug_time >= debug_interval:
                #     audio_level = 0
                #     if buffer_size > 0:
                #         with self.buffer_lock:
                #             if len(self.audio_buffer) > 0:
                #                 recent_chunk = self.audio_buffer[-1]
                #                 audio_array = np.frombuffer(recent_chunk, dtype=np.int16)
                #                 audio_level = np.abs(audio_array).mean()
                #     print(f"[STT] Debug: buffer_size={buffer_size}, time_since_audio={time_since_audio:.1f}s, audio_level={audio_level:.0f}")
                #     last_debug_time = current_time
                
                if buffer_size == 0:
                    time.sleep(0.1)
                    continue
                
                # If we have audio and enough silence, process it
                if time_since_audio >= self.phrase_timeout and buffer_size > 0:
                    # Get audio from buffer
                    with self.buffer_lock:
                        audio_chunks = self.audio_buffer[:]
                        self.audio_buffer = []
                    
                    # Combine audio chunks
                    audio_data = b''.join(audio_chunks)
                    
                    # Check minimum and maximum length
                    audio_duration = len(audio_data) / (self.RATE * 2)  # 2 bytes per sample
                    
                    if audio_duration < self.min_audio_length:
                        continue
                    
                    # Limit maximum length to prevent long processing times
                    if audio_duration > self.max_audio_length:
                        # Trim to last N seconds
                        max_bytes = int(self.max_audio_length * self.RATE * 2)
                        audio_data = audio_data[-max_bytes:]
                    
                    # Process audio in a separate thread to avoid blocking
                    # Process in thread to avoid blocking recognition loop
                    def process_async():
                        try:
                            if self.use_whisper:
                                text = self._process_audio_with_whisper(audio_data)
                            else:
                                text = self._process_audio_with_sr(audio_data)
                            
                            # Log transcribed text and call callback
                            if text:
                                # Call callback if set
                                if hasattr(self, 'on_text_callback') and self.on_text_callback:
                                    try:
                                        self.on_text_callback(text)
                                    except Exception as e:
                                        print(f"[STT] Error in callback: {e}")
                                        import traceback
                                        traceback.print_exc()
                        except Exception as e:
                            print(f"[STT] Error processing audio: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Start processing in background thread
                    process_thread = threading.Thread(target=process_async, daemon=True)
                    process_thread.start()
                    
                    # Continue immediately without waiting for processing
                    continue
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[STT] Error in recognition loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
    
    def start_listening(self, on_text_callback=None):
        """
        Start listening to microphone and transcribing speech
        
        Args:
            on_text_callback: Optional callback function(text) called when text is transcribed
        """
        if not PYAUDIO_AVAILABLE:
            print("[STT] PyAudio not available, cannot start listening")
            return False
        
        if self.is_listening:
            print("[STT] Already listening")
            return True
        
        print("[STT] Starting speech-to-text...")
        print(f"[STT] Using {'Whisper' if self.use_whisper else 'speech_recognition'}")
        print(f"[STT] Model: {self.model_size if self.use_whisper else 'N/A'}")
        print(f"[STT] Sample rate: {self.RATE}Hz, Channels: {self.CHANNELS}")
        
        self.on_text_callback = on_text_callback
        self.is_listening = True
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.audio_thread.start()
        
        # Give audio thread a moment to start
        time.sleep(0.5)
        
        # Start recognition thread
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.recognition_thread.start()
        
        print("[STT] Speech-to-text listening started - speak into microphone")
        return True
    
    def stop_listening(self):
        """Stop listening to microphone"""
        if not self.is_listening:
            return
        
        print("[STT] Stopping speech-to-text...")
        self.is_listening = False
        
        # Wait for threads to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
        
        print("[STT] Speech-to-text stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()


if __name__ == "__main__":
    # Test the speech-to-text module
    print("Testing Speech-to-Text module...")
    print("Speak into the microphone. Say 'quit' to exit.")
    
    stt = SpeechToText(model_size="base", use_whisper=True)
    
    def on_text(text):
        print(f"[CALLBACK] Received text: {text}")
        if "quit" in text.lower():
            print("Quit command detected, stopping...")
            stt.stop_listening()
    
    try:
        stt.start_listening(on_text_callback=on_text)
        
        # Keep main thread alive
        while stt.is_listening:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stt.cleanup()

