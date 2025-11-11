"""
LLM-based Command Handler for Decompression Mode

Uses a local LLM to dynamically generate actions, responses, and behaviors
based on user input, gestures, and context. Replaces hardcoded JSON config.
"""

import json
import os
import sys
from typing import Optional, Dict, List, Tuple, Callable
from threading import Lock, Thread
from queue import Queue, Empty
import time
import requests

# Add parent directory for logger
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from logger import get_logger
from response_storage import ResponseStorage


class LLMCommandHandler:
    """Handles commands using a local LLM to generate actions and responses dynamically"""
    
    def __init__(self, state_manager=None, llm_api_url: str = "http://localhost:11434", 
                 model_name: str = "qwen2.5-coder:0.5b", use_llm: bool = True):
        """Initialize LLM command handler
        
        Args:
            state_manager: Optional StateManager instance for status transitions
            llm_api_url: URL for local LLM API (default: Ollama at localhost:11434)
            model_name: Name of the LLM model to use
            use_llm: If False, falls back to simple rule-based matching
        """
        self.state_manager = state_manager
        self.llm_api_url = llm_api_url
        self.model_name = model_name
        self.use_llm = use_llm
        
        self.logger = get_logger("LLM Command Handler")
        
        # Response storage for known commands (CSV-based)
        self.response_storage = ResponseStorage()
        
        # Callbacks for actions
        self.status_change_callback: Optional[Callable[[str], bool]] = None
        self.tts_callback: Optional[Callable[[str], None]] = None
        
        # Pending command tracking for unknown commands
        self.pending_command_text = None
        self.pending_command_key = None
        self.pending_command_callback: Optional[Callable[[Dict, str], None]] = None
        
        # Response cache to avoid repeated LLM calls
        self.response_cache: Dict[str, Dict] = {}
        self.cache_lock = Lock()
        self.cache_ttl = 3600  # Cache for 1 hour
        
        # Async LLM call queue and pending requests
        self.llm_queue = Queue()
        self.pending_requests: Dict[str, Dict] = {}  # cache_key -> {'callback': func, 'result': None}
        self.pending_lock = Lock()
        self.llm_thread = None
        self.running = True
        
        # Start background thread for LLM calls
        self._start_llm_thread()
        
        # Available behaviors/statuses
        self.available_statuses = ['eye', 'people', 'wave', 'smile', 'thumbs_up']
        self.available_gestures = ['wave', 'smile', 'thumbs_up']
        self.tts_output_device_index = None  # Audio output device index (can be set if needed)
        
        # System prompt for LLM
        self.system_prompt = self._build_system_prompt()
        
        # Test LLM connection - fail fast if LLM required but unavailable
        if self.use_llm:
            if not self._test_llm_connection():
                raise RuntimeError(f"LLM not available at {self.llm_api_url} with model '{self.model_name}'. Please ensure Ollama is running and the model is installed.")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines expected behavior and output format"""
        # Shorter, more focused prompt for faster generation
        return """Interactive art installation assistant. Generate JSON responses.

STATUSES: "eye", "people", "wave", "smile", "thumbs_up"
GESTURES: "wave", "smile", "thumbs_up"

VOICE COMMAND JSON:
{"action": {"type": "status"|"tts_only", "status": "eye"|"people"|"wave"|"smile"|"thumbs_up"}, "tts_response": "...", "text_scroller": {"text": "...", "wobble_amount": 0.05, "scroll_time": 8.0} | null}

GESTURE JSON:
{"action": {"type": "status", "status": "wave"|"smile"|"thumbs_up", "substate": "hand_waving"}, "tts_response": "...", "text_scroller": {"text": "...", "wobble_amount": 0.05, "scroll_time": 8.0} | null}

TTS TRIGGER JSON:
{"type": "random"|"time", "interval_min": 30.0, "interval_max": 90.0, "interval": 60.0, "response": "..."}

Style: Burning Man/Decompression themed, warm, playful, concise. Respond with JSON only."""
    
    def _start_llm_thread(self):
        """Start background thread for async LLM calls"""
        self.llm_thread = Thread(target=self._llm_worker, daemon=True)
        self.llm_thread.start()
        self.logger.info("LLM background thread started")
    
    def _llm_worker(self):
        """Background worker thread that processes LLM requests"""
        while self.running:
            try:
                # Get request from queue (with timeout to allow checking self.running)
                try:
                    request = self.llm_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                prompt = request['prompt']
                cache_key = request['cache_key']
                
                # Log start of processing
                if cache_key:
                    self.logger.info(f"[LLM] Starting processing: {cache_key}")
                else:
                    self.logger.info("[LLM] Starting processing: (no cache key)")
                
                start_time = time.time()
                
                # Make the actual LLM call (no timeout - let it take as long as needed)
                result = self._call_llm_sync(prompt, cache_key)
                
                elapsed_time = time.time() - start_time
                
                # Log completion
                if cache_key:
                    if result:
                        self.logger.info(f"[LLM] Finished processing: {cache_key} (took {elapsed_time:.2f}s)")
                    else:
                        self.logger.warning(f"[LLM] Processing failed: {cache_key} (took {elapsed_time:.2f}s)")
                else:
                    if result:
                        self.logger.info(f"[LLM] Finished processing: (no cache key) (took {elapsed_time:.2f}s)")
                    else:
                        self.logger.warning(f"[LLM] Processing failed: (no cache key) (took {elapsed_time:.2f}s)")
                
                # Store result and notify waiting code
                if cache_key:
                    with self.pending_lock:
                        if cache_key in self.pending_requests:
                            self.pending_requests[cache_key]['result'] = result
                            callback = self.pending_requests[cache_key].get('callback')
                            if callback:
                                self.logger.debug(f"[LLM] Executing callback for: {cache_key}")
                                try:
                                    callback(result)
                                except Exception as e:
                                    self.logger.error(f"[LLM] Error in callback for {cache_key}: {e}")
                        else:
                            self.logger.warning(f"[LLM] Cache key {cache_key} not found in pending_requests after processing")
            except Exception as e:
                self.logger.error(f"Error in LLM worker thread: {e}")
                import traceback
                traceback.print_exc()
    
    def _test_llm_connection(self):
        """Test if LLM API is available
        
        Returns:
            True if LLM is available, False otherwise
        """
        response = requests.get(f"{self.llm_api_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            if any(self.model_name in name for name in model_names):
                self.logger.info(f"LLM connection successful. Model '{self.model_name}' available.")
                return True
            else:
                self.logger.error(f"Model '{self.model_name}' not found. Available: {model_names}")
                return False
        else:
            self.logger.error(f"LLM API returned status {response.status_code}")
            return False
    
    def _call_llm(self, prompt: str, cache_key: str = None, callback: Callable = None) -> Optional[Dict]:
        """Call the local LLM API asynchronously (non-blocking)
        
        Args:
            prompt: User prompt
            cache_key: Optional cache key to check/store response
            callback: Optional callback function(result) to call when result is ready
            
        Returns:
            Cached response if available immediately, None if request is queued
        """
        if not self.use_llm:
            return None
        
        # Check cache first - return immediately if cached
        if cache_key:
            with self.cache_lock:
                cached = self.response_cache.get(cache_key)
                if cached and (time.time() - cached['timestamp']) < self.cache_ttl:
                    self.logger.info(f"[LLM] Using cached response for: {cache_key}")
                    return cached['response']
            
            # Check if already pending or processing
            with self.pending_lock:
                if cache_key in self.pending_requests:
                    # Request already pending/processing, just add callback if provided
                    if callback:
                        existing_callback = self.pending_requests[cache_key].get('callback')
                        if existing_callback:
                            # Chain callbacks - call both
                            def chained_callback(result):
                                try:
                                    existing_callback(result)
                                except Exception as e:
                                    self.logger.error(f"[LLM] Error in existing callback: {e}")
                                try:
                                    callback(result)
                                except Exception as e:
                                    self.logger.error(f"[LLM] Error in new callback: {e}")
                            self.pending_requests[cache_key]['callback'] = chained_callback
                        else:
                            self.pending_requests[cache_key]['callback'] = callback
                    self.logger.debug(f"[LLM] Request already pending for: {cache_key}, added callback")
                    return None
        
        # Queue the request for background processing
        with self.pending_lock:
            self.pending_requests[cache_key] = {'callback': callback, 'result': None, 'queued_at': time.time()}
        
        self.llm_queue.put({
            'prompt': prompt,
            'cache_key': cache_key
        })
        
        self.logger.info(f"[LLM] Queued request for: {cache_key}")
        return None  # Request queued, will be processed async
    
    def _call_llm_sync(self, prompt: str, cache_key: str = None) -> Optional[Dict]:
        """Synchronous LLM call (used by background thread)
        
        Args:
            prompt: User prompt
            cache_key: Optional cache key to check/store response
            
        Returns:
            Parsed JSON response or None if failed
        """
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Call Ollama API with optimizations for speed
        # Limit response length and use num_predict to speed up generation
        response = requests.post(
            f"{self.llm_api_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "format": "json",  # Request JSON format
                "options": {
                    "num_predict": 256,  # Limit response to ~256 tokens (enough for JSON response)
                    "temperature": 0.7,  # Lower temperature for more deterministic/faster responses
                    "top_p": 0.9,  # Nucleus sampling for faster generation
                }
            },
            timeout=None  # No timeout - let it take as long as needed
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('message', {}).get('content', '')
            
            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                if '```json' in content:
                    start = content.find('```json') + 7
                    end = content.find('```', start)
                    content = content[start:end].strip()
                elif '```' in content:
                    start = content.find('```') + 3
                    end = content.find('```', start)
                    content = content[start:end].strip()
                
                parsed = json.loads(content)
                
                # Cache the response
                if cache_key:
                    with self.cache_lock:
                        self.response_cache[cache_key] = {
                            'response': parsed,
                            'timestamp': time.time()
                        }
                
                return parsed
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM JSON response: {e}")
                self.logger.error(f"Response content: {content[:200]}")
                return None
        else:
            self.logger.error(f"LLM API returned status {response.status_code}: {response.text}")
            return None
    
    def get_pending_result(self, cache_key: str) -> Optional[Dict]:
        """Check if a pending LLM request has completed
        
        Args:
            cache_key: Cache key for the request
            
        Returns:
            Result if ready, None if still pending or not found
        """
        with self.pending_lock:
            if cache_key in self.pending_requests:
                result = self.pending_requests[cache_key].get('result')
                if result is not None:
                    # Result is ready, remove from pending
                    del self.pending_requests[cache_key]
                    return result
        return None
    
    def process_voice_command(self, text: str) -> Tuple[bool, bool]:
        """Process a voice command - checks CSV first, then LLM if needed
        
        Args:
            text: Input text from speech-to-text
            
        Returns:
            Tuple of (success: bool, is_unknown: bool)
            - success: True if command was processed or queued
            - is_unknown: True if command is unknown and needs LLM processing
        """
        if not text:
            return False, False
        
        self.logger.info(f"[LLM] Voice command received: '{text}'")
        
        # Normalize text for lookup
        text_lower = text.lower().strip()
        
        # Check known commands first (greetings, common commands)
        known_commands = {
            'hello': 'hello',
            'hi': 'hello',
            'hey': 'hello',
            'greetings': 'hello',
            "what's up": 'hello',
            'howdy': 'hello',
            'eye': 'eye',
            'look': 'eye',
            'see': 'eye',
            'watch': 'eye',
            'gaze': 'eye',
            'people': 'people',
            'person': 'people',
            'outline': 'people',
            'silhouette': 'people',
            'crowd': 'people',
        }
        
        # Try to match known command
        command_key = None
        for trigger, key in known_commands.items():
            if trigger in text_lower:
                command_key = key
                break
        
        # Check CSV storage for known command
        if command_key:
            stored_response = self.response_storage.get_response('voice_command', command_key)
            if stored_response:
                self.logger.info(f"[LLM] Using stored response for: '{text}' -> '{command_key}'")
                self._execute_response(stored_response, text)
                return True, False  # Success, not unknown
        
        # Not found in CSV, use LLM (but save result to CSV)
        # This is an unknown command that needs LLM processing
        cache_key = f"voice:{text_lower}"
        
        # Generate prompt for LLM
        prompt = f"""User said: "{text}"

Interpret this voice command and generate an appropriate response.
Consider the context: This is an interactive art installation at a Burning Man/Decompression event.
The user is interacting with the installation through voice.

Generate a response that:
1. Matches the user's intent to an available behavior/status
2. Provides a single TTS response (warm, playful, concise)
3. Includes text scroller config if the action involves visual text display

Respond with JSON matching the voice command format."""
        
        # Store text for later execution (will be set by mode when LLM response is ready)
        # The callback will be set by the mode to handle pending responses
        self.pending_command_text = text
        self.pending_command_key = command_key
        
        # Define callback to handle result when ready (for unknown commands)
        def handle_unknown_result(result):
            if not result:
                self.logger.warning(f"[LLM] No LLM response for: '{text}'")
                if self.pending_command_callback:
                    self.pending_command_callback(None, text)
                return
            
            self.logger.info(f"[LLM] Processing result for unknown command: '{text}'")
            
            # Save to CSV if this looks like a known command type
            if command_key:
                self.response_storage.save_response('voice_command', command_key, result)
                self.logger.info(f"[LLM] Saved new response to CSV for: '{command_key}'")
            
            # Call pending command callback if set
            if self.pending_command_callback:
                self.pending_command_callback(result, text)
        
        # Call LLM async with callback for unknown commands
        result = self._call_llm(prompt, cache_key=cache_key, callback=handle_unknown_result)
        
        # If cached result available immediately, process it
        if result:
            handle_unknown_result(result)
            return True, False  # Cached, not unknown
        
        # Request queued, is unknown
        return True, True
    
    def set_pending_command_callback(self, callback: Callable[[Dict, str], None]):
        """Set callback for when pending unknown command LLM response is ready
        
        Args:
            callback: Function(result_dict, original_text) called when LLM response is ready
        """
        self.pending_command_callback = callback
    
    def _execute_response(self, result: Dict, text: str):
        """Execute a response dict (action + TTS)
        
        Args:
            result: Response dict with action and tts_response
            text: Original command text (for logging)
        """
        # Execute action
        action = result.get('action', {})
        action_type = action.get('type')
        
        success = False
        if action_type == 'status':
            status = action.get('status')
            if status and self.status_change_callback:
                self.logger.info(f"[LLM] Executing status change to: {status}")
                success = self.status_change_callback(status)
                if success:
                    self.logger.info(f"[LLM] Status changed successfully to: {status}")
                else:
                    self.logger.warning(f"[LLM] Status change failed for: {status}")
        elif action_type == 'tts_only':
            success = True
            self.logger.info(f"[LLM] TTS-only action for: '{text}'")
        
        # Speak response if available
        if success or action_type == 'tts_only':
            tts_response = result.get('tts_response')
            if tts_response and self.tts_callback:
                self.logger.info(f"[LLM] Triggering TTS: '{tts_response}'")
                self.tts_callback(tts_response)
            else:
                if not tts_response:
                    self.logger.warning(f"[LLM] No TTS response in result for: '{text}'")
                if not self.tts_callback:
                    self.logger.warning(f"[LLM] No TTS callback set")
        
        self.logger.info(f"[LLM] Finished processing command: '{text}'")
    
    def get_gesture_trigger(self, gesture_name: str) -> Optional[Dict]:
        """Get gesture trigger configuration - checks CSV first, then LLM if needed
        
        Args:
            gesture_name: Name of gesture (e.g., 'wave', 'smile', 'thumbs_up')
            
        Returns:
            Gesture trigger configuration dict if cached/stored, None if queued (will be available later)
        """
        if gesture_name not in self.available_gestures:
            return None
        
        # Check CSV storage first
        stored_response = self.response_storage.get_response('gesture', gesture_name)
        if stored_response:
            self.logger.info(f"[LLM] Using stored gesture response for: {gesture_name}")
            result = stored_response.copy()
            result['gesture'] = gesture_name
            return result
        
        cache_key = f"gesture:{gesture_name}"
        
        # Check cache first
        with self.cache_lock:
            cached = self.response_cache.get(cache_key)
            if cached and (time.time() - cached['timestamp']) < self.cache_ttl:
                result = cached['response'].copy()
                result['gesture'] = gesture_name
                return result
        
        # Check if result is ready from pending request
        result = self.get_pending_result(cache_key)
        if result:
            result['gesture'] = gesture_name
            # Save to CSV for future use
            self.response_storage.save_response('gesture', gesture_name, result)
            self.logger.info(f"[LLM] Saved new gesture response to CSV for: {gesture_name}")
            return result
        
        # Check if already pending (before logging/queuing)
        with self.pending_lock:
            if cache_key in self.pending_requests:
                # Already queued, just return None
                queued_time = self.pending_requests[cache_key].get('queued_at', 0)
                wait_time = time.time() - queued_time
                self.logger.debug(f"[LLM] Gesture config request already pending for: {gesture_name} (waiting {wait_time:.1f}s)")
                return None
        
        # Queue request if not already pending
        self.logger.info(f"[LLM] Queuing gesture config generation for: {gesture_name}")
        
        prompt = f"""A user performed the "{gesture_name}" gesture.

Generate an appropriate response configuration for this gesture.
The response should:
1. Trigger the appropriate status/behavior
2. Include a single TTS response (warm, playful, concise)
3. Include text scroller configuration if text should be displayed

For wave gesture, use substate "hand_waving".
For smile gesture, consider showing text like "i see you smiling" or similar.
For thumbs_up, keep it simple and celebratory.

Respond with JSON matching the gesture trigger format."""
        
        # Define callback to save result to CSV
        def save_to_csv(result):
            if result:
                self.response_storage.save_response('gesture', gesture_name, result)
                self.logger.info(f"[LLM] Saved new gesture response to CSV for: {gesture_name}")
        
        # Queue async request with callback to save to CSV
        self._call_llm(prompt, cache_key=cache_key, callback=save_to_csv)
        
        return None  # Not ready yet, will be available on next call
    
    def get_tts_trigger(self) -> Optional[Dict]:
        """Get TTS trigger configuration using LLM (async, non-blocking)
        
        Returns:
            TTS trigger configuration dict if cached/ready, None if queued
        """
        cache_key = "tts_trigger"
        
        # Check cache first
        with self.cache_lock:
            cached = self.response_cache.get(cache_key)
            if cached and (time.time() - cached['timestamp']) < self.cache_ttl:
                return cached['response']
        
        # Check if result is ready from pending request
        result = self.get_pending_result(cache_key)
        if result:
            return result
        
        # Check if already pending (before logging/queuing)
        with self.pending_lock:
            if cache_key in self.pending_requests:
                # Already queued, just return None
                queued_time = self.pending_requests[cache_key].get('queued_at', 0)
                wait_time = time.time() - queued_time
                self.logger.debug(f"[LLM] TTS trigger request already pending (waiting {wait_time:.1f}s)")
                return None
        
        # Queue request if not already pending
        self.logger.info("[LLM] Queuing TTS trigger configuration generation")
        
        prompt = """Generate a configuration for periodic TTS announcements.

These should be welcoming messages that play randomly every 30-90 seconds.
The message should:
- Welcome people to decompression
- Be warm and inviting
- Reference Burning Man/Decompression themes
- Be concise (short phrase)

Respond with JSON matching the TTS trigger format (type: "random", interval_min: 30.0, interval_max: 90.0, response: "...")."""
        
        # Queue async request
        self._call_llm(prompt, cache_key=cache_key)
        
        return None  # Not ready yet, will be available on next call
    
    def set_status_change_callback(self, callback: Callable[[str], bool]):
        """Set callback for status changes"""
        self.status_change_callback = callback
    
    def set_tts_callback(self, callback: Callable[[str], None]):
        """Set callback for TTS responses"""
        self.tts_callback = callback
    
    def get_gesture_triggers(self) -> List[Dict]:
        """Get all gesture trigger configurations (generated on-demand)"""
        triggers = []
        for gesture in self.available_gestures:
            trigger = self.get_gesture_trigger(gesture)
            if trigger:
                triggers.append(trigger)
        return triggers
    
    def get_tts_triggers(self) -> List[Dict]:
        """Get TTS trigger configurations"""
        trigger = self.get_tts_trigger()
        if trigger:
            return [trigger]
        return []
    
    def reload_config(self):
        """Reload configuration (clear cache to regenerate)"""
        with self.cache_lock:
            self.response_cache.clear()
        self.logger.info("LLM command handler cache cleared - will regenerate responses")

