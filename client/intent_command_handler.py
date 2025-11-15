"""
Intent-based Command Handler for Decompression Mode

Uses a lightweight intent classification model to map user input to predefined intents,
then looks up actions/responses from CSV. Much faster than full LLM generation.
"""

import json
import os
import sys
import csv
import time
from typing import Optional, Dict, List, Tuple, Callable
from threading import Lock
import random

# Add parent directory for logger
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from logger import get_logger

# Try to import transformers for intent classification
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")


class IntentCommandHandler:
    """Handles commands using intent classification + CSV lookup"""
    
    def __init__(self, intent_csv_path: str = "client/intent_mappings.csv", 
                 use_intent_model: bool = True):
        """Initialize intent command handler
        
        Args:
            intent_csv_path: Path to CSV file with intent mappings
            use_intent_model: If True, use ML model for classification (required). Model must be available.
        """
        self.intent_csv_path = intent_csv_path
        self.use_intent_model = use_intent_model and TRANSFORMERS_AVAILABLE
        
        self.logger = get_logger("Intent Command Handler")
        
        # Callbacks for actions
        self.status_change_callback: Optional[Callable[[str], bool]] = None
        self.tts_callback: Optional[Callable[[str], None]] = None
        self.text_scroller_callback: Optional[Callable[[str, float], None]] = None
        
        # Available behaviors/statuses
        self.available_statuses = ['eye', 'people', 'wave', 'smile', 'thumbs_up', 'peace']
        self.available_gestures = ['wave', 'smile', 'thumbs_up', 'peace']
        self.tts_output_device_index = None
        
        # Intent classification model (lazy load)
        self._classifier = None
        self._intent_labels = []
        
        # Load intent mappings from CSV
        self.intent_mappings: Dict[str, List[Dict]] = {}
        self._load_intent_mappings()
        
        # Preload model if available (to avoid dynamic installation)
        if self.use_intent_model:
            try:
                self.logger.info("Preloading intent classification model...")
                self.preload_model()
                self.logger.info("Intent classification model preloaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not preload model (will load on first use): {e}")
    
    def _load_intent_mappings(self):
        """Load intent mappings from CSV file with pipe-separated values"""
        original_path = self.intent_csv_path
        
        # Resolve path to absolute if relative
        if not os.path.isabs(self.intent_csv_path):
            # Try relative to current working directory first
            if os.path.exists(self.intent_csv_path):
                # Path exists as-is, use it
                pass
            else:
                # Try relative to this file's location (client/intent_command_handler.py)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                alt_path = os.path.join(current_dir, self.intent_csv_path)
                if os.path.exists(alt_path):
                    self.intent_csv_path = alt_path
                else:
                    # Try one level up (client/intent_mappings.csv -> client/intent_mappings.csv)
                    # If path contains "client/", try without the client/ prefix
                    if "client/" in self.intent_csv_path:
                        alt_path2 = os.path.join(current_dir, self.intent_csv_path.replace("client/", ""))
                        if os.path.exists(alt_path2):
                            self.intent_csv_path = alt_path2
                        else:
                            # Try absolute path from current_dir
                            alt_path3 = os.path.join(current_dir, "..", "intent_mappings.csv")
                            alt_path3 = os.path.normpath(alt_path3)
                            if os.path.exists(alt_path3):
                                self.intent_csv_path = alt_path3
                    else:
                        # Try absolute path from current_dir
                        alt_path2 = os.path.join(current_dir, "..", "intent_mappings.csv")
                        alt_path2 = os.path.normpath(alt_path2)
                        if os.path.exists(alt_path2):
                            self.intent_csv_path = alt_path2
        
        if not os.path.exists(self.intent_csv_path):
            self.logger.error(f"Intent mappings CSV not found!")
            self.logger.error(f"  Original path: {original_path}")
            self.logger.error(f"  Resolved path: {self.intent_csv_path}")
            self.logger.error(f"  Current working directory: {os.getcwd()}")
            self.logger.error(f"  Handler file location: {os.path.dirname(os.path.abspath(__file__))}")
            self.logger.warning("Creating example CSV file (will only have 3 intents!)...")
            self.logger.warning("This means only 3 intents will be available. Fix the CSV path!")
            self._create_example_csv()
            return
        
        self.logger.info(f"Loading intent mappings from: {self.intent_csv_path}")
        
        self.intent_mappings = {}
        
        try:
            with open(self.intent_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip empty rows
                    if not row:
                        continue
                    
                    # Handle None values safely - convert to empty string before calling strip()
                    intent = (row.get('intent') or '').strip()
                    if not intent:
                        continue  # Skip rows without intent
                    
                    if intent not in self.intent_mappings:
                        self.intent_mappings[intent] = []
                    
                    # Parse pipe-separated values - handle None values safely
                    tts_responses_str = (row.get('tts_responses') or '').strip()
                    tts_responses = [r.strip() for r in tts_responses_str.split('|') if r.strip()] if tts_responses_str else []
                    
                    text_scroller_text_str = (row.get('text_scroller_text') or '').strip()
                    text_scroller_texts = [t.strip() for t in text_scroller_text_str.split('|') if t.strip()] if text_scroller_text_str else []
                    
                    # Get text scroller config (single values) - handle empty strings and None
                    # scroll_time is calculated dynamically from text length, not stored in CSV
                    text_scroller_wobble_str = (row.get('text_scroller_wobble') or '').strip()
                    # Try to parse wobble value, but handle cases where text might have leaked into this column
                    text_scroller_wobble = None
                    if text_scroller_wobble_str:
                        try:
                            # Extract first numeric value if there's text mixed in
                            import re
                            match = re.search(r'(\d+\.?\d*)', text_scroller_wobble_str)
                            if match:
                                text_scroller_wobble = float(match.group(1))
                            else:
                                text_scroller_wobble = float(text_scroller_wobble_str)
                        except (ValueError, AttributeError):
                            # If parsing fails, default to 0.0
                            self.logger.warning(f"Could not parse text_scroller_wobble '{text_scroller_wobble_str}' for intent '{intent}', using default 0.0")
                            text_scroller_wobble = 0.0
                    
                    # Get action config - handle None values safely
                    action_type = (row.get('action_type') or 'tts_only').strip() or 'tts_only'
                    action_status = (row.get('action_status') or '').strip() or None
                    action_substate = (row.get('action_substate') or '').strip() or None
                    
                    # Handle periodic_tts special case
                    if intent == 'periodic_tts':
                        # Parse interval values from action_substate (pipe-separated: min|max|default)
                        interval_str = row.get('action_substate') or '30.0|90.0|60.0'
                        try:
                            intervals = [float(x.strip()) for x in interval_str.split('|')]
                            interval_min = intervals[0] if len(intervals) > 0 else 30.0
                            interval_max = intervals[1] if len(intervals) > 1 else 90.0
                            interval = intervals[2] if len(intervals) > 2 else 60.0
                        except (ValueError, IndexError):
                            interval_min = 30.0
                            interval_max = 90.0
                            interval = 60.0
                        
                        # Create one response per TTS response
                        for tts_response in tts_responses:
                            response = {
                                "type": "random",
                                "interval_min": interval_min,
                                "interval_max": interval_max,
                                "interval": interval,
                                "response": tts_response
                            }
                            self.intent_mappings[intent].append({
                                'response': response,
                                'example_phrases': (row.get('example_phrases') or '').split('|') if row.get('example_phrases') else [],
                                'audio_file': (row.get('audio_file') or '').strip() if row.get('audio_file') else None,
                                'usage_count': int(row.get('usage_count') or 0)
                            })
                    else:
                        # Generate all combinations of TTS responses and text scroller texts
                        # If no text_scroller_texts, use None for all
                        if not text_scroller_texts:
                            text_scroller_texts = [None]
                        
                        # Create responses for each TTS/text combination
                        for tts_response in tts_responses:
                            for text_scroller_text in text_scroller_texts:
                                # Build action
                                action = {'type': action_type}
                                if action_status:
                                    action['status'] = action_status
                                if action_substate:
                                    action['substate'] = action_substate
                                
                                # Build text_scroller if text provided
                                # scroll_time will be calculated dynamically from text length when used
                                text_scroller = None
                                if text_scroller_text:
                                    text_scroller = {
                                        'text': text_scroller_text,
                                        'wobble_amount': text_scroller_wobble if text_scroller_wobble is not None else 0.0
                                        # scroll_time calculated dynamically - not stored in CSV
                                    }
                                
                                # Build response
                                response = {
                                    'action': action,
                                    'tts_response': tts_response,
                                    'text_scroller': text_scroller
                                }
                                
                                self.intent_mappings[intent].append({
                                    'response': response,
                                    'example_phrases': (row.get('example_phrases') or '').split('|') if row.get('example_phrases') else [],
                                    'audio_file': (row.get('audio_file') or '').strip() if row.get('audio_file') else None,
                                    'usage_count': int(row.get('usage_count') or 0)
                                })
            
            # Extract unique intent labels for classifier
            self._intent_labels = list(self.intent_mappings.keys())
            
            total_responses = sum(len(resps) for resps in self.intent_mappings.values())
            self.logger.info(f"Loaded {len(self.intent_mappings)} intents with {total_responses} total responses")
            
            # Verify critical intents exist
            critical_intents = ['people', 'eye', 'wave', 'greeting']
            missing = [intent for intent in critical_intents if intent not in self.intent_mappings]
            if missing:
                self.logger.warning(f"[Intent] Missing critical intents: {missing}")
        except Exception as e:
            self.logger.error(f"Error loading intent mappings: {e}")
            import traceback
            traceback.print_exc()
            self.intent_mappings = {}
    
    def _create_example_csv(self):
        """Create an example CSV file with sample mappings"""
        example_data = [
            {
                'intent': 'greeting',
                'example_phrases': 'hello|hi|hey|greetings|howdy|what\'s up',
                'response_json': json.dumps({
                    "action": {"type": "status", "status": "wave"},
                    "tts_response": "hello beautiful human",
                    "text_scroller": None
                }),
                'audio_file': '',
                'usage_count': 0
            },
            {
                'intent': 'eye',
                'example_phrases': 'eye|look|see|watch|gaze|vision',
                'response_json': json.dumps({
                    "action": {"type": "status", "status": "eye"},
                    "tts_response": "opening my eye for you",
                    "text_scroller": None
                }),
                'audio_file': '',
                'usage_count': 0
            },
            {
                'intent': 'people',
                'example_phrases': 'people|person|outline|silhouette|crowd',
                'response_json': json.dumps({
                    "action": {"type": "status", "status": "people"},
                    "tts_response": "showing you the people",
                    "text_scroller": None
                }),
                'audio_file': '',
                'usage_count': 0
            },
        ]
        
        os.makedirs(os.path.dirname(self.intent_csv_path), exist_ok=True)
        
        with open(self.intent_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['intent', 'example_phrases', 'response_json', 'audio_file', 'usage_count'])
            writer.writeheader()
            writer.writerows(example_data)
        
        self.logger.info(f"Created example intent mappings CSV: {self.intent_csv_path}")
        self._load_intent_mappings()
    
    def _get_classifier(self):
        """Load the intent classification model (should be pre-installed)
        
        Model MUST be pre-installed manually:
        python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
        
        The model is cached in ~/.cache/huggingface/ after first installation.
        """
        if self._classifier is None and self.use_intent_model:
            try:
                self.logger.info("Loading intent classification model...")
                # Use zero-shot classification - no training needed!
                # This model can classify text into any set of labels you provide
                # Model should be pre-installed (see INSTALL_INTENT_MODEL.md)
                self._classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1  # Use CPU (-1), or 0 for GPU if available
                )
                self.logger.info("Intent classification model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load intent classification model: {e}")
                self.logger.error("Model must be pre-installed. Run: python -c \"from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')\"")
                raise RuntimeError(f"Intent classification model is required but not found. Pre-install with: python -c \"from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')\"")
        return self._classifier
    
    def preload_model(self):
        """Preload the intent classification model manually (optional)
        
        Call this method to download/load the model before first use.
        Useful for pre-warming or verifying model availability.
        """
        return self._get_classifier()
    
    def _classify_intent_keyword(self, text: str) -> Optional[str]:
        """Classify user input to an intent using keyword matching
        
        Args:
            text: User input text
            
        Returns:
            Intent name if found, None otherwise
        """
        if not text:
            return None
        
        # Normalize input text
        text_lower = text.lower().strip()
        text_words = set(text_lower.split())
        
        # Score each intent based on keyword matches
        intent_scores = {}
        
        for intent_name, mappings in self.intent_mappings.items():
            if not mappings:
                continue
            
            # Get all example phrases for this intent
            all_phrases = []
            for mapping in mappings:
                example_phrases = mapping.get('example_phrases', [])
                all_phrases.extend(example_phrases)
            
            # Count matches
            match_count = 0
            exact_match = False
            
            for phrase in all_phrases:
                phrase_lower = phrase.lower().strip()
                phrase_words = set(phrase_lower.split())
                
                # Check for exact phrase match (highest priority)
                if phrase_lower in text_lower or text_lower in phrase_lower:
                    exact_match = True
                    match_count += 10  # High score for exact match
                    break
                
                # Count word overlaps
                word_overlap = len(text_words & phrase_words)
                if word_overlap > 0:
                    match_count += word_overlap
            
            if match_count > 0 or exact_match:
                intent_scores[intent_name] = match_count
        
        if not intent_scores:
            return None
        
        # Return intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        best_score = intent_scores[best_intent]
        
        self.logger.info(f"[Intent] Keyword matched '{text}' to '{best_intent}' (score: {best_score})")
        return best_intent
    
    def _classify_intent(self, text: str) -> Optional[str]:
        """Classify user input to an intent using ML model or keyword matching
        
        Args:
            text: User input text
            
        Returns:
            Intent name if found, None otherwise
        """
        if not text:
            return None
        
        # Use keyword matching if model is disabled
        if not self.use_intent_model:
            return self._classify_intent_keyword(text)
        
        if not self._intent_labels:
            self.logger.error(f"Intent labels are empty. Loaded {len(self.intent_mappings)} intents from CSV, but _intent_labels is empty.")
            # Fallback to keyword matching
            return self._classify_intent_keyword(text)
        
        try:
            classifier = self._get_classifier()
            if not classifier:
                self.logger.error("Intent classifier not available, falling back to keyword matching")
                return self._classify_intent_keyword(text)
            
            # Create enhanced labels for classification
            # Format: "intent_name: example1, example2, example3"
            # This provides more context to the zero-shot model
            enhanced_labels = []
            for intent_name in self._intent_labels:
                mappings = self.intent_mappings.get(intent_name, [])
                if mappings:
                    # Use first few example phrases for context
                    example_phrases = mappings[0].get('example_phrases', [])
                    if example_phrases:
                        context = ", ".join(example_phrases[:5])  # Use up to 5 examples
                        enhanced_labels.append(f"{intent_name}: {context}")
                    else:
                        enhanced_labels.append(intent_name)
                else:
                    enhanced_labels.append(intent_name)
            
            # Log how many labels we're using (for debugging)
            if len(enhanced_labels) != len(self._intent_labels):
                self.logger.warning(f"Label count mismatch: {len(enhanced_labels)} enhanced vs {len(self._intent_labels)} base")
            
            result = classifier(text, enhanced_labels)
            
            # Get top intent if confidence is reasonable
            if result['labels'] and result['scores']:
                top_label = result['labels'][0]
                confidence = result['scores'][0]
                
                # Extract original intent name from enhanced label
                top_intent = top_label.split(':')[0].strip()
                
                # Accept any confidence level (no threshold)
                self.logger.info(f"[Intent] Classified '{text}' as '{top_intent}' (confidence: {confidence:.2f})")
                return top_intent
            else:
                self.logger.debug(f"No classification result for '{text}', falling back to keyword matching")
                return self._classify_intent_keyword(text)
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}, falling back to keyword matching")
            return self._classify_intent_keyword(text)
    
    def process_voice_command(self, text: str) -> Tuple[bool, bool]:
        """Process a voice command using intent classification
        
        Args:
            text: Input text from speech-to-text
            
        Returns:
            Tuple of (success: bool, is_unknown: bool)
            - success: True if command was processed
            - is_unknown: True if intent not found (always False for intent system since it's instant)
        """
        if not text:
            return False, False
        
        self.logger.info(f"[Intent] Voice command received: '{text}'")
        
        # Classify intent (uses keyword matching if model disabled)
        if self.use_intent_model:
            self.logger.info(f"[Intent] Classifying intent for: '{text}' (using ML model)")
        else:
            self.logger.info(f"[Intent] Classifying intent for: '{text}' (using keyword matching)")
        intent = self._classify_intent(text)
        
        if not intent:
            self.logger.warning(f"[Intent] No intent found for: '{text}'")
            return False, True
        
        self.logger.info(f"[Intent] Intent matched: '{intent}' for text: '{text}'")
        
        # Get response for this intent
        responses = self.intent_mappings.get(intent, [])
        self.logger.info(f"[Intent] Found {len(responses)} response(s) for intent '{intent}'")
        if not responses:
            # Log available intents for debugging
            available_intents = sorted(self.intent_mappings.keys())
            self.logger.warning(f"[Intent] No responses found for intent: '{intent}'")
            self.logger.debug(f"[Intent] Available intents ({len(available_intents)}): {', '.join(available_intents[:10])}{'...' if len(available_intents) > 10 else ''}")
            # Try to reload mappings if intent should exist
            if intent in ['people', 'eye', 'wave', 'greeting']:  # Common intents that should always exist
                self.logger.warning(f"[Intent] Attempting to reload intent mappings...")
                self._load_intent_mappings()
                responses = self.intent_mappings.get(intent, [])
                if responses:
                    self.logger.info(f"[Intent] Successfully reloaded intent '{intent}' with {len(responses)} responses")
                else:
                    self.logger.error(f"[Intent] Intent '{intent}' still not found after reload")
            return False, True
        
        # Select random response
        selected = random.choice(responses)
        response = selected['response']
        
        # Increment usage count
        selected['usage_count'] = selected.get('usage_count', 0) + 1
        
        # Execute response immediately (intent system is instant, no async needed)
        self._execute_response(response, text)
        
        self.logger.info(f"[Intent] Processed '{text}' -> intent '{intent}' successfully")
        return True, False
    
    def set_pending_command_callback(self, callback: Callable[[Dict, str], None]):
        """Set callback for pending commands (compatibility with LLM handler interface)
        
        Note: Intent handler executes immediately, so this callback is not used.
        Kept for interface compatibility.
        """
        # Intent handler doesn't need async callbacks since classification is instant
        pass
    
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
                self.logger.info(f"[Intent] Executing status change to: {status}")
                success = self.status_change_callback(status)
                if success:
                    self.logger.info(f"[Intent] Status changed successfully to: {status}")
                else:
                    self.logger.warning(f"[Intent] Status change failed for: {status}")
        elif action_type == 'tts_only':
            success = True
            self.logger.info(f"[Intent] TTS-only action for: '{text}'")
        
        # Speak response if available
        if success or action_type == 'tts_only':
            tts_response = result.get('tts_response')
            if tts_response and self.tts_callback:
                self.logger.info(f"[Intent] Triggering TTS: '{tts_response}'")
                self.tts_callback(tts_response)
            else:
                if not tts_response:
                    self.logger.warning(f"[Intent] No TTS response in result for: '{text}'")
                if not self.tts_callback:
                    self.logger.warning(f"[Intent] No TTS callback set")
        
        # Check for text scroller and trigger it if available
        text_scroller = result.get('text_scroller')
        if text_scroller and text_scroller.get('text'):
            text = text_scroller.get('text')
            wobble = text_scroller.get('wobble_amount', 0.0)
            if self.text_scroller_callback:
                self.logger.info(f"[Intent] Triggering text scroller: '{text}' (wobble: {wobble})")
                self.text_scroller_callback(text, wobble)
            else:
                self.logger.warning(f"[Intent] Text scroller text available but no callback set")
        
        self.logger.info(f"[Intent] Finished processing command: '{text}'")
    
    # Compatibility property for LLM handler interface
    @property
    def response_storage(self):
        """Compatibility property - returns self for interface compatibility"""
        return self
    
    def get_gesture_trigger(self, gesture_name: str) -> Optional[Dict]:
        """Get gesture trigger configuration from intent mappings
        
        Args:
            gesture_name: Name of gesture (e.g., 'wave', 'smile', 'thumbs_up')
            
        Returns:
            Gesture trigger configuration dict if found, None otherwise
        """
        if gesture_name not in self.available_gestures:
            self.logger.debug(f"Gesture '{gesture_name}' not in available_gestures: {self.available_gestures}")
            return None
        
        # Special case: wave gesture maps to greeting intent
        if gesture_name == 'wave':
            intent = 'greeting'
        else:
            # Look up gesture intent - try gesture_ prefix first
            intent = f"gesture_{gesture_name}"
        
        responses = self.intent_mappings.get(intent, [])
        
        if not responses:
            # Fallback: try just the gesture name (for smile, thumbs_up, peace)
            responses = self.intent_mappings.get(gesture_name, [])
        
        if not responses:
            # Log available intents for debugging
            available_intents = sorted(self.intent_mappings.keys())
            self.logger.warning(
                f"[Intent] No gesture mapping found for: {gesture_name} "
                f"(tried intents: {intent}, {gesture_name}). "
                f"Available intents: {len(available_intents)} total. "
                f"Sample: {available_intents[:5]}..."
            )
            return None
        
        # Select random response
        selected = random.choice(responses)
        result = selected['response'].copy()
        result['gesture'] = gesture_name
        
        return result
    
    def get_tts_trigger(self) -> Optional[Dict]:
        """Get TTS trigger configuration from intent mappings
        
        Returns:
            TTS trigger configuration dict if found, None otherwise
        """
        intent = 'periodic_tts'
        responses = self.intent_mappings.get(intent, [])
        
        if not responses:
            return None
        
        selected = random.choice(responses)
        return selected['response']
    
    def set_status_change_callback(self, callback: Callable[[str], bool]):
        """Set callback for status changes"""
        self.status_change_callback = callback
    
    def set_tts_callback(self, callback: Callable[[str], None]):
        """Set callback for TTS responses"""
        self.tts_callback = callback
    
    def set_text_scroller_callback(self, callback: Callable[[str, float], None]):
        """Set callback for text scroller display
        
        Args:
            callback: Function that takes (text: str, wobble_amount: float) and displays the text scroller
        """
        self.text_scroller_callback = callback
    
    def get_gesture_triggers(self) -> List[Dict]:
        """Get all gesture trigger configurations"""
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
        """Reload intent mappings from CSV"""
        self._load_intent_mappings()
        self.logger.info("Intent mappings reloaded from CSV")

