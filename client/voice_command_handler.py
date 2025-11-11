"""
Voice Command Handler for Decompression Mode

Handles voice commands with fuzzy matching and Burning Man-themed responses.
Integrates with state management system for dynamic configuration.
"""

import json
import os
from typing import Optional, Dict, List, Tuple, Callable
from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings (0.0 to 1.0)
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class VoiceCommandHandler:
    """Handles voice commands with fuzzy matching and configurable responses"""
    
    def __init__(self, config_path: str = None, state_manager=None):
        """Initialize voice command handler
        
        Args:
            config_path: Path to voice commands JSON config file
            state_manager: Optional StateManager instance for status transitions
        """
        self.state_manager = state_manager
        self.config_path = config_path or self._get_default_config_path()
        self.commands = []
        self.gesture_triggers = []
        self.tts_triggers = []
        self.similarity_threshold = 0.6
        self.max_responses = 1
        
        # Callbacks for actions
        self.status_change_callback: Optional[Callable[[str], bool]] = None  # Returns success
        self.tts_callback: Optional[Callable[[str], None]] = None
        
        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default path to voice commands config file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, 'voice_commands.json')
    
    def load_config(self):
        """Load voice command configuration from JSON file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.commands = config.get('commands', [])
        self.gesture_triggers = config.get('gesture_triggers', [])
        self.tts_triggers = config.get('tts_triggers', [])
        self.tts_output_device_index = config.get('tts_output_device_index', None)
        self.similarity_threshold = config.get('similarity_threshold', 0.6)
        self.max_responses = config.get('max_responses', 1)
        
        print(f"[VoiceCommandHandler] Loaded {len(self.commands)} voice commands, {len(self.gesture_triggers)} gesture triggers, and {len(self.tts_triggers)} TTS triggers from {self.config_path}")
        if self.tts_output_device_index is not None:
            print(f"[VoiceCommandHandler] TTS output device index: {self.tts_output_device_index}")
    
    def reload_config(self):
        """Reload configuration from file"""
        self.load_config()
    
    def set_status_change_callback(self, callback: Callable[[str], bool]):
        """Set callback for status changes
        
        Args:
            callback: Function(status) -> bool that changes status and returns success
        """
        self.status_change_callback = callback
    
    def set_tts_callback(self, callback: Callable[[str], None]):
        """Set callback for TTS responses
        
        Args:
            callback: Function(text) that speaks the text
        """
        self.tts_callback = callback
    
    def find_best_match(self, text: str) -> Optional[Tuple[Dict, float]]:
        """Find the best matching command for the given text
        
        Uses fuzzy matching with word-level and phrase-level similarity.
        
        Args:
            text: Input text to match against commands
            
        Returns:
            Tuple of (command_dict, similarity_score) or None if no match found
        """
        if not text or not self.commands:
            return None
        
        text_lower = text.lower().strip()
        best_match = None
        best_score = 0.0
        
        for command in self.commands:
            triggers = command.get('triggers', [])
            
            # Check each trigger
            for trigger in triggers:
                trigger_lower = trigger.lower()
                trigger_words = trigger_lower.split()
                
                # Calculate similarity scores
                scores = []
                
                # 1. Exact phrase similarity
                phrase_similarity = similarity(text_lower, trigger_lower)
                scores.append(phrase_similarity)
                
                # 2. Word-level matching (all words present)
                if len(trigger_words) > 1:
                    all_words_present = all(word in text_lower for word in trigger_words)
                    if all_words_present:
                        # Calculate word order similarity
                        matched_words = sum(1 for word in trigger_words if word in text_lower)
                        word_score = matched_words / len(trigger_words)
                        # Boost if words are in similar order
                        if word_score > 0.5:
                            word_score = min(1.0, word_score + 0.2)
                        scores.append(word_score)
                
                # 3. Individual word matches (for single word triggers)
                if len(trigger_words) == 1:
                    if trigger_lower in text_lower:
                        # Word found - calculate context similarity
                        word_similarity = similarity(text_lower, trigger_lower)
                        scores.append(word_similarity)
                
                # Use the best score from all methods
                if scores:
                    score = max(scores)
                    if score > best_score:
                        best_score = score
                        best_match = command
        
        # Only return if score meets threshold
        if best_match and best_score >= self.similarity_threshold:
            return (best_match, best_score)
        
        return None
    
    def process_command(self, text: str) -> bool:
        """Process a voice command and execute the appropriate action
        
        Args:
            text: Input text from speech-to-text
            
        Returns:
            True if a command was matched and executed, False otherwise
        """
        if not text:
            return False
        
        match_result = self.find_best_match(text)
        if not match_result:
            return False
        
        command, score = match_result
        action = command.get('action', {})
        action_type = action.get('type')
        
        print(f"[VoiceCommandHandler] Matched command (similarity: {score:.2f}): {text}")
        
        # Execute action
        success = False
        if action_type == 'status':
            status = action.get('status')
            if status and self.status_change_callback:
                success = self.status_change_callback(status)
                if success:
                    print(f"[VoiceCommandHandler] Status changed to: {status}")
        elif action_type == 'tts_only':
            # TTS-only commands always succeed (they just speak)
            success = True
        
        # Speak response if available
        if success or action_type == 'tts_only':
            responses = command.get('responses', [])
            if responses and self.tts_callback:
                import random
                response = random.choice(responses)
                self.tts_callback(response)
                print(f"[VoiceCommandHandler] Response: {response}")
            elif responses and not self.tts_callback:
                print(f"[VoiceCommandHandler] TTS callback not set, would have spoken: {responses[0]}")
        
        return success
    
    def add_command(self, triggers: List[str], action: Dict, responses: List[str]):
        """Dynamically add a command at runtime
        
        Args:
            triggers: List of trigger phrases
            action: Action dictionary with 'type' and action-specific fields
            responses: List of possible TTS responses
        """
        command = {
            'triggers': triggers,
            'action': action,
            'responses': responses
        }
        self.commands.append(command)
        print(f"[VoiceCommandHandler] Added command: {triggers}")
    
    def remove_command(self, index: int) -> bool:
        """Remove a command by index
        
        Args:
            index: Index of command to remove
            
        Returns:
            True if command was removed, False if index invalid
        """
        if 0 <= index < len(self.commands):
            removed = self.commands.pop(index)
            print(f"[VoiceCommandHandler] Removed command: {removed.get('triggers', [])}")
            return True
        return False
    
    def get_commands(self) -> List[Dict]:
        """Get all current commands
        
        Returns:
            List of command dictionaries
        """
        return self.commands.copy()
    
    def get_gesture_trigger(self, gesture_name: str) -> Optional[Dict]:
        """Get gesture trigger configuration for a gesture
        
        Args:
            gesture_name: Name of gesture (e.g., 'wave', 'smile', 'thumbs_up')
            
        Returns:
            Gesture trigger configuration dict or None if not found
        """
        for trigger in self.gesture_triggers:
            if trigger.get('gesture') == gesture_name:
                return trigger
        return None
    
    def get_gesture_triggers(self) -> List[Dict]:
        """Get all gesture trigger configurations
        
        Returns:
            List of gesture trigger dictionaries
        """
        return self.gesture_triggers.copy()
    
    def get_tts_triggers(self) -> List[Dict]:
        """Get all TTS trigger configurations
        
        Returns:
            List of TTS trigger dictionaries
        """
        return self.tts_triggers.copy()

