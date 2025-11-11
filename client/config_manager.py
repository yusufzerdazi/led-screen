"""
Centralized Configuration Manager for Decompression Mode

Manages all configuration (gestures, voice commands, TTS triggers) in a unified,
maintainable way. Handles loading from CSV, LLM generation, and caching.
"""

import json
import os
from typing import Optional, Dict, List, Any
from threading import Lock
from logger import get_logger
from response_storage import ResponseStorage


class ConfigManager:
    """Centralized configuration manager for all decompression mode configs"""
    
    def __init__(self, response_storage: ResponseStorage):
        """Initialize config manager
        
        Args:
            response_storage: ResponseStorage instance for CSV persistence
        """
        self.response_storage = response_storage
        self.logger = get_logger("Config Manager")
        self.lock = Lock()
        
        # Cache for loaded configs
        self._gesture_configs: Dict[str, List[Dict]] = {}
        self._voice_configs: Dict[str, List[Dict]] = {}
        self._tts_triggers: List[Dict] = []
        
        # Load all configs from CSV
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configurations from CSV storage"""
        with self.lock:
            # Load gesture configs
            gesture_types = ['gesture']
            for gesture_type in gesture_types:
                # Get all gesture names from available gestures
                gesture_names = self.get_available_gestures()
                for gesture_name in gesture_names:
                    response = self.response_storage.get_response(gesture_type, gesture_name)
                    if response:
                        if gesture_name not in self._gesture_configs:
                            self._gesture_configs[gesture_name] = []
                        self._gesture_configs[gesture_name].append(response)
            
            # Load voice command configs
            voice_types = ['voice_command']
            voice_keys = ['hello', 'hi', 'eye', 'people']
            for voice_type in voice_types:
                for key in voice_keys:
                    response = self.response_storage.get_response(voice_type, key)
                    if response:
                        if key not in self._voice_configs:
                            self._voice_configs[key] = []
                        self._voice_configs[key].append(response)
            
            self.logger.info(f"Loaded {sum(len(v) for v in self._gesture_configs.values())} gesture configs, "
                           f"{sum(len(v) for v in self._voice_configs.values())} voice configs")
    
    def get_available_gestures(self) -> List[str]:
        """Get list of all available gesture names"""
        return [
            'wave', 'thumbs_up', 'smile', 'peace', 'heart', 'rock_on',
            'point', 'clap', 'fist_pump'
        ]
    
    def get_gesture_config(self, gesture_name: str) -> Optional[Dict]:
        """Get a random gesture configuration
        
        Args:
            gesture_name: Name of gesture (e.g., 'wave', 'peace', 'heart')
            
        Returns:
            Gesture config dict or None if not found
        """
        with self.lock:
            configs = self._gesture_configs.get(gesture_name, [])
            if configs:
                import random
                return random.choice(configs).copy()
            return None
    
    def get_voice_config(self, command_key: str) -> Optional[Dict]:
        """Get a random voice command configuration
        
        Args:
            command_key: Command key (e.g., 'hello', 'hi', 'eye')
            
        Returns:
            Voice config dict or None if not found
        """
        with self.lock:
            configs = self._voice_configs.get(command_key, [])
            if configs:
                import random
                return random.choice(configs).copy()
            return None
    
    def save_gesture_config(self, gesture_name: str, config: Dict):
        """Save a gesture configuration to storage
        
        Args:
            gesture_name: Name of gesture
            config: Configuration dict
        """
        self.response_storage.save_response('gesture', gesture_name, config)
        # Reload cache
        with self.lock:
            if gesture_name not in self._gesture_configs:
                self._gesture_configs[gesture_name] = []
            self._gesture_configs[gesture_name].append(config)
    
    def save_voice_config(self, command_key: str, config: Dict):
        """Save a voice command configuration to storage
        
        Args:
            command_key: Command key
            config: Configuration dict
        """
        self.response_storage.save_response('voice_command', command_key, config)
        # Reload cache
        with self.lock:
            if command_key not in self._voice_configs:
                self._voice_configs[command_key] = []
            self._voice_configs[command_key].append(config)
    
    def reload(self):
        """Reload all configurations from storage"""
        with self.lock:
            self._gesture_configs.clear()
            self._voice_configs.clear()
            self._load_all_configs()
    
    def has_gesture_config(self, gesture_name: str) -> bool:
        """Check if gesture config exists
        
        Args:
            gesture_name: Name of gesture
            
        Returns:
            True if config exists
        """
        with self.lock:
            return gesture_name in self._gesture_configs and len(self._gesture_configs[gesture_name]) > 0
    
    def has_voice_config(self, command_key: str) -> bool:
        """Check if voice config exists
        
        Args:
            command_key: Command key
            
        Returns:
            True if config exists
        """
        with self.lock:
            return command_key in self._voice_configs and len(self._voice_configs[command_key]) > 0


