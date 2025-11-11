"""
Response Storage System for Known Commands

Stores and loads pre-prepared responses for known commands (gestures, greetings)
in a CSV file for efficient access. New responses can be saved to the CSV.
"""

import csv
import json
import os
import random
from typing import Optional, Dict, List
from threading import Lock

# Add parent directory for logger
import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from logger import get_logger


class ResponseStorage:
    """Manages storage and retrieval of pre-prepared responses from CSV"""
    
    def __init__(self, csv_path: str = "client/known_responses.csv"):
        """Initialize response storage
        
        Args:
            csv_path: Path to CSV file for storing responses
        """
        self.csv_path = csv_path
        self.lock = Lock()
        self.logger = get_logger("Response Storage")
        
        # In-memory cache for faster access
        self._cache: Dict[str, List[Dict]] = {}
        self._cache_loaded = False
        
        # Ensure CSV file exists with headers
        self._ensure_csv_exists()
        
        # Load cache on initialization
        self._load_cache()
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            # Create directory if needed
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            
            # Create CSV with headers
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['command_type', 'command_key', 'response_json', 'usage_count'])
            
            self.logger.info(f"Created new response storage CSV: {self.csv_path}")
    
    def _load_cache(self):
        """Load all responses from CSV into memory cache"""
        with self.lock:
            if self._cache_loaded:
                return
            
            self._cache = {}
            
            if not os.path.exists(self.csv_path):
                self._cache_loaded = True
                return
            
            try:
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        command_type = row['command_type']
                        command_key = row['command_key']
                        response_json_str = row['response_json']
                        usage_count = int(row.get('usage_count', 0))
                        
                        # Parse JSON response
                        try:
                            response = json.loads(response_json_str)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse JSON for {command_type}:{command_key}: {e}")
                            continue
                        
                        # Store in cache
                        cache_key = f"{command_type}:{command_key}"
                        if cache_key not in self._cache:
                            self._cache[cache_key] = []
                        
                        self._cache[cache_key].append({
                            'response': response,
                            'usage_count': usage_count
                        })
                
                self._cache_loaded = True
                self.logger.info(f"Loaded {sum(len(resps) for resps in self._cache.values())} responses from CSV")
            except Exception as e:
                self.logger.error(f"Error loading response cache: {e}")
                self._cache_loaded = True
    
    def get_response(self, command_type: str, command_key: str) -> Optional[Dict]:
        """Get a random response for a known command
        
        Args:
            command_type: Type of command ('voice_command', 'gesture')
            command_key: Key identifying the command (e.g., 'hello', 'wave')
            
        Returns:
            Response dict if found, None otherwise
        """
        self._load_cache()
        
        cache_key = f"{command_type}:{command_key}"
        
        with self.lock:
            responses = self._cache.get(cache_key, [])
            
            if not responses:
                return None
            
            # Select random response (weighted by usage count if desired, or just random)
            # For now, just pick random
            selected = random.choice(responses)
            
            # Increment usage count
            selected['usage_count'] = selected.get('usage_count', 0) + 1
            
            # Update CSV (async, don't block)
            self._update_usage_count_async(command_type, command_key, selected['usage_count'])
            
            return selected['response'].copy()
    
    def _update_usage_count_async(self, command_type: str, command_key: str, usage_count: int):
        """Update usage count in CSV (non-blocking)"""
        # This is a simple implementation - in production you might want a background thread
        # For now, we'll update synchronously but it's fast
        try:
            # Read all rows
            rows = []
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['command_type'] == command_type and row['command_key'] == command_key:
                        row['usage_count'] = str(usage_count)
                    rows.append(row)
            
            # Write back
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=['command_type', 'command_key', 'response_json', 'usage_count'])
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as e:
            self.logger.warning(f"Failed to update usage count: {e}")
    
    def save_response(self, command_type: str, command_key: str, response: Dict):
        """Save a new response to CSV
        
        Args:
            command_type: Type of command ('voice_command', 'gesture')
            command_key: Key identifying the command
            response: Response dict to save
        """
        with self.lock:
            # Reload cache to ensure we have latest data
            self._load_cache()
            
            # Check if this exact response already exists
            cache_key = f"{command_type}:{command_key}"
            response_json_str = json.dumps(response, sort_keys=True)
            
            # Check if already exists
            if cache_key in self._cache:
                for existing in self._cache[cache_key]:
                    existing_json = json.dumps(existing['response'], sort_keys=True)
                    if existing_json == response_json_str:
                        self.logger.debug(f"Response already exists for {cache_key}")
                        return
            
            # Add to cache
            if cache_key not in self._cache:
                self._cache[cache_key] = []
            
            self._cache[cache_key].append({
                'response': response,
                'usage_count': 0
            })
            
            # Append to CSV
            try:
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([command_type, command_key, response_json_str, 0])
                
                self.logger.info(f"Saved new response for {command_type}:{command_key}")
            except Exception as e:
                self.logger.error(f"Failed to save response to CSV: {e}")
    
    def has_response(self, command_type: str, command_key: str) -> bool:
        """Check if we have responses for a command
        
        Args:
            command_type: Type of command
            command_key: Key identifying the command
            
        Returns:
            True if responses exist, False otherwise
        """
        self._load_cache()
        
        cache_key = f"{command_type}:{command_key}"
        with self.lock:
            return cache_key in self._cache and len(self._cache[cache_key]) > 0
    
    def get_all_responses(self, command_type: str, command_key: str) -> List[Dict]:
        """Get all responses for a command (for debugging/inspection)
        
        Args:
            command_type: Type of command
            command_key: Key identifying the command
            
        Returns:
            List of all response dicts
        """
        self._load_cache()
        
        cache_key = f"{command_type}:{command_key}"
        with self.lock:
            responses = self._cache.get(cache_key, [])
            return [r['response'].copy() for r in responses]
    
    def reload(self):
        """Reload cache from CSV"""
        with self.lock:
            self._cache_loaded = False
            self._cache.clear()
            self._load_cache()



