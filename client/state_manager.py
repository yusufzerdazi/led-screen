"""
State Management System for Decompression Mode

Handles state transitions based on time, random selection, and user interactions.
Designed to be dynamically configurable and extensible.
"""

import time
import random
from threading import Lock
from typing import Optional, List, Dict, Callable, Any


class StateManager:
    """Manages state transitions with support for time-based, random, and interaction-based rules"""
    
    def __init__(self, status_cooldowns: Dict[str, float], 
                 main_statuses: List[str] = None,
                 interactive_statuses: List[str] = None):
        """Initialize state manager
        
        Args:
            status_cooldowns: Dictionary mapping status names to cooldown times (seconds)
            main_statuses: List of main statuses that participate in automatic transitions
            interactive_statuses: List of interactive statuses (temporary, don't interrupt auto transitions)
        """
        self.status_cooldowns = status_cooldowns.copy()
        self.status_last_executed = {}  # Maps status -> timestamp of last execution
        
        self.main_statuses = main_statuses.copy() if main_statuses else []
        self.interactive_statuses = interactive_statuses.copy() if interactive_statuses else []
        
        self.transition_rules = []
        self.last_time_based_transition = time.time()
        self.transition_enabled = True
        
        self.lock = Lock()
        
        # Callbacks for state changes
        self.on_status_change_callbacks: List[Callable[[str, str], None]] = []  # (old_status, new_status)
        self.on_transition_evaluated_callbacks: List[Callable[[Optional[str]], None]] = []  # (target_status)
    
    def register_status_change_callback(self, callback: Callable[[str, str], None]):
        """Register a callback to be called when status changes
        
        Args:
            callback: Function(old_status, new_status) called when status changes
        """
        self.on_status_change_callbacks.append(callback)
    
    def register_transition_evaluated_callback(self, callback: Callable[[Optional[str]], None]):
        """Register a callback to be called when transitions are evaluated
        
        Args:
            callback: Function(target_status) called with the evaluated target status (or None)
        """
        self.on_transition_evaluated_callbacks.append(callback)
    
    def is_status_on_cooldown(self, status: str) -> tuple[bool, float]:
        """Check if a status is currently on cooldown
        
        Args:
            status: Status name to check
            
        Returns:
            tuple: (is_on_cooldown: bool, remaining_time: float)
                   If not on cooldown, remaining_time is 0.0
        """
        if status not in self.status_cooldowns:
            return (False, 0.0)
        
        if status not in self.status_last_executed:
            return (False, 0.0)
        
        cooldown_time = self.status_cooldowns[status]
        last_executed = self.status_last_executed[status]
        elapsed = time.time() - last_executed
        remaining = max(0.0, cooldown_time - elapsed)
        
        return (remaining > 0.0, remaining)
    
    def get_cooldown_info(self) -> Dict[str, Dict[str, float]]:
        """Get cooldown information for all statuses
        
        Returns:
            dict: Maps status -> {'remaining': float, 'cooldown': float, 'progress': float}
                  progress is 0.0 (just started) to 1.0 (cooldown complete)
        """
        cooldown_info = {}
        current_time = time.time()
        
        for status, cooldown_time in self.status_cooldowns.items():
            if status not in self.status_last_executed:
                # Never executed, cooldown complete
                cooldown_info[status] = {
                    'remaining': 0.0,
                    'cooldown': cooldown_time,
                    'progress': 1.0
                }
            else:
                last_executed = self.status_last_executed[status]
                elapsed = current_time - last_executed
                remaining = max(0.0, cooldown_time - elapsed)
                progress = min(1.0, elapsed / cooldown_time) if cooldown_time > 0 else 1.0
                
                cooldown_info[status] = {
                    'remaining': remaining,
                    'cooldown': cooldown_time,
                    'progress': progress
                }
        
        return cooldown_info
    
    def record_status_execution(self, status: str):
        """Record that a status was executed (for cooldown tracking)
        
        Args:
            status: Status name that was executed
        """
        with self.lock:
            self.status_last_executed[status] = time.time()
    
    def add_transition_rule(self, rule_type: str, **kwargs):
        """Add a transition rule to the state management system
        
        Args:
            rule_type: Type of rule - 'time', 'random', or 'interaction'
            **kwargs: Rule-specific parameters:
                - For 'time': interval (seconds), from_status (optional), to_status (optional, can be list)
                - For 'random': interval (seconds), status_pool (list of statuses to choose from)
                - For 'interaction': gesture (string), status (target status), duration (optional)
        """
        rule = {
            'type': rule_type,
            'enabled': True,
            **kwargs
        }
        self.transition_rules.append(rule)
        print(f"[StateManager] Added transition rule: {rule_type} with params {kwargs}")
    
    def get_available_statuses(self, status_pool: Optional[List[str]] = None) -> List[str]:
        """Get list of statuses available for transition (not on cooldown)
        
        Args:
            status_pool: Optional list of statuses to filter from. If None, uses all statuses.
            
        Returns:
            List of status names that are not on cooldown
        """
        if status_pool is None:
            status_pool = list(self.status_cooldowns.keys())
        
        available = []
        for status in status_pool:
            is_on_cooldown, _ = self.is_status_on_cooldown(status)
            if not is_on_cooldown:
                available.append(status)
        
        return available
    
    def _evaluate_time_transition(self, rule: Dict, current_time: float, current_status: str) -> Optional[str]:
        """Evaluate a time-based transition rule
        
        Args:
            rule: Transition rule dictionary
            current_time: Current timestamp
            current_status: Current status name
            
        Returns:
            Target status if transition should occur, None otherwise
        """
        if not rule.get('enabled', True):
            return None
        
        interval = rule.get('interval', 0)
        if interval <= 0:
            return None
        
        # Check if enough time has passed since last time-based transition
        time_since_last = current_time - self.last_time_based_transition
        if time_since_last < interval:
            return None
        
        # Check if current status matches from_status (if specified)
        from_status = rule.get('from_status')
        if from_status and current_status != from_status:
            return None
        
        # Don't transition if we're in an interactive status
        if current_status in self.interactive_statuses:
            return None
        
        # Get target status
        to_status = rule.get('to_status')
        if to_status is None:
            return None
        
        # Handle list of possible target statuses
        if isinstance(to_status, list):
            # Filter to available (not on cooldown) statuses
            available = self.get_available_statuses(to_status)
            if not available:
                # If no statuses are available (all on cooldown), still pick one from the list
                # This ensures transitions continue even when cooldowns haven't expired
                if to_status:
                    to_status = random.choice(to_status)
                else:
                    return None
            else:
                # Randomly choose from available
                to_status = random.choice(available)
        
        # Check if target status is on cooldown
        # For time-based transitions with a single status, allow it even if on cooldown
        # This ensures transitions continue even when cooldowns haven't expired
        is_on_cooldown, remaining = self.is_status_on_cooldown(to_status)
        if is_on_cooldown:
            # Log that we're allowing a transition despite cooldown
            # This is intentional to ensure transitions continue
            pass  # Allow transition even if on cooldown
        
        return to_status
    
    def _evaluate_random_transition(self, rule: Dict, current_time: float, current_status: str) -> Optional[str]:
        """Evaluate a random transition rule
        
        Args:
            rule: Transition rule dictionary
            current_time: Current timestamp
            current_status: Current status name
            
        Returns:
            Target status if transition should occur, None otherwise
        """
        if not rule.get('enabled', True):
            return None
        
        interval = rule.get('interval', 0)
        if interval <= 0:
            return None
        
        # Check if enough time has passed since last time-based transition
        time_since_last = current_time - self.last_time_based_transition
        if time_since_last < interval:
            return None
        
        # Don't transition if we're in an interactive status
        if current_status in self.interactive_statuses:
            return None
        
        # Get status pool
        status_pool = rule.get('status_pool', self.main_statuses)
        if not status_pool:
            return None
        
        # Filter to available (not on cooldown) statuses, excluding current
        available = [s for s in self.get_available_statuses(status_pool) if s != current_status]
        if not available:
            # If no statuses are available (all on cooldown), still pick one from the pool
            # This ensures transitions continue even when cooldowns haven't expired
            fallback_pool = [s for s in status_pool if s != current_status]
            if fallback_pool:
                return random.choice(fallback_pool)
            return None
        
        # Randomly choose from available
        return random.choice(available)
    
    def evaluate_transitions(self, current_status: str, status_duration: Optional[float] = None, 
                            status_start_time: Optional[float] = None) -> Optional[str]:
        """Evaluate all transition rules and return the next status if any should trigger
        
        Args:
            current_status: Current status name
            status_duration: Optional duration of current status (None = indefinite)
            status_start_time: Optional start time of current status (for checking if duration elapsed)
            
        Returns:
            Target status if transition should occur, None otherwise
        """
        if not self.transition_enabled:
            return None
        
        current_time = time.time()
        
        # Don't transition if we're in a timed status that hasn't completed
        if status_duration and status_start_time:
            elapsed = current_time - status_start_time
            if elapsed < status_duration:
                return None
        
        # Evaluate all rules in order
        for rule in self.transition_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'time':
                target_status = self._evaluate_time_transition(rule, current_time, current_status)
                if target_status:
                    self.last_time_based_transition = current_time
                    # Notify callbacks
                    for callback in self.on_transition_evaluated_callbacks:
                        callback(target_status)
                    return target_status
            
            elif rule_type == 'random':
                target_status = self._evaluate_random_transition(rule, current_time, current_status)
                if target_status:
                    self.last_time_based_transition = current_time
                    # Notify callbacks
                    for callback in self.on_transition_evaluated_callbacks:
                        callback(target_status)
                    return target_status
            
            # 'interaction' rules are handled separately via gesture detection
            # They don't need evaluation here as they're triggered by user actions
        
        return None
    
    def enable_transitions(self):
        """Enable automatic state transitions"""
        self.transition_enabled = True
        print("[StateManager] State transitions enabled")
    
    def disable_transitions(self):
        """Disable automatic state transitions"""
        self.transition_enabled = False
        print("[StateManager] State transitions disabled")
    
    def clear_transition_rules(self):
        """Clear all transition rules"""
        self.transition_rules = []
        print("[StateManager] All transition rules cleared")
    
    def get_transition_rules(self) -> List[Dict]:
        """Get all current transition rules
        
        Returns:
            List of transition rule dictionaries
        """
        return self.transition_rules.copy()
    
    def remove_transition_rule(self, index: int) -> bool:
        """Remove a transition rule by index
        
        Args:
            index: Index of rule to remove
            
        Returns:
            True if rule was removed, False if index invalid
        """
        if 0 <= index < len(self.transition_rules):
            removed = self.transition_rules.pop(index)
            print(f"[StateManager] Removed transition rule: {removed}")
            return True
        return False
    
    def add_main_status(self, status: str):
        """Add a status to the main statuses pool (for random selection)
        
        Args:
            status: Status name to add
        """
        if status not in self.main_statuses:
            self.main_statuses.append(status)
            print(f"[StateManager] Added '{status}' to main statuses pool")
    
    def remove_main_status(self, status: str):
        """Remove a status from the main statuses pool
        
        Args:
            status: Status name to remove
        """
        if status in self.main_statuses:
            self.main_statuses.remove(status)
            print(f"[StateManager] Removed '{status}' from main statuses pool")
    
    def add_interactive_status(self, status: str):
        """Add a status to the interactive statuses list
        
        Interactive statuses are temporary and don't participate in automatic transitions.
        
        Args:
            status: Status name to add
        """
        if status not in self.interactive_statuses:
            self.interactive_statuses.append(status)
            print(f"[StateManager] Added '{status}' to interactive statuses")
    
    def get_state_info(self, current_status: str, previous_status: Optional[str] = None,
                      status_start_time: Optional[float] = None, 
                      status_duration: Optional[float] = None,
                      status_substate: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive state information for debugging/monitoring
        
        Args:
            current_status: Current status name
            previous_status: Previous status name (optional)
            status_start_time: Start time of current status (optional)
            status_duration: Duration of current status (optional)
            status_substate: Current substate (optional)
            
        Returns:
            dict with current state, available transitions, cooldowns, and rules
        """
        current_time = time.time()
        return {
            'current_status': current_status,
            'previous_status': previous_status,
            'status_start_time': status_start_time,
            'status_duration': status_duration,
            'status_substate': status_substate,
            'elapsed': current_time - status_start_time if status_start_time else 0.0,
            'main_statuses': self.main_statuses.copy(),
            'interactive_statuses': self.interactive_statuses.copy(),
            'transition_enabled': self.transition_enabled,
            'transition_rules': self.get_transition_rules(),
            'available_statuses': self.get_available_statuses(),
            'cooldown_info': self.get_cooldown_info(),
            'last_time_based_transition': self.last_time_based_transition,
            'time_since_last_transition': current_time - self.last_time_based_transition
        }

