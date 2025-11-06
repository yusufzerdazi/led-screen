"""
Display modes for LED screen client.

Each mode is a separate class that implements the display logic.
"""

from .base_mode import BaseMode
from .website_mode import WebsiteMode
from .tush_mode import TushMode
from .dashboard_mode import DashboardMode
from .decompression_mode import DecompressionMode
from .hydra_mask_mode import HydraMaskMode

# Mode registry - maps mode names to mode classes
MODE_REGISTRY = {
    'website': WebsiteMode,
    'tush': TushMode,
    'music': TushMode,  # Alias for tush
    'dashboard': DashboardMode,
    'decompression': DecompressionMode,
    'hydra_mask': HydraMaskMode,
    'mask': HydraMaskMode,  # Alias
}

def get_mode(mode_name, width=256, height=144):
    """
    Get a mode instance by name.
    
    Args:
        mode_name: Name of the mode to instantiate
        width: Display width in pixels
        height: Display height in pixels
        
    Returns:
        Mode instance or None if mode not found
    """
    mode_class = MODE_REGISTRY.get(mode_name.lower())
    if mode_class:
        return mode_class(width, height)
    return None

def list_modes():
    """List all available modes"""
    return list(MODE_REGISTRY.keys())

