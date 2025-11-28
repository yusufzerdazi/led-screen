import time
import threading
import sys
import os
import shutil
from collections import deque
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import queue

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.prompt import Prompt
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Install with: pip install rich")


@dataclass
class ServiceStatus:
    """Status information for a service"""
    name: str
    status: str = "Unknown"  # "Running", "Stopped", "Error", etc.
    details: str = ""
    last_update: float = field(default_factory=time.time)
    fps: Optional[float] = None
    frame_count: int = 0
    error_count: int = 0
    enabled: bool = True  # Whether service is enabled/disabled


@dataclass
class LogEntry:
    """A log entry from any service"""
    timestamp: float
    service: str
    level: str  # "INFO", "WARNING", "ERROR", "DEBUG"
    message: str


class LogCapture:
    """Captures logs from all services"""

    def __init__(self, max_logs: int = 100):
        self.logs: deque = deque(maxlen=max_logs)
        self.lock = threading.Lock()

    def add_log(self, service: str, level: str, message: str):
        """Add a log entry"""
        with self.lock:
            self.logs.append(LogEntry(
                timestamp=time.time(),
                service=service,
                level=level,
                message=message
            ))

    def get_recent_logs(self, count: int = 20) -> List[LogEntry]:
        """Get recent log entries"""
        with self.lock:
            return list(self.logs)[-count:]


class SystemMonitor:
    """Monitors system CPU and memory usage"""

    def __init__(self, max_samples: int = 60):
        self.max_samples = max_samples
        self.cpu_history: deque = deque(maxlen=max_samples)
        self.memory_history: deque = deque(maxlen=max_samples)
        self.lock = threading.Lock()
        self.process = None
        self.cpu_count = 1
        if PSUTIL_AVAILABLE:
            try:
                self.process = psutil.Process(os.getpid())
                self.cpu_count = psutil.cpu_count() or 1  # Get number of CPU cores
            except Exception:
                pass

    def update(self):
        """Update system metrics"""
        if not PSUTIL_AVAILABLE:
            return

        with self.lock:
            try:
                # Get system-wide CPU usage (can exceed 100% on multi-core)
                cpu_percent = psutil.cpu_percent(interval=None)

                # Get process memory
                if self.process:
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
                else:
                    # Fallback to system memory if process not available
                    memory_info = psutil.virtual_memory()
                    memory_mb = memory_info.used / 1024 / 1024

                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_mb)
            except Exception:
                pass

    def get_cpu_stats(self) -> Dict:
        """Get CPU statistics"""
        with self.lock:
            if not self.cpu_history:
                return {'current': 0.0, 'avg': 0.0, 'max': 0.0, 'history': []}

            history = list(self.cpu_history)
            return {
                'current': history[-1] if history else 0.0,
                'avg': sum(history) / len(history) if history else 0.0,
                'max': max(history) if history else 0.0,
                'history': history
            }

    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        with self.lock:
            if not self.memory_history:
                return {'current': 0.0, 'avg': 0.0, 'max': 0.0, 'history': []}

            history = list(self.memory_history)
            return {
                'current': history[-1] if history else 0.0,
                'avg': sum(history) / len(history) if history else 0.0,
                'max': max(history) if history else 0.0,
                'history': history
            }


class PerformanceTracker:
    """Tracks performance metrics for services"""

    def __init__(self):
        self.trackers: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def update(self, service_name: str, frame_time: Optional[float] = None):
        """Update performance metrics for a service"""
        with self.lock:
            if service_name not in self.trackers:
                self.trackers[service_name] = {
                    'frame_times': deque(maxlen=60),  # Keep last 60 frames
                    'last_frame_time': time.time(),
                    'frame_count': 0,
                    'total_time': 0.0
                }

            tracker = self.trackers[service_name]
            current_time = time.time()

            if frame_time is not None:
                tracker['frame_times'].append(frame_time)
            else:
                # Calculate frame time from last update
                elapsed = current_time - tracker['last_frame_time']
                if elapsed > 0:
                    tracker['frame_times'].append(elapsed)

            tracker['last_frame_time'] = current_time
            tracker['frame_count'] += 1

    def get_fps(self, service_name: str) -> Optional[float]:
        """Get current FPS for a service"""
        with self.lock:
            if service_name not in self.trackers:
                return None

            tracker = self.trackers[service_name]
            frame_times = list(tracker['frame_times'])

            if len(frame_times) < 2:
                return None

            # Calculate average FPS from recent frame times
            avg_frame_time = sum(frame_times) / len(frame_times)
            if avg_frame_time > 0:
                return 1.0 / avg_frame_time
            return None

    def get_stats(self, service_name: str) -> Dict:
        """Get performance statistics for a service"""
        with self.lock:
            if service_name not in self.trackers:
                return {}

            tracker = self.trackers[service_name]
            frame_times = list(tracker['frame_times'])

            stats = {
                'frame_count': tracker['frame_count'],
                'fps': self.get_fps(service_name)
            }

            if frame_times:
                stats['avg_frame_time'] = sum(frame_times) / len(frame_times)
                stats['min_frame_time'] = min(frame_times)
                stats['max_frame_time'] = max(frame_times)

            return stats


class KaleidoscapeUI:
    """Main console UI for Kaleidoscape monitoring"""

    def __init__(self, client=None, mode=None):
        if not RICH_AVAILABLE:
            raise ImportError("rich library is required. Install with: pip install rich")

        self.client = client
        self.mode = mode

        # Detect terminal dimensions and leave one spare column to avoid hard-wrap
        term = shutil.get_terminal_size()
        safe_width = max(40, (term.columns or 80))  # <-- key fix: minus one column
        safe_height = term.lines or 24

        # Create console
        self.console = Console(
            force_terminal=True,
            width=safe_width,
            height=safe_height,
            legacy_windows=False,
        )
        self.terminal_width = self.console.width
        self.terminal_height = self.console.height

        self.log_capture = LogCapture(max_logs=500)
        self.performance = PerformanceTracker()
        self.system_monitor = SystemMonitor(max_samples=60)
        self.services: Dict[str, ServiceStatus] = {}
        self.running = False
        self.command_queue = queue.Queue()
        self.command_handlers: Dict[str, Callable] = {}

        # Input state for displaying current input in UI
        self.current_input = ""
        self.input_lock = threading.Lock()
        
        # Flag to pause Live updates for command input
        self.pause_live = threading.Event()
        self.command_pending = threading.Event()

        # Set up centralized logging for services
        self._setup_service_logging()

        # Initialize service statuses
        self._init_services()
        self._init_command_handlers()

        # Start update thread
        self.update_thread = None

    def _init_services(self):
        """Initialize service status tracking"""
        self.services = {
            'LED Display': ServiceStatus(name='LED Display'),
            'Camera': ServiceStatus(name='Camera'),
            'Face Detection': ServiceStatus(name='Face Detection'),
            'Gesture Detection': ServiceStatus(name='Gesture Detection'),
            'People Segmentation': ServiceStatus(name='People Segmentation'),
            'Audio (TTS)': ServiceStatus(name='Audio (TTS)'),
            'Audio (STT)': ServiceStatus(name='Audio (STT)'),
            'Video Manager': ServiceStatus(name='Video Manager'),
            'AI Service': ServiceStatus(name='AI Service'),
        }

    def _setup_service_logging(self):
        """Set up centralized logging so services can log to the console UI"""
        try:
            from logger import set_log_callback
            set_log_callback(self.log)
        except ImportError:
            # Logger module not available, services will use print
            pass

    def _init_command_handlers(self):
        """Initialize command handlers"""
        self.command_handlers = {
            'status': self._cmd_set_status,
            'wave': self._cmd_trigger_wave,
            'thumbs_up': self._cmd_trigger_thumbs_up,
            'smile': self._cmd_trigger_smile,
            'eye': lambda: self._cmd_set_status('eye'),
            'people': lambda: self._cmd_set_status('people'),
            'start': self._cmd_start_service,
            'stop': self._cmd_stop_service,
            'enable': self._cmd_enable_service,
            'disable': self._cmd_disable_service,
            'switch_visual': self._cmd_switch_visual,
            'switch_color': self._cmd_switch_color,
            'color_mode': self._cmd_switch_color,  # Alias
            'help': self._cmd_help,
        }

    def log(self, service: str, level: str, message: str):
        """Add a log entry"""
        self.log_capture.add_log(service, level, message)

    def update_service_status(self, service_name: str, status: str, details: str = "", fps: Optional[float] = None):
        """Update status of a service"""
        if service_name in self.services:
            self.services[service_name].status = status
            self.services[service_name].details = details
            self.services[service_name].last_update = time.time()
            if fps is not None:
                self.services[service_name].fps = fps

    def track_performance(self, service_name: str, frame_time: Optional[float] = None):
        """Track performance for a service"""
        self.performance.update(service_name, frame_time)
        fps = self.performance.get_fps(service_name)
        if fps is not None:
            self.update_service_status(service_name, "Running", f"FPS: {fps:.1f}", fps)

    def _create_status_table(self) -> Table:
        """Create status overview table using Rich defaults"""
        # expand=True and pad_edge=False help precise width control
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, expand=True, pad_edge=False)
        # Slightly tighter columns, and let Details fold/wrap
        table.add_column("Service", style="cyan", no_wrap=True, width=16)
        table.add_column("Status", no_wrap=True, width=9)
        table.add_column("FPS", justify="right", no_wrap=True, width=6)
        table.add_column("Details", overflow="fold")  # allow wrapping instead of pushing into last column

        for service in self.services.values():
            status_style = {
                "Running": "green",
                "Stopped": "red",
                "Error": "red bold",
                "Initializing": "yellow",
                "Unknown": "dim"
            }.get(service.status, "white")

            fps_str = f"{service.fps:.1f}" if service.fps else "N/A"
            table.add_row(
                service.name,
                Text(service.status, style=status_style),
                fps_str,
                service.details or "-"
            )

        return table

    def _create_logs_panel(self) -> Panel:
        """Create logs panel"""
        logs = self.log_capture.get_recent_logs(40)

        if not logs:
            return Panel("No logs yet", title="Recent Logs", border_style="blue", padding=(0, 1))

        log_text = Text()
        for log in logs:
            dt = datetime.fromtimestamp(log.timestamp)
            time_str = dt.strftime("%H:%M:%S")

            level_colors = {
                "ERROR": "red",
                "WARNING": "yellow",
                "INFO": "blue",
                "DEBUG": "dim"
            }
            level_color = level_colors.get(log.level, "white")

            log_text.append(f"[{time_str}] ", style="dim")
            log_text.append(f"[{log.service}] ", style="cyan")
            log_text.append(f"{log.level}: ", style=level_color)
            log_text.append(f"{log.message}\n", style="white")

        return Panel(log_text, title="Recent Logs", border_style="blue", padding=(0, 1))

    def _create_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a text-based progress bar
        
        Args:
            progress: Progress value from 0.0 to 1.0
            width: Width of the progress bar in characters
            
        Returns:
            String representation of progress bar
        """
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        filled = int(progress * width)
        empty = width - filled
        
        # Use Unicode block characters for smoother appearance
        if filled == width:
            bar = "█" * width
        elif filled == 0:
            bar = "░" * width
        else:
            bar = "█" * filled + "░" * empty
        
        return bar
    
    def _create_performance_panel(self) -> str:
        """Create performance metrics panel"""
        overall_fps = self.performance.get_fps('LED Display')
        overall_fps_str = f"{overall_fps:.1f}" if overall_fps else "N/A"

        mode_info = ""
        if self.mode:
            try:
                current_status = self.mode.get_status()
                mode_info = f"Mode: {current_status}\n"
            except Exception:
                pass

        cpu_stats = self.system_monitor.get_cpu_stats()
        memory_stats = self.system_monitor.get_memory_stats()

        # psutil.cpu_percent() already returns normalized percentage (0-100%) averaged across all cores
        # No need to divide by cpu_count - that was incorrect
        cpu_current = cpu_stats['current']
        cpu_avg = cpu_stats['avg']
        cpu_max = cpu_stats['max']

        perf_text = f"FPS: {overall_fps_str}\n"
        if mode_info:
            perf_text += mode_info
        perf_text += f"CPU: {cpu_current:.1f}% (avg:{cpu_avg:.1f}% max:{cpu_max:.1f}%)\n"

        if cpu_stats['history']:
            perf_text += self._create_sparkline(cpu_stats['history'], 0, 100)

        perf_text += f"Mem: {memory_stats['current']:.0f}MB (avg:{memory_stats['avg']:.0f}MB max:{memory_stats['max']:.0f}MB)\n"

        if memory_stats['history']:
            max_mem = max(memory_stats['history']) if memory_stats['history'] else 100
            perf_text += self._create_sparkline(memory_stats['history'], 0, max_mem * 1.1)

        fps_list = []
        for service_name in ['Camera', 'Face Detection', 'Gesture Detection', 'People Segmentation']:
            fps = self.performance.get_fps(service_name)
            if fps:
                short_name = service_name.replace(' Detection', '').replace(' Segmentation', '').replace('People ', '')
                fps_list.append(f"{short_name}:{fps:.1f}")

        if fps_list:
            perf_text += f"Services: {', '.join(fps_list)}\n"
        
        # Add GPU acceleration status
        try:
            from gpu_utils import get_gpu_detector
            gpu_detector = get_gpu_detector()
            if gpu_detector.is_gpu_available():
                gpu_backends = [k.replace('_', ' ').title() for k, v in gpu_detector.gpu_info.items() if v]
                perf_text += f"GPU: {', '.join(gpu_backends)}\n"
            else:
                perf_text += "GPU: CPU only\n"
        except ImportError:
            pass  # GPU utils not available
        except Exception:
            pass  # GPU detection failed

        return perf_text
    
    def _create_cooldowns_panel(self) -> str:
        """Create cooldowns panel"""
        if not self.mode:
            return "Mode not available"
        
        try:
            cooldown_info = self.mode.get_cooldown_info()
            if not cooldown_info:
                return "No cooldown info available"
            
            cooldown_text = ""
            # Display cooldowns in order: eye, people, wave, smile, thumbs_up
            status_order = ['eye', 'people', 'wave', 'smile', 'thumbs_up']
            for status in status_order:
                if status in cooldown_info:
                    info = cooldown_info[status]
                    remaining = info['remaining']
                    progress = info['progress']
                    
                    # Format remaining time
                    if remaining > 0:
                        if remaining >= 60:
                            time_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                        else:
                            time_str = f"{int(remaining)}s"
                    else:
                        time_str = "ready"
                    
                    # Create progress bar (inverted - shows remaining cooldown)
                    # progress=1.0 means cooldown complete, progress=0.0 means just started
                    # We want to show remaining, so use (1.0 - progress)
                    remaining_progress = 1.0 - progress
                    bar = self._create_progress_bar(remaining_progress, width=16)
                    
                    # Status name formatting
                    status_display = status.replace('_', ' ').title()
                    
                    cooldown_text += f"{status_display:12} [{bar}] {time_str}\n"
            
            return cooldown_text.strip() if cooldown_text else "No cooldowns"
        except Exception:
            return "Error loading cooldowns"
    
    def _create_status_tree_panel(self):
        """Create status transition tree panel showing current status and possible transitions"""
        if not self.mode:
            return Panel("Mode not available", title="Status Transitions", border_style="magenta")
        
        try:
            # Get current status and state info
            current_status = self.mode.get_status()
            state_info = self.mode.get_state_info()
            status_info = self.mode.get_status_info() if hasattr(self.mode, 'get_status_info') else None
            
            if not state_info:
                return Panel("No state info available", title="Status Transitions", border_style="magenta")
            
            # Get available statuses (excluding current)
            available_statuses = state_info.get('available_statuses', [])
            main_statuses = state_info.get('main_statuses', [])
            interactive_statuses = state_info.get('interactive_statuses', [])
            previous_status = state_info.get('previous_status')
            
            # Build tree visualization using Rich Text
            tree_text = Text()
            
            # Show previous status if exists
            if previous_status and previous_status != current_status:
                tree_text.append(f"┌─ {previous_status.replace('_', ' ').title()}\n", style="dim")
                tree_text.append("│\n", style="dim")
                tree_text.append("└─→ ", style="dim")
            else:
                tree_text.append("┌─ ", style="dim")
            
            # Current status (highlighted)
            current_display = current_status.replace('_', ' ').title()
            tree_text.append(current_display, style="bold cyan")
            
            # Add remaining time and progress bar if available
            if status_info and status_info.get('remaining') is not None:
                remaining = status_info['remaining']
                duration = status_info.get('elapsed', 0) + remaining if remaining else None
                
                if duration and duration > 0:
                    progress = status_info['elapsed'] / duration
                    progress = max(0.0, min(1.0, progress))
                    
                    # Format remaining time
                    if remaining >= 60:
                        time_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                    else:
                        time_str = f"{int(remaining)}s"
                    
                    # Create progress bar
                    bar = self._create_progress_bar(progress, width=20)
                    tree_text.append(f" ({time_str})", style="yellow")
                    tree_text.append(f" [{bar}]", style="dim")
            
            tree_text.append("\n")
            
            # Show possible transitions
            if available_statuses:
                # Filter out current status
                transitions = [s for s in available_statuses if s != current_status]
                
                if transitions:
                    tree_text.append("│\n", style="dim")
                    tree_text.append("├─ Possible transitions:\n", style="dim")
                    
                    # Group by main vs interactive
                    main_transitions = [s for s in transitions if s in main_statuses]
                    interactive_transitions = [s for s in transitions if s in interactive_statuses]
                    other_transitions = [s for s in transitions if s not in main_statuses and s not in interactive_statuses]
                    
                    # Show main status transitions
                    if main_transitions:
                        for i, status in enumerate(main_transitions):
                            is_last = (i == len(main_transitions) - 1 and not interactive_transitions and not other_transitions)
                            prefix = "└─" if is_last else "├─"
                            status_display = status.replace('_', ' ').title()
                            tree_text.append(prefix, style="dim")
                            tree_text.append(f" {status_display}", style="green")
                            tree_text.append(" (main)\n", style="dim")
                    
                    # Show interactive status transitions
                    if interactive_transitions:
                        for i, status in enumerate(interactive_transitions):
                            is_last = (i == len(interactive_transitions) - 1 and not other_transitions)
                            prefix = "└─" if is_last else "├─"
                            status_display = status.replace('_', ' ').title()
                            tree_text.append(prefix, style="dim")
                            tree_text.append(f" {status_display}", style="yellow")
                            tree_text.append(" (interactive)\n", style="dim")
                    
                    # Show other transitions
                    if other_transitions:
                        for i, status in enumerate(other_transitions):
                            is_last = (i == len(other_transitions) - 1)
                            prefix = "└─" if is_last else "├─"
                            status_display = status.replace('_', ' ').title()
                            tree_text.append(f"{prefix} {status_display}\n", style="dim")
                else:
                    tree_text.append("│\n", style="dim")
                    tree_text.append("└─ No transitions available\n", style="dim")
            else:
                tree_text.append("│\n", style="dim")
                tree_text.append("└─ No status info\n", style="dim")
            
            # Show transition rules if any
            transition_rules = state_info.get('transition_rules', [])
            if transition_rules:
                tree_text.append("\nTransition rules:\n", style="dim")
                for rule in transition_rules[:3]:  # Show max 3 rules
                    rule_type = rule.get('type', 'unknown')
                    if rule_type == 'time':
                        interval = rule.get('interval', 0)
                        tree_text.append(f"  • Time-based: every {interval}s\n", style="dim")
                    elif rule_type == 'random':
                        interval = rule.get('interval', 0)
                        tree_text.append(f"  • Random: every {interval}s\n", style="dim")
            
            return Panel(tree_text, title="Status Transitions", border_style="magenta", padding=(0, 1))
        except Exception as e:
            return Panel(f"Error: {str(e)[:50]}", title="Status Transitions", border_style="magenta")

    def _create_sparkline(self, data: List[float], min_val: float, max_val: float, width: int = 20) -> str:
        """Create a simple sparkline graph using Unicode block characters"""
        if not data:
            return "  (no data)\n"

        # Keep graph narrow enough for side panels
        try:
            width = max(10, min(width, self.console.width // 3))
        except Exception:
            width = max(10, width)

        # Handle case where all values are the same or range is zero
        if max_val <= min_val or len(set(data)) == 1:
            # Return a flat line using middle block character
            return "  " + "▄" * min(width, len(data)) + "\n"

        # Normalize data to 0-1 range
        normalized = [(v - min_val) / (max_val - min_val) for v in data]
        normalized = [max(0.0, min(1.0, v)) for v in normalized]  # Clamp

        # Use Unicode block characters for better visualization
        blocks = "▁▂▃▄▅▆▇█"
        block_count = len(blocks)

        # Sample data if too long (ensure it fits exactly)
        if len(normalized) > width:
            step = len(normalized) / width
            sampled = [normalized[int(i * step)] for i in range(width)]
        else:
            sampled = normalized

        # Create sparkline (ensure it fits on one line exactly)
        sparkline = "  "
        for val in sampled:
            idx = int(val * (block_count - 1))
            idx = max(0, min(block_count - 1, idx))  # Ensure valid index
            sparkline += blocks[idx]

        return sparkline + "\n"

    def _create_input_panel(self) -> Panel:
        """Create input display panel - shows prompt and instructions"""
        with self.input_lock:
            # Show last command entered, or prompt if none
            if self.current_input:
                display_text = f"Last: {self.current_input}"
            else:
                display_text = "Press ':' to enter command"
        
        # Truncate if too long to prevent overflow
        max_len = min(60, self.terminal_width - 20)
        if len(display_text) > max_len:
            display_text = display_text[:max_len-3] + "..."
        
        content = f"Kaleidoscape> {display_text}"
        return Panel(content, title="Input (Press ':' for command)", border_style="cyan", padding=(0, 1))

    def _create_layout(self) -> Layout:
        """Create the main layout using Rich defaults (header + status + performance + logs + input)"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="input", size=3)  # Input panel at bottom
        )

        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="logs", ratio=2),
        )

        layout["left"].split_column(
            Layout(name="status", ratio=1),
            Layout(name="middle"),
        )
        
        layout["middle"].split_column(
            Layout(name="performance", ratio=1),
            Layout(name="bottom_middle"),
        )
        
        layout["bottom_middle"].split_row(
            Layout(name="cooldowns", ratio=1),
            Layout(name="status_tree", ratio=1),
        )

        # Header
        title = Text("KALEIDOSCAPE", style="bold magenta")
        subtitle = Text("Service Monitoring & Debugging Console", style="dim")
        layout["header"].update(
            Panel(Align.center(title + "\n" + subtitle), border_style="magenta")
        )

        # Status table
        status_table = self._create_status_table()
        layout["status"].update(Panel(status_table, title="Service Status", border_style="cyan"))

        # Performance panel
        perf_content = self._create_performance_panel()
        layout["performance"].update(Panel(perf_content, title="Performance Metrics", border_style="green"))

        # Cooldowns panel
        cooldown_content = self._create_cooldowns_panel()
        layout["cooldowns"].update(Panel(cooldown_content, title="Status Cooldowns", border_style="yellow"))

        # Status tree panel
        status_tree_panel = self._create_status_tree_panel()
        layout["status_tree"].update(status_tree_panel)

        # Logs panel
        layout["logs"].update(self._create_logs_panel())

        # Input panel at bottom
        layout["input"].update(self._create_input_panel())

        return layout

    def _cmd_set_status(self, status: str):
        """Set mode status"""
        if not self.mode:
            self.log("UI", "ERROR", "Mode not available")
            return

        if status in ['eye', 'people', 'wave', 'smile', 'thumbs_up']:
            success = self.mode.set_status(status)
            if success:
                self.log("UI", "INFO", f"Status set to: {status}")
            else:
                # Check cooldown info to show remaining time
                is_on_cooldown, remaining = self.mode.is_status_on_cooldown(status)
                if is_on_cooldown:
                    if remaining >= 60:
                        time_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                    else:
                        time_str = f"{int(remaining)}s"
                    self.log("UI", "WARNING", f"Status '{status}' is on cooldown. {time_str} remaining.")
                else:
                    self.log("UI", "ERROR", f"Failed to set status '{status}'")
        else:
            self.log("UI", "ERROR", f"Invalid status: {status}. Use 'eye', 'people', 'wave', 'smile', or 'thumbs_up'")

    def _cmd_trigger_wave(self):
        """Trigger wave gesture"""
        if not self.mode:
            self.log("UI", "ERROR", "Mode not available")
            return

        # Calculate duration
        duration = None
        if self.mode.video_manager.has_video('hand_waving'):
            video_duration = self.mode.video_manager.get_duration('hand_waving')
            if video_duration > 0:
                hai_scroll_time = 5.0 + 1.0  # Scroll time + buffer
                duration = video_duration + 0.5 + hai_scroll_time
        
        success = self.mode.set_status('wave', duration=duration, substate='hand_waving')
        if success:
            self.log("UI", "INFO", "Wave gesture triggered")
        else:
            # Check cooldown
            is_on_cooldown, remaining = self.mode.is_status_on_cooldown('wave')
            if is_on_cooldown:
                if remaining >= 60:
                    time_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                else:
                    time_str = f"{int(remaining)}s"
                self.log("UI", "WARNING", f"Wave is on cooldown. {time_str} remaining.")
            else:
                self.log("UI", "ERROR", "Failed to trigger wave gesture")

    def _cmd_trigger_thumbs_up(self):
        """Trigger thumbs up gesture"""
        if not self.mode:
            self.log("UI", "ERROR", "Mode not available")
            return

        # Calculate duration
        duration = None
        if self.mode.video_manager.has_video('thumbs_up'):
            video_duration = self.mode.video_manager.get_duration('thumbs_up')
            if video_duration > 0:
                duration = video_duration + 0.2
        
        success = self.mode.set_status('thumbs_up', duration=duration)
        if success:
            self.log("UI", "INFO", "Thumbs up gesture triggered")
        else:
            # Check cooldown
            is_on_cooldown, remaining = self.mode.is_status_on_cooldown('thumbs_up')
            if is_on_cooldown:
                if remaining >= 60:
                    time_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                else:
                    time_str = f"{int(remaining)}s"
                self.log("UI", "WARNING", f"Thumbs up is on cooldown. {time_str} remaining.")
            else:
                self.log("UI", "ERROR", "Failed to trigger thumbs up gesture")

    def _cmd_trigger_smile(self):
        """Trigger smile gesture"""
        if not self.mode:
            self.log("UI", "ERROR", "Mode not available")
            return

        # Calculate duration
        duration = None
        if self.mode.video_manager.has_video('smile'):
            video_duration = self.mode.video_manager.get_duration('smile')
            if video_duration > 0:
                scroll_time = 8.0
                duration = video_duration + scroll_time + 1.0
        
        success = self.mode.set_status('smile', duration=duration)
        if success:
            self.log("UI", "INFO", "Smile gesture triggered")
        else:
            # Check cooldown
            is_on_cooldown, remaining = self.mode.is_status_on_cooldown('smile')
            if is_on_cooldown:
                if remaining >= 60:
                    time_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                else:
                    time_str = f"{int(remaining)}s"
                self.log("UI", "WARNING", f"Smile is on cooldown. {time_str} remaining.")
            else:
                self.log("UI", "ERROR", "Failed to trigger smile gesture")

    def _cmd_switch_visual(self, *args):
        """Switch to a different Hydra visual/sketch
        
        Usage:
            switch_visual [sketch_name]
            If sketch_name is provided, switches to that sketch (partial match allowed)
            If omitted, chooses a random sketch
        """
        if not self.mode:
            self.log("UI", "ERROR", "Mode not available")
            return
        
        # Check if mode supports visual switching
        if not hasattr(self.mode, 'switch_visual'):
            self.log("UI", "ERROR", "Current mode does not support visual switching")
            return
        
        # Join all args to handle sketch names with spaces
        sketch_arg = ' '.join(args) if args else None
        success = self.mode.switch_visual(sketch=sketch_arg)
        
        if success:
            if sketch_arg:
                self.log("UI", "INFO", f"Switched to sketch: {sketch_arg}")
            else:
                self.log("UI", "INFO", "Switched to random sketch")
        else:
            self.log("UI", "ERROR", "Failed to switch visual")
    
    def _cmd_switch_color(self, mode=None):
        """Switch color transformation mode
        
        Usage:
            switch_color <mode>
            switch_color 1  - RGB interpolation mode
            switch_color 2  - Closest color mode
            switch_color 3  - Two color mode
            switch_color 4  - TBD mode
            switch_color 0  - Disable color transformation
        """
        if not self.mode:
            self.log("UI", "ERROR", "Mode not available")
            return
        
        # Check if mode supports color switching
        if not hasattr(self.mode, 'set_color_mode'):
            self.log("UI", "ERROR", "Current mode does not support color mode switching")
            return
        
        if not mode:
            self.log("UI", "ERROR", "Usage: switch_color <mode> (0-4)")
            self.log("UI", "INFO", "Modes: 0=disabled, 1=RGB interpolation, 2=closest color, 3=two colors, 4=TBD")
            return
        
        try:
            mode_int = int(mode)
            if mode_int < 0 or mode_int > 4:
                self.log("UI", "ERROR", "Color mode must be between 0 and 4")
                return
            
            self.mode.set_color_mode(mode_int)
            mode_names = {
                0: "disabled",
                1: "RGB interpolation",
                2: "closest color",
                3: "two colors",
                4: "TBD"
            }
            self.log("UI", "INFO", f"Color mode switched to {mode_int} ({mode_names.get(mode_int, 'unknown')})")
        except ValueError:
            self.log("UI", "ERROR", f"Invalid color mode: {mode}. Must be a number (0-4)")

    def _cmd_start_service(self, service_name: str = None):
        """Start a service"""
        if not service_name:
            self.log("UI", "ERROR", "Usage: start <service_name>")
            self.log("UI", "INFO", "Available services: camera, face, gesture, segmentation, tts, stt")
            return

        service_name_lower = service_name.lower()
        service_map = {
            'camera': 'Camera',
            'face': 'Face Detection',
            'gesture': 'Gesture Detection',
            'segmentation': 'People Segmentation',
            'tts': 'Audio (TTS)',
            'stt': 'Audio (STT)',
        }

        mapped_name = service_map.get(service_name_lower)
        if not mapped_name:
            self.log("UI", "ERROR", f"Unknown service: {service_name}")
            return

        if not self.mode:
            self.log("UI", "ERROR", "Mode not available")
            return

        # Enable the service
        if mapped_name in self.services:
            self.services[mapped_name].enabled = True
            self.log("UI", "INFO", f"Service '{mapped_name}' enabled")

        # Actually start/restart the service based on type
        if service_name_lower == 'camera':
            if not self.mode.camera_running:
                # Restart camera thread if camera exists
                if self.mode.camera:
                    self.mode.camera_running = True
                    import threading
                    self.mode.camera_thread = threading.Thread(target=self.mode._camera_loop, daemon=True)
                    self.mode.camera_thread.start()
                    self.log("UI", "INFO", "Camera started")
            else:
                self.log("UI", "INFO", "Camera already running")

        elif service_name_lower == 'stt':
            if not self.mode.audio_service.stt_enabled:
                self.mode.audio_service.initialize_stt()
                self.log("UI", "INFO", "Speech-to-text started")
            else:
                self.log("UI", "INFO", "Speech-to-text already running")

        elif service_name_lower in ['face', 'gesture', 'segmentation']:
            # These are controlled by enable flag, which is already set above
            self.log("UI", "INFO", f"Service '{mapped_name}' will be enabled on next camera frame")

    def _cmd_stop_service(self, service_name: str = None):
        """Stop a service"""
        if not service_name:
            self.log("UI", "ERROR", "Usage: stop <service_name>")
            return

        service_name_lower = service_name.lower()
        service_map = {
            'camera': 'Camera',
            'face': 'Face Detection',
            'gesture': 'Gesture Detection',
            'segmentation': 'People Segmentation',
            'tts': 'Audio (TTS)',
            'stt': 'Audio (STT)',
        }

        mapped_name = service_map.get(service_name_lower)
        if not mapped_name:
            self.log("UI", "ERROR", f"Unknown service: {service_name}")
            return

        # Disable the service - this is the source of truth
        if mapped_name in self.services:
            self.services[mapped_name].enabled = False
            self.log("UI", "INFO", f"Service '{mapped_name}' disabled")
            
            # For segmentation, also log that processing will stop
            if mapped_name == 'People Segmentation':
                self.log("UI", "INFO", "Segmentation will stop processing frames on next check")

        # Actually stop the service
        if service_name_lower == 'camera':
            self.mode.camera_running = False
            self.log("UI", "INFO", "Camera stopped")
        
        elif service_name_lower == 'face':
            # Clear face position when stopping face detection
            with self.mode.face_detection_lock:
                self.mode.target_face_position = None
                self.mode.detected_faces = []
            self.log("UI", "INFO", "Face detection stopped - face position cleared")

        elif service_name_lower == 'segmentation':
            # The enabled flag is already set to False above
            # The mode checks this flag each frame in _camera_loop()
            # So when enabled=False, detection will stop on the next frame check
            
            # Clear the mask buffer immediately so old masks don't persist
            self.mode.mask_detector_module.mask_buffer.clear()
            self.log("UI", "INFO", "People segmentation buffer cleared - processing will stop on next frame")
            
            # Verify the enabled flag is set
            seg_service = self.services.get('People Segmentation')
            if seg_service:
                self.log("UI", "INFO", f"Segmentation enabled flag: {seg_service.enabled} (should be False)")

        elif service_name_lower == 'stt':
            if self.mode.audio_service.speech_to_text:
                self.mode.audio_service.speech_to_text.stop_listening()
                self.log("UI", "INFO", "Speech-to-text stopped")

    def _cmd_enable_service(self, service_name: str = None):
        """Enable a service (alias for start)"""
        self._cmd_start_service(service_name)

    def _cmd_disable_service(self, service_name: str = None):
        """Disable a service (alias for stop)"""
        self._cmd_stop_service(service_name)

    def _cmd_help(self):
        """Show help"""
        help_text = """
Available Commands:

Mode Control:
  status <eye|people>      - Set mode status (eye or people)
  eye                     - Shortcut for 'status eye'
  people                  - Shortcut for 'status people'
  wave                    - Trigger wave gesture
  thumbs_up               - Trigger thumbs up gesture
  smile                   - Trigger smile gesture

Service Control:
  start <service>         - Start/Enable a service
  stop <service>          - Stop/Disable a service
  enable <service>        - Alias for 'start <service>'
  disable <service>       - Alias for 'stop <service>'

Available Services:
  camera                  - Camera capture
  face                    - Face detection
  gesture                 - Gesture detection
  segmentation            - People segmentation
  tts                     - Text-to-speech (TTS)
  stt                     - Speech-to-text (STT)

Examples:
  start camera            - Start camera service
  stop face               - Stop face detection
  enable gesture          - Enable gesture detection
  disable segmentation    - Disable people segmentation

Visual Control (Mischief Mode):
  switch_visual [sketch] - Switch to a different Hydra visual/sketch
                           If sketch name is provided, switches to that sketch (partial match)
                           If omitted, chooses a random sketch
  switch_color <mode>     - Switch color transformation mode
  color_mode <mode>       - Alias for switch_color
                           Modes: 0=disabled, 1=RGB interpolation, 2=closest color, 3=two colors, 4=TBD

Other:
  help                    - Show this help
  quit / exit             - Exit console
"""
        self.log("UI", "INFO", help_text)

    def _process_command(self, command: str):
        """Process a command"""
        parts = command.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if cmd == 'quit' or cmd == 'exit':
            self.running = False
            return

        if cmd in self.command_handlers:
            try:
                if args:
                    self.command_handlers[cmd](*args)
                else:
                    self.command_handlers[cmd]()
            except Exception as e:
                self.log("UI", "ERROR", f"Command error: {e}")
        else:
            self.log("UI", "WARNING", f"Unknown command: {cmd}. Type 'help' for available commands")

    def _update_loop(self):
        """Update loop for the UI"""
        while self.running:
            try:
                # Update service statuses from mode
                if self.mode:
                    self._update_mode_status()

                # Update performance metrics
                if self.client and self.client.current_mode:
                    # Track overall display FPS
                    self.track_performance('LED Display')

                # Update system monitoring
                self.system_monitor.update()

                time.sleep(0.1)  # Update UI 10 times per second
            except Exception as e:
                self.log("UI", "ERROR", f"Update error: {e}")

    def _update_mode_status(self):
        """Update status from mode"""
        if not self.mode:
            return

        # Update camera status
        camera_enabled = self.services.get('Camera', ServiceStatus(name='Camera')).enabled
        if self.mode.camera and camera_enabled:
            if self.mode.camera_running:
                self.update_service_status('Camera', 'Running', 'Active')
                self.track_performance('Camera')
            else:
                self.update_service_status('Camera', 'Stopped', 'Disabled')
        else:
            status = 'Disabled' if not camera_enabled else 'Not initialized'
            self.update_service_status('Camera', 'Stopped', status)

        # Update face detection
        face_enabled = self.services.get('Face Detection', ServiceStatus(name='Face Detection')).enabled
        if self.mode.face_detector_module and face_enabled:
            self.update_service_status('Face Detection', 'Running', 'Active')
            self.track_performance('Face Detection')
        else:
            status = 'Disabled' if not face_enabled else 'Not initialized'
            self.update_service_status('Face Detection', 'Stopped', status)

        # Update gesture detection
        gesture_enabled = self.services.get('Gesture Detection', ServiceStatus(name='Gesture Detection')).enabled
        if self.mode.gesture_detector_module and gesture_enabled:
            status = 'Running'
            details = 'AI + Manual'
            if not self.mode.gesture_detector_module.gesture_recognizer_available:
                details = 'Manual only'
            self.update_service_status('Gesture Detection', status, details)
            self.track_performance('Gesture Detection')
        else:
            status_text = 'Disabled' if not gesture_enabled else 'Not initialized'
            self.update_service_status('Gesture Detection', 'Stopped', status_text)

        # Update people segmentation
        segmentation_enabled = self.services.get('People Segmentation', ServiceStatus(name='People Segmentation')).enabled
        if self.mode.mask_detector_module and segmentation_enabled:
            self.update_service_status('People Segmentation', 'Running', 'Active')
            self.track_performance('People Segmentation')
        else:
            status = 'Disabled' if not segmentation_enabled else 'Not initialized'
            self.update_service_status('People Segmentation', 'Stopped', status)

        # Update video manager
        video_count = len(self.mode.video_config)
        self.update_service_status('Video Manager', 'Running', f'{video_count} videos loaded')

        # Update audio services
        self.update_service_status('Audio (TTS)', 'Running', 'Piper TTS')
        self.update_service_status('Audio (STT)', 'Running', 'Whisper STT')
        
        # Update AI service (LLM)
        if hasattr(self.mode, 'llm_command_handler') and self.mode.llm_command_handler:
            llm_handler = self.mode.llm_command_handler
            if llm_handler.use_llm:
                # Check if LLM is available by checking if thread is running
                if llm_handler.llm_thread and llm_handler.llm_thread.is_alive():
                    # Count pending requests
                    with llm_handler.pending_lock:
                        pending_count = len([k for k, v in llm_handler.pending_requests.items() if v.get('result') is None])
                    details = f"Ollama ({llm_handler.model_name})"
                    if pending_count > 0:
                        details += f" - {pending_count} pending"
                    self.update_service_status('AI Service', 'Running', details)
                else:
                    self.update_service_status('AI Service', 'Stopped', 'Thread not running')
            else:
                self.update_service_status('AI Service', 'Stopped', 'LLM disabled')
        else:
            self.update_service_status('AI Service', 'Stopped', 'Not initialized')

    def run(self):
        """Run the console UI"""
        if not RICH_AVAILABLE:
            self.console.print("[red]Error: rich library not available[/red]")
            return

        self.running = True

        # Start update thread silently
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        # Small delay to ensure thread starts
        time.sleep(0.1)

        # Initial log (will be shown in logs panel)
        try:
            self.log("UI", "INFO", "Kaleidoscape console started")
        except Exception:
            pass

        # Main UI loop - Live widget with separate input thread
        try:

            # Input queue for thread-safe command passing
            input_queue = queue.Queue()

            # Input thread - detects ':' key press without interfering with Rich's Prompt
            # This thread only reads when Live is NOT paused (i.e., when Prompt is not active)
            def input_thread():
                """Background thread that detects ':' key to trigger command prompt"""
                import select
                import termios
                import tty
                
                old_settings = None
                try:
                    if sys.stdin.isatty():
                        old_settings = termios.tcgetattr(sys.stdin)
                        tty.setcbreak(sys.stdin.fileno())
                except Exception:
                    old_settings = None
                
                try:
                    while self.running:
                        # Only check for ':' when Live is NOT paused
                        # When paused, Rich's Prompt is handling input, so we restore terminal and wait
                        if self.pause_live.is_set():
                            # Restore terminal to normal mode so Rich's Prompt can work
                            if old_settings is not None:
                                try:
                                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                                except Exception:
                                    pass
                            time.sleep(0.1)
                        elif old_settings is not None:
                            # Live is not paused - check for ':' key
                            try:
                                # Restore cbreak mode (in case it was changed)
                                try:
                                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                                    tty.setcbreak(sys.stdin.fileno())
                                except Exception:
                                    pass
                                
                                # Non-blocking check for input
                                if select.select([sys.stdin], [], [], 0.1)[0]:
                                    char = sys.stdin.read(1)
                                    if char == ':':
                                        # Trigger command prompt
                                        self.command_pending.set()
                                    # If it's not ':', we lose the character
                                    # This is a limitation, but necessary to detect ':'
                            except Exception:
                                pass
                finally:
                    # Restore terminal settings on exit
                    if old_settings is not None:
                        try:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        except Exception:
                            pass

            # Start input thread
            input_thread_obj = threading.Thread(target=input_thread, daemon=True)
            input_thread_obj.start()

            def make_layout():
                return self._create_layout()

            # Disable terminal auto-wrap while the Live view is active
            try:
                self.console.control("\x1b[?7l")  # DECAWM off
            except Exception:
                pass

            # Use screen=True to clear screen and position at top
            # For command input, we'll pause Live and use Rich's Prompt directly
            # This avoids stdin conflicts - Rich handles all terminal I/O
            with Live(make_layout(), refresh_per_second=2, screen=True, console=self.console, 
                     auto_refresh=False) as live:
                while self.running:
                    try:
                        # Check if command prompt was triggered
                        if self.command_pending.is_set():
                            self.command_pending.clear()
                            
                            # Pause Live updates FIRST - this stops the background thread from reading
                            self.pause_live.set()
                            
                            # Small delay to ensure background thread stops reading and restores terminal
                            time.sleep(0.15)
                            
                            # Render current layout one last time before input
                            live.update(make_layout())
                            live.refresh()
                            
                            try:
                                # Use console.input() directly - this should show characters as you type
                                # The terminal is in normal mode now, so input should be visible
                                self.console.print("\n[cyan]Kaleidoscape[/cyan]: ", end="")
                                command = self.console.input()
                                
                                if command and command.strip():
                                    # Update input display
                                    with self.input_lock:
                                        self.current_input = command.strip()
                                    
                                    # Process the command
                                    self._process_command(command.strip())
                                    
                                    # Clear input after processing
                                    with self.input_lock:
                                        self.current_input = ""
                            except (EOFError, KeyboardInterrupt):
                                # User cancelled or interrupted
                                pass
                            except Exception as e:
                                self.log("UI", "ERROR", f"Command input error: {e}")
                            
                            # Resume Live updates - background thread can resume checking for ':'
                            self.pause_live.clear()
                            # Refresh layout after input
                            live.update(make_layout())
                            live.refresh()
                        
                        # Only update Live if not paused
                        # When paused, Live updates are stopped so input is visible
                        if not self.pause_live.is_set():
                            # Update the display
                            live.update(make_layout())
                            live.refresh()
                        else:
                            # When paused, don't update Live at all - this keeps input visible
                            # The pause_live flag prevents any updates
                            pass

                        # Check for queued commands (legacy support)
                        try:
                            command = input_queue.get_nowait()
                            if command:
                                self._process_command(command)
                        except queue.Empty:
                            pass

                        time.sleep(0.2)  # Check more frequently for ':' key
                    except KeyboardInterrupt:
                        self.running = False
                        break
                    except Exception as e:
                        try:
                            self.log("UI", "ERROR", f"Update error: {e}")
                        except Exception:
                            pass
                        time.sleep(0.5)
        except Exception as e:
            # Log error but don't print to console (would interfere)
            try:
                self.log("UI", "ERROR", f"Console UI error: {e}")
            except Exception:
                pass
            # Fallback mode - simple input loop
            while self.running:
                try:
                    command = input("\nKaleidoscape> ")
                    if command:
                        self._process_command(command)
                except (EOFError, KeyboardInterrupt):
                    self.running = False
                    break
        finally:
            # Re-enable terminal auto-wrap
            try:
                self.console.control("\x1b[?7h")  # DECAWM on
            except Exception:
                pass

            try:
                self.log("UI", "INFO", "Kaleidoscape console stopped")
            except Exception:
                pass