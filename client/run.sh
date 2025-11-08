#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any zombie processes that cause timing jitter
pkill chromedriver 2>/dev/null

# Function to handle cleanup on script exit
cleanup() {
    echo "Shutting down services..."
    # Kill all background processes in the current process group
    kill $(jobs -p) 2>/dev/null
    pkill -f rpicam-vid
    pkill -f "npm run dev"
    exit 0
}

# Set up trap for cleanup on script termination
trap cleanup SIGINT SIGTERM

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv --system-site-packages
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
    #pip install -r requirements.txt
fi

# Parse mode from arguments (default to tush for backward compatibility)
MODE="tush"
for arg in "$@"; do
    if [[ "$arg" == "--mode" ]]; then
        MODE_FLAG=true
    elif [[ "$MODE_FLAG" == true ]]; then
        MODE="$arg"
        MODE_FLAG=false
        break
    fi
done

# Start Hydra for modes that need it (tush/music/hydra_mask/mask/decompression)
if [[ "$MODE" == "tush" ]] || [[ "$MODE" == "music" ]] || [[ "$MODE" == "hydra_mask" ]] || [[ "$MODE" == "mask" ]] || [[ "$MODE" == "decompression" ]]; then
    echo "Starting local Hydra instance for $MODE mode..."
    cd "$SCRIPT_DIR/../../hydra" && npm run dev &
    
    # Wait for Hydra to start up
    sleep 5
    
    # Return to script directory
    cd "$SCRIPT_DIR"
    
    echo "Starting $MODE mode..."
    # Check if console mode is enabled
    CONSOLE_MODE=false
    for arg in "$@"; do
        if [[ "$arg" == "--console" ]]; then
            CONSOLE_MODE=true
            break
        fi
    done
    
    # Run in foreground for console mode or hydra_mask/mask modes (to allow stdin input)
    if [[ "$CONSOLE_MODE" == true ]] || [[ "$MODE" == "hydra_mask" ]] || [[ "$MODE" == "mask" ]]; then
        python3 client.py "$@"
    else
        python3 client.py "$@" &
    fi
else
    echo "Starting $MODE mode (Hydra not needed)..."
    # Check if console mode is enabled
    CONSOLE_MODE=false
    for arg in "$@"; do
        if [[ "$arg" == "--console" ]]; then
            CONSOLE_MODE=true
            break
        fi
    done
    
    # Run in foreground for console mode (to allow stdin input)
    if [[ "$CONSOLE_MODE" == true ]]; then
        python3 client.py "$@"
    else
        python3 client.py "$@" &
    fi
fi

# Wait a moment to ensure services are running
sleep 2

echo "All services started. Press Ctrl+C to stop."

# Wait for all background processes
wait 
