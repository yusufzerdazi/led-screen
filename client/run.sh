#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
fi

# Audio dependencies removed for music visualizer

# Start local Hydra instance
echo "Starting local Hydra instance..."
cd "$SCRIPT_DIR/../../hydra" && npm run dev &

# Wait for Hydra to start up
sleep 5

# Return to script directory
cd "$SCRIPT_DIR"

# Start the music visualizer
# All parameters are forwarded to the Python script
echo "Starting music visualizer..."
python3 client.py --mode tush "$@" &

# Wait a moment to ensure services are running
sleep 2

echo "All services started. Press Ctrl+C to stop."

# Wait for all background processes
wait 
