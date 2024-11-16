#!/bin/bash

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

sudo apt install python3-pyaudio

# Start local Hydra instance
echo "Starting local Hydra instance..."
cd /home/yusuf/Code/hydra && npm run dev &

# Wait for Hydra to start up
sleep 5

# Start the camera stream using rpicam-vid
echo "Starting camera stream..."
rpicam-vid -t 0 --width 120 --height 80 --codec h264 --listen -o tcp://0.0.0.0:10001 -n &

# Start the LED client with local Hydra URL
echo "Starting LED client..."
python3 client.py --mode website --website http://localhost:5173?code=czAuaW5pdEltYWdlKCUyMmh0dHAlM0ElMkYlMkZsb2NhbGhvc3QlM0ExMDAwMSUyMiklMEFvc2MoNikubW9kdWxhdGUoc3JjKHMwKSUyQzEpLm91dChvMCk%3D &

# Start the speech recognition listener
echo "Starting speech listener..."
python3 listener.py &

# Wait a moment to ensure services are running
sleep 2

echo "All services started. Press Ctrl+C to stop."

# Wait for all background processes
wait 