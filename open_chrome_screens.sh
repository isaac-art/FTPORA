#!/bin/bash

# Open two Chrome windows to the specified URLs on Linux
URL1="http://localhost:8000/screen1"
URL2="http://localhost:8000/screen2"

# Try google-chrome, fallback to chromium-browser
if command -v google-chrome > /dev/null; then
    google-chrome --new-window --kiosk --window-position=0,0 "$URL1" &
    sleep 1
    google-chrome --new-window --kiosk --window-position=1920,0 "$URL2" &
else
    echo "Neither google-chrome nor chromium-browser is installed."
    exit 1
fi 