#!/bin/bash
# Script to cleanly stop evlib-web and release the webcam

echo "Stopping evlib-web server..."

# Find and kill all evlib-web processes
pkill -f evlib-web

# Give it a moment to clean up
sleep 1

# Force kill if still running
pkill -9 -f evlib-web 2>/dev/null

echo "Server stopped. Webcam should be released."
