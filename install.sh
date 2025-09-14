#!/bin/bash

echo "Installing Piano Finger Game dependencies..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install requirements
echo "Installing Python packages..."
pip3 install -r requirements.txt

echo "Installation complete!"
echo "Run the game with: python3 piano_game.py"

