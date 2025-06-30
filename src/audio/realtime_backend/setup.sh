#!/usr/bin/env bash
# Setup script for compiling the realtime_backend Rust crate on Windows 11
# This installs the Rust toolchain, build tools, Python dependencies, and
# builds the Python extension using maturin.

set -e

# Ensure script is run from the repository root or this directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Rust via winget if not present
if ! command_exists rustup; then
    echo "Rustup not found. Installing via winget..."
    if command_exists winget; then
        winget install -e --id Rustlang.Rustup || {
            echo "Failed to install rustup with winget. Install it manually from https://rustup.rs."; exit 1; }
    else
        echo "winget is not available. Please install rustup from https://rustup.rs"; exit 1
    fi
fi

# Update toolchain and ensure stable channel
rustup update
rustup default stable

# Install Visual Studio Build Tools if MSVC compiler is unavailable
if ! command_exists cl; then
    echo "MSVC build tools not found. Installing via winget..."
    if command_exists winget; then
        winget install -e --id Microsoft.VisualStudio.2022.BuildTools || {
            echo "Failed to install Build Tools. Please install them manually."; exit 1; }
    else
        echo "winget is not available. Install Visual Studio Build Tools 2022 manually."; exit 1
    fi
fi

# Ensure Python is installed
if ! command_exists python; then
    echo "Python not found. Installing via winget..."
    if command_exists winget; then
        winget install -e --id Python.Python.3 || {
            echo "Failed to install Python. Install it manually from https://www.python.org"; exit 1; }
    else
        echo "winget is not available. Please install Python 3 manually."; exit 1
    fi
fi

# Upgrade pip and install maturin
python -m pip install --upgrade pip
python -m pip install --upgrade maturin

# Add cargo bin to PATH for the current session
export PATH="$HOME/.cargo/bin:$PATH"

# Build the realtime_backend Python extension
maturin develop --release

echo "Realtime backend built successfully."
