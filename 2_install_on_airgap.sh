#!/bin/bash

# Installation script for air-gapped system
# This installs all packages without internet access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "======================================================"
echo "Installing packages for offline Hugging Face setup"
echo "======================================================"

if [ ! -d "$WHEELS_DIR" ]; then
    echo "✗ Error: wheels/ directory not found!"
    echo "  Make sure you downloaded files using 1_download_on_internet.py"
    exit 1
fi

echo ""
echo "📦 Installing from: $WHEELS_DIR"
echo ""

python3 -m pip install --no-index --find-links "$WHEELS_DIR" -r requirements.txt

echo ""
echo "======================================================"
echo "✓ Installation complete!"
echo "======================================================"
echo ""
echo "Next: python 3_run_model.py"
