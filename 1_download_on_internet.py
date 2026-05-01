#!/usr/bin/env python3
"""
Download Hugging Face model and Python wheels for offline use
Run this on an internet-connected machine
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).parent

MODELS_DIR = OUTPUT_DIR / "models"
WHEELS_DIR = OUTPUT_DIR / "wheels"

def create_directories():
    """Create necessary directories"""
    MODELS_DIR.mkdir(exist_ok=True)
    WHEELS_DIR.mkdir(exist_ok=True)
    print(f"✓ Created directories: {MODELS_DIR}, {WHEELS_DIR}")

def download_model():
    """Download model and tokenizer"""
    print(f"\n📥 Downloading model: {MODEL_NAME}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODELS_DIR)
        )
        
        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODELS_DIR)
        )
        
        print(f"✓ Model downloaded to: {MODELS_DIR}")
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

def download_wheels():
    """Download Python packages as wheels"""
    print(f"\n📦 Downloading Python wheels to: {WHEELS_DIR}")
    
    requirements_file = OUTPUT_DIR / "requirements.txt"
    
    try:
        cmd = [
            sys.executable, "-m", "pip", "download",
            "-r", str(requirements_file),
            "-d", str(WHEELS_DIR),
            "--python-version", "38",
            "--only-binary=:all:"
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print(f"✓ Wheels downloaded successfully")
        
        # List wheels
        wheels = list(WHEELS_DIR.glob("*.whl")) + list(WHEELS_DIR.glob("*.tar.gz"))
        print(f"\n  Downloaded {len(wheels)} packages:")
        for wheel in sorted(wheels):
            print(f"    - {wheel.name} ({wheel.stat().st_size / (1024*1024):.1f} MB)")
        
        return True
    except Exception as e:
        print(f"✗ Error downloading wheels: {e}")
        return False

def main():
    print("=" * 60)
    print("Hugging Face Model Downloader for Offline Use")
    print("=" * 60)
    
    create_directories()
    success = download_model() and download_wheels()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Copy this entire folder to your air-gapped system (USB drive)")
        print("2. On air-gapped system, run: ./2_install_on_airgap.sh")
        print("3. Then run: python 3_run_model.py")
    else:
        print("\n✗ Download failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
