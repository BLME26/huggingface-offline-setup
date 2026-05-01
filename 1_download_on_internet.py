#!/usr/bin/env python3
"""
Download any Hugging Face model and Python wheels for offline use
Run this on an internet-connected machine

USAGE:
    python 1_download_on_internet.py                    # Uses gpt2 (default)
    python 1_download_on_internet.py llama2             # Custom model name
"""

import os
import sys
import subprocess
from pathlib import Path

# Default model - change this or pass as argument
DEFAULT_MODEL = "gpt2"

OUTPUT_DIR = Path(__file__).parent
MODELS_DIR = OUTPUT_DIR / "models"
WHEELS_DIR = OUTPUT_DIR / "wheels"

def get_model_name():
    """Get model name from command line or use default"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return DEFAULT_MODEL

def create_directories():
    """Create necessary directories"""
    MODELS_DIR.mkdir(exist_ok=True)
    WHEELS_DIR.mkdir(exist_ok=True)
    print(f"✓ Created directories: {MODELS_DIR}, {WHEELS_DIR}")

def download_model(model_name):
    """Download model and tokenizer"""
    print(f"\n📥 Downloading model: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(MODELS_DIR),
            trust_remote_code=True  # For custom models
        )
        
        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(MODELS_DIR),
            trust_remote_code=True  # For custom models
        )
        
        print(f"✓ Model downloaded to: {MODELS_DIR}")
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\n💡 Tips:")
        print("  - Check model name on huggingface.co/models")
        print("  - Use format: username/model-name")
        print("  - Example: mistralai/Mistral-7B-v0.1")
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
            size_mb = wheel.stat().st_size / (1024*1024)
            print(f"    - {wheel.name} ({size_mb:.1f} MB)")
        
        return True
    except Exception as e:
        print(f"✗ Error downloading wheels: {e}")
        return False

def main():
    model_name = get_model_name()
    
    print("=" * 60)
    print("Hugging Face Model Downloader for Offline Use")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    
    create_directories()
    success = download_model(model_name) and download_wheels()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Copy this entire folder to your air-gapped system (USB drive)")
        print("2. On air-gapped system, run: ./2_install_on_airgap.sh")
        print("3. Then run: python 3_run_model.py")
        print("\n💡 To use a different model on air-gapped system:")
        print("   1. Download it with: python 1_download_on_internet.py model-name")
        print("   2. Transfer to air-gapped system")
    else:
        print("\n✗ Download failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
