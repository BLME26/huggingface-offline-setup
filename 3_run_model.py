#!/usr/bin/env python3
"""
Run Hugging Face model inference offline (no internet required)
IMPROVED: Better GPU detection, padding support, comprehensive logging
"""

import os
import sys
from pathlib import Path

# Enable offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_gpu():
    """Check and report GPU status"""
    print("\n🖥️  GPU/Device Detection:")
    print("-" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_mem:.1f} GB")
        device = "cuda"
        print("✓ GPU will be used for inference")
    else:
        print("⚠️  GPU not detected, using CPU")
        device = "cpu"
        print("  (Check: nvidia-smi and torch.cuda.is_available())")
    
    return device

def find_model_path():
    """Find the downloaded model directory"""
    model_dir = Path(__file__).parent / "models"
    
    if not model_dir.exists():
        print("✗ No models/ directory found")
        return None
    
    # Find model subdirectory
    model_subdirs = list(model_dir.glob("models--*"))
    if not model_subdirs:
        print(f"✗ No model found in {model_dir}")
        print("  Did you run: python 1_download_on_internet.py?")
        return None
    
    model_path = model_subdirs[0] / "snapshots"
    snapshot_dirs = list(model_path.glob("*"))
    
    if not snapshot_dirs:
        print("✗ No model snapshot found")
        return None
    
    return snapshot_dirs[0]

def main():
    print("=" * 60)
    print("Hugging Face Offline Inference")
    print("=" * 60)
    
    # Check GPU
    device = check_gpu()
    
    # Find model
    model_path = find_model_path()
    if not model_path:
        sys.exit(1)
    
    print(f"\n📂 Model path: {model_path}")
    
    try:
        # Load model and tokenizer
        print("\n⏳ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # FIX: Set pad token if not set (fixes batch processing issues)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  ✓ Set pad_token to eos_token")
        
        print("⏳ Loading model...")
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
        
        # Move to device
        print(f"Moving model to {device.upper()}...")
        model.to(device)
        model.eval()
        
        # Test inference
        print("\n✓ Model loaded successfully!")
        print("\n" + "=" * 60)
        print("Running inference test...")
        print("=" * 60)
        
        test_prompt = "The future of artificial intelligence is"
        print(f"\n📝 Prompt: {test_prompt}")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=1,
                temperature=0.7,
                top_p=0.95
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n📤 Output:\n{result}")
        
        print("\n" + "=" * 60)
        print(f"✓ Inference test completed successfully!")
        print(f"✓ Model running on: {device.upper()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
