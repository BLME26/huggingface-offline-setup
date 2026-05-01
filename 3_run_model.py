#!/usr/bin/env python3
"""
Run Hugging Face model inference offline (no internet required)
"""

import os
import sys
from pathlib import Path

# Enable offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("=" * 60)
    print("Hugging Face Offline Inference")
    print("=" * 60)
    
    # Model directory
    model_dir = Path(__file__).parent / "models"
    
    # Find model subdirectory
    model_subdirs = list(model_dir.glob("models--*"))
    if not model_subdirs:
        print("✗ No model found in models/ directory")
        print("  Did you run 1_download_on_internet.py?")
        sys.exit(1)
    
    model_path = model_subdirs[0] / "snapshots"
    snapshot_dirs = list(model_path.glob("*"))
    if not snapshot_dirs:
        print("✗ No model snapshot found")
        sys.exit(1)
    
    model_path = snapshot_dirs[0]
    print(f"\n📂 Model path: {model_path}")
    
    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device.upper()}")
    
    try:
        # Load model and tokenizer
        print("\n⏳ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        print("⏳ Loading model...")
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
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
        print("✓ Inference test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
