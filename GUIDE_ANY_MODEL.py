#!/usr/bin/env python3
"""
Complete guide for using different Hugging Face models offline

IMPORTANT: Models must be downloaded on internet machine first, then transferred
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

print("=" * 70)
print("GUIDE: Using Any Hugging Face Model in Air-Gapped System")
print("=" * 70)

# ============================================================================
# SECTION 1: How to Download & Transfer Different Models
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 1: Downloading Models on Internet Machine")
print("=" * 70)

guide_download = """
Step 1: On your INTERNET-CONNECTED machine, edit 1_download_on_internet.py
        or use command line:

EXAMPLES:

1. GPT2 (default, 500MB, text generation)
   python 1_download_on_internet.py

2. DistilGPT2 (smaller, 167MB, text generation)
   python 1_download_on_internet.py distilgpt2

3. Mistral 7B (14GB, advanced generation - REQUIRES 16GB+ RAM)
   python 1_download_on_internet.py mistralai/Mistral-7B-v0.1

4. Llama 2 7B (13GB, general purpose - REQUIRES 16GB+ RAM)
   python 1_download_on_internet.py meta-llama/Llama-2-7b

5. T5 Base (892MB, sequence-to-sequence)
   python 1_download_on_internet.py google-t5/t5-base

6. Sentence Transformers (for embeddings, 133MB)
   python 1_download_on_internet.py sentence-transformers/all-MiniLM-L6-v2

7. CodeLlama (code generation, 13GB)
   python 1_download_on_internet.py codellama/CodeLlama-7b

Step 2: Copy entire huggingface-offline-setup/ folder to USB/external drive

Step 3: Transfer to air-gapped system

Step 4: On air-gapped system, reinstall:
        ./2_install_on_airgap.sh
        python 3_run_model.py
"""

print(guide_download)

# ============================================================================
# SECTION 2: Model Recommendations by Use Case
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Model Recommendations")
print("=" * 70)

models_table = """
┌─────────────────────────────────────────────────────────────────────────┐
│ Use Case              │ Model                    │ Size    │ GPU Memory  │
├─────────────────────────────────────────────────────────────────────────┤
│ Text Generation       │ gpt2                     │ 500MB   │ 2GB         │
│ (Light)              │ distilgpt2               │ 167MB   │ 1GB         │
├─────────────────────────────────────────────────────────────────────────┤
│ Text Generation       │ mistralai/Mistral-7B-v0 │ 14GB    │ 16GB        │
│ (Advanced)           │ meta-llama/Llama-2-7b   │ 13GB    │ 16GB        │
├─────────────────────────────────────────────────────────────────────────┤
│ Code Generation      │ codellama/CodeLlama-7b  │ 13GB    │ 16GB        │
│                      │ gpt2                     │ 500MB   │ 2GB         │
├─────────────────────────────────────────────────────────────────────────┤
│ Embeddings/          │ sentence-transformers/  │ 133MB   │ 1GB         │
│ Similarity           │ all-MiniLM-L6-v2        │         │             │
├─────────────────────────────────────────────────────────────────────────┤
│ Seq2Seq              │ google-t5/t5-base       │ 892MB   │ 4GB         │
│ (Translation)        │ google-t5/t5-small      │ 242MB   │ 2GB         │
├─────────────────────────────────────────────────────────────────────────┤
│ Question Answering   │ distilbert-base-uncased │ 326MB   │ 1GB         │
│                      │ bert-base-uncased       │ 440MB   │ 2GB         │
└─────────────────────────────────────────────────────────────────────────┘
"""

print(models_table)

# ============================================================================
# SECTION 3: Loading Models in Air-Gapped System
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Loading Models in Python (Air-Gapped System)")
print("=" * 70)

code_example = """
# BASIC EXAMPLE

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 1. Find model directory
from pathlib import Path
model_dir = Path("models/models--openai--gpt2/snapshots")
model_path = list(model_dir.glob("*"))[0]  # Get first snapshot

# 2. Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForCausalLM.from_pretrained(str(model_path))

# 3. Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 4. Generate text
prompt = "Hello, how are"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
"""

print(code_example)

# ============================================================================
# SECTION 4: GPU Detection
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: GPU Detection & Troubleshooting")
print("=" * 70)

print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("\n✓ GPU is ready for inference!")
else:
    print("\n⚠️  GPU not detected. Checking why:")
    print("  1. Check: nvidia-smi (should show your GPU)")
    print("  2. Check: python -c 'import torch; print(torch.cuda.is_available())'")
    print("  3. Try: pip install torch torchvision torchaudio pytorch-cuda=11.8")

gpu_troubleshoot = """
IF GPU IS NOT DETECTED:

1. Check NVIDIA drivers:
   nvidia-smi

2. If nvidia-smi works but PyTorch doesn't detect GPU:
   - Reinstall PyTorch with CUDA support
   - Get right version from: pytorch.org
   - Example for CUDA 11.8:
     pip install torch torchvision torchaudio pytorch-cuda=11.8

3. Verify in Python:
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))

4. Force CPU (if GPU issues):
   device = "cpu"  # in your code
"""

print(gpu_troubleshoot)

# ============================================================================
# SECTION 5: Complete Working Example
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Complete Working Example")
print("=" * 70)

working_example = """
# save this as: custom_inference.py

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

class OfflineModelRunner:
    def __init__(self, device=None):
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Find model
        model_dir = Path("models")
        model_subdirs = list(model_dir.glob("models--*"))
        if not model_subdirs:
            raise ValueError("No model found in models/ directory")
        
        model_path = list((model_subdirs[0] / "snapshots").glob("*"))[0]
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device.upper()}")
    
    def generate(self, prompt, max_length=50, temperature=0.7):
        \"\"\"Generate text from prompt\"\"\"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.95
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
if __name__ == "__main__":
    runner = OfflineModelRunner()  # Auto-detects GPU
    
    prompt = "Artificial intelligence is"
    result = runner.generate(prompt, max_length=100)
    print(f"\\nPrompt: {prompt}")
    print(f"Output: {result}")
"""

print(working_example)

print("\n" + "=" * 70)
print("✓ Guide complete! Now you can use any Hugging Face model offline")
print("=" * 70)
