#!/usr/bin/env python3
"""
4 Examples of using Hugging Face models offline
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Get model path
model_dir = Path(__file__).parent.parent / "models"
model_subdirs = list(model_dir.glob("models--*"))
if not model_subdirs:
    print("✗ No model found. Run 1_download_on_internet.py first.")
    exit(1)

model_path = list((model_subdirs[0] / "snapshots").glob("*"))[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(device)

print("=" * 60)
print("4 Examples of Hugging Face Offline Inference")
print("=" * 60)

# Example 1: Simple text generation
print("\n[Example 1] Simple Text Generation")
print("-" * 60)
prompt = "Python is a great programming language because"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=30)
print(f"Input:  {prompt}")
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# Example 2: Batch processing
print("\n[Example 2] Batch Processing")
print("-" * 60)
prompts = [
    "Machine learning is",
    "Deep learning helps",
    "Neural networks can"
]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**inputs, max_length=25)
for i, prompt in enumerate(prompts):
    result = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(f"Prompt {i+1}: {result}")

# Example 3: Device info
print("\n[Example 3] Device Information")
print("-" * 60)
print(f"Using device: {device.upper()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Example 4: Error handling
print("\n[Example 4] Error Handling")
print("-" * 60)
try:
    bad_prompt = None
    inputs = tokenizer(bad_prompt, return_tensors="pt")
except TypeError as e:
    print(f"✓ Caught error gracefully: {type(e).__name__}")
    print(f"  Message: Invalid prompt type")

print("\n" + "=" * 60)
print("✓ All examples completed!")
print("=" * 60)
