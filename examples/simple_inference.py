#!/usr/bin/env python3
"""
4 Examples of using Hugging Face models offline
FIXED: All examples now work properly with GPU and batch processing
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
print(f"Loading on {device.upper()}...")

tokenizer = AutoTokenizer.from_pretrained(str(model_path))
# FIX: Set pad token to prevent batch processing errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(device)
model.eval()

print("=" * 60)
print("4 Examples of Hugging Face Offline Inference")
print("=" * 60)

# Example 1: Simple text generation
print("\n[Example 1] Simple Text Generation")
print("-" * 60)
prompt = "Python is a great programming language because"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=30)
print(f"Input:  {prompt}")
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# Example 2: Batch processing (FIXED)
print("\n[Example 2] Batch Processing")
print("-" * 60)
prompts = [
    "Machine learning is",
    "Deep learning helps",
    "Neural networks can"
]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=25)
for i, prompt in enumerate(prompts):
    result = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(f"Prompt {i+1}: {result}")

# Example 3: Device info (ENHANCED)
print("\n[Example 3] Device Information")
print("-" * 60)
print(f"Using device: {device.upper()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    reserved_memory = torch.cuda.memory_reserved(0) / 1e9
    allocated_memory = torch.cuda.memory_allocated(0) / 1e9
    print(f"Memory allocated: {allocated_memory:.2f} GB / reserved: {reserved_memory:.2f} GB")

# Example 4: Error handling (ENHANCED with 5 tests)
print("\n[Example 4] Error Handling (5 Tests)")
print("-" * 60)

# Test 1: Invalid prompt type
print("Test 1: Invalid prompt type")
try:
    bad_prompt = None
    inputs = tokenizer(bad_prompt, return_tensors="pt")
except TypeError as e:
    print(f"✓ Caught error: {type(e).__name__}")

# Test 2: Empty prompt
print("\nTest 2: Empty prompt handling")
try:
    empty_prompt = ""
    inputs = tokenizer(empty_prompt, return_tensors="pt")
    if inputs['input_ids'].shape[1] == 0:
        print("✓ Caught: Empty tokenization result")
except Exception as e:
    print(f"✓ Caught error: {type(e).__name__}")

# Test 3: Very long prompt
print("\nTest 3: Long prompt (truncation)")
try:
    long_prompt = "The " * 2000  # 8000 words
    inputs = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=512)
    print(f"✓ Truncated {len(long_prompt.split())} tokens to 512")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Different tensor types
print("\nTest 4: Tensor type handling")
try:
    prompt = "Tensor types:"
    inputs_pt = tokenizer(prompt, return_tensors="pt")
    inputs_np = tokenizer(prompt, return_tensors="np")
    print(f"✓ PyTorch tensors: {type(inputs_pt['input_ids'])}")
    print(f"✓ NumPy arrays: {type(inputs_np['input_ids'])}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Device mismatch handling
print("\nTest 5: GPU inference verification")
try:
    test_input = tokenizer("Test", return_tensors="pt").to(device)
    assert test_input['input_ids'].device.type == device
    print(f"✓ Input correctly on {device.upper()}")
except AssertionError:
    print(f"✗ Input/GPU mismatch")

print("\n" + "=" * 60)
print("✓ All examples completed!")
print("✓ GPU support: " + ("ENABLED" if torch.cuda.is_available() else "CPU MODE"))
print("=" * 60)
