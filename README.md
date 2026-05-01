# Hugging Face Offline Setup

Run Hugging Face transformer models completely offline on air-gapped systems using only Python and standard libraries. No frameworks like Ollama or ChatLLM required.

## 📋 Overview

This project provides a complete workflow to:
1. Download models & dependencies on an internet-connected machine
2. Transfer to an air-gapped system
3. Run inference without any internet access

## 📁 Folder Structure

```
huggingface-offline-setup/
├── README.md                      # This file
├── 1_download_on_internet.py      # Download script (run on connected machine)
├── 2_install_on_airgap.sh         # Installation script (run on air-gapped system)
├── 3_run_model.py                 # Inference script (run on air-gapped system)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git configuration
├── models/                        # (Empty - will contain downloaded models)
├── wheels/                        # (Empty - will contain .whl packages)
└── examples/
    └── simple_inference.py        # 4 practical examples
```

## 🚀 Quick Start

### Step 1: On Internet-Connected Machine

```bash
cd huggingface-offline-setup
python 1_download_on_internet.py
```

**This will:**
- Download distilbert-base-uncased model (~300MB)
- Download all Python packages as .whl files (~2GB)
- Create `models/` and `wheels/` directories

### Step 2: Transfer to Air-Gapped System

Copy the entire `huggingface-offline-setup/` folder to your air-gapped machine via:
- USB drive
- External hard drive
- Network transfer (if available)

### Step 3: On Air-Gapped Machine

```bash
cd huggingface-offline-setup
chmod +x 2_install_on_airgap.sh
./2_install_on_airgap.sh
python 3_run_model.py
```

## 📦 Supported Models

Default: **distilbert-base-uncased** (~300MB)

To use a different model, edit `1_download_on_internet.py` and change:

```python
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Change this
```

### Recommended Models by Size

| Model | Size | Use Case |
|-------|------|----------|
| distilbert-base-uncased | 326 MB | Classification, embeddings (default) |
| gpt2 | 548 MB | Text generation |
| distilgpt2 | 167 MB | Lightweight text generation |
| meta-llama/Llama-2-7b | 13 GB | General-purpose generation |
| mistralai/Mistral-7B-v0.1 | 14 GB | High-quality generation |

## 💻 Requirements

### Internet-Connected Machine
- Python 3.8+
- pip
- ~20 GB free space (models + dependencies)

### Air-Gapped Machine
- Python 3.8+
- pip
- RAM: 4GB minimum (more for larger models)
- GPU: Optional (CUDA-enabled GPU for faster inference)

## 📖 File Descriptions

### 1_download_on_internet.py
Downloads model and all Python wheels. Run on internet-connected machine.

```bash
python 1_download_on_internet.py
```

### 2_install_on_airgap.sh
Installs packages using downloaded .whl files without internet. Run on air-gapped system.

```bash
./2_install_on_airgap.sh
```

### 3_run_model.py
Loads model and runs inference test. Run on air-gapped system.

```bash
python 3_run_model.py
```

Features:
- Automatic device detection (CPU/GPU)
- Offline mode enabled
- Example inference
- Error handling

### examples/simple_inference.py
Four practical examples:
1. Simple text generation
2. Batch processing
3. Device information
4. Error handling

```bash
python examples/simple_inference.py
```

## 🔧 Customization

### Use Different Model

Edit `1_download_on_internet.py`:

```python
MODEL_NAME = "gpt2"  # or any HuggingFace model
```

### Adjust Generation Parameters

Edit `3_run_model.py` or create custom script:

```python
outputs = model.generate(
    **inputs,
    max_length=100,           # Output length
    temperature=0.7,          # Creativity (0-1)
    top_p=0.95,              # Diversity
    num_beams=1              # Beam search
)
```

### Use GPU

Automatic GPU detection is enabled. Requires CUDA-compatible GPU and PyTorch with CUDA.

## 🐛 Troubleshooting

### Issue: "Module not found" on air-gapped system

**Solution:** Check that `wheels/` directory contains .whl files and installation completed successfully.

```bash
ls -lah wheels/
```

### Issue: "Model not found" on air-gapped system

**Solution:** Verify `models/` directory exists and contains model files.

```bash
ls -lah models/
```

### Issue: Out of memory

**Solutions:**
- Use smaller model (distilbert, distilgpt2)
- Enable quantization (8-bit or 4-bit)
- Reduce batch size
- Use CPU instead of GPU

### Issue: Python version mismatch

**Solution:** Ensure Python 3.8+ on both machines.

```bash
python --version
```

## 🔐 Security & Privacy

✅ **No data leaves your system** - completely offline  
✅ **No API calls** - all processing local  
✅ **No tracking** - pure Python code  
✅ **Open source** - audit the code  

## 📚 Examples

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/path/to/local/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "Artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Batch Processing

```python
prompts = ["Hello", "How are", "What is"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(**inputs, max_length=30)

for i, prompt in enumerate(prompts):
    result = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(f"{prompt} -> {result}")
```

### With GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)
```

## 🤝 Contributing

Feel free to:
- Add more examples
- Improve documentation
- Add support for other model types
- Report issues

## 📄 License

MIT License - Use freely in personal and commercial projects.

## ❓ FAQ

**Q: Can I use different models?**  
A: Yes! Change `MODEL_NAME` in `1_download_on_internet.py` to any HuggingFace model.

**Q: What if I don't have GPU?**  
A: Scripts automatically fall back to CPU. Works fine but slower.

**Q: How much space do I need?**  
A: ~5GB for small models, ~20GB+ for larger models. Check model card on HuggingFace.

**Q: Can I run this on Windows?**  
A: Yes! Adjust path separators and use `.bat` instead of `.sh` if needed.

**Q: Is internet access ever required?**  
A: Only during the download phase. After that, 100% offline.

## 📞 Support

For issues:
1. Check troubleshooting section
2. Review HuggingFace documentation
3. Open an issue on GitHub

---

**Happy offline modeling!** 🚀
