
# Sentence Transformer model to onnx

This repository contains scripts for working with a Sentence Transformer model using ONNX for inference.

## Usage:

### 1. `compile.py`
This script compiles a Sentence Transformer model to ONNX format. It includes:
- Tokenization and model setup.
- ONNX export with specified input and output names and dynamic axes.

### 2. `run_onnx.py`
This script runs inference using the compiled ONNX model. It includes:
- Setting up ONNX runtime session with CUDA support if available.
- Tokenization using Hugging Face transformers.
- Comparison of embeddings between the ONNX and PyTorch models.

### Example Usage:
- Run `python compile.py` to compile the model to ONNX format.
- After compilation, use `python run_onnx.py` to perform inference with the ONNX model and compare results with the original PyTorch model.
