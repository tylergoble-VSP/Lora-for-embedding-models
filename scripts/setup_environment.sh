#!/bin/bash
# Environment setup script for LoRA embedding fine-tuning project

echo "Setting up environment for LoRA EmbeddingGemma Fine-Tuning..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. GPU may not be available."
fi

# Check Hugging Face login
echo ""
echo "Checking Hugging Face authentication..."
if huggingface-cli whoami &> /dev/null; then
    echo "✓ Logged in to Hugging Face"
    huggingface-cli whoami
else
    echo "✗ Not logged in to Hugging Face"
    echo "Please run: huggingface-cli login"
fi

echo ""
echo "Setup complete!"
echo "To activate the environment, run: source .venv/bin/activate"

