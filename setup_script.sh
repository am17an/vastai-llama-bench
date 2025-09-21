#!/bin/bash

# Enhanced setup script with logging
set -e  # Exit on any error
set -x  # Print commands as they're executed

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a setup.log
}

log "Starting VastAI benchmark setup..."

# Update package lists
log "Updating package lists..."
apt-get update
apt install -y ccache
# Install required packages
log "Installing required packages..."
apt-get install -y libcurl4-openssl-dev sqlite3

# Install Python packages
log "Installing Python packages..."
pip install GitPython tabulate

# Clone llama.cpp
log "Cloning llama.cpp repository..."
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure git
log "Configuring git..."
git config user.email "aman@example.com"
git config user.name "Benchmark User"

pip install huggingface-hub


# Apply patch
log "Applying patch..."
cp ~/patch.diff .
git apply patch.diff
git add -u
git checkout -b patch
git commit -m 'Applied performance patch'

# Run benchmark
log "Starting benchmark comparison..."
log "This may take a while - comparing master vs patch for MUL_MAT operations..."

log "Downloading model..."
#huggingface-cli download ibm-granite/granite-4.0-tiny-preview-GGUF --include "granite-4.0-tiny-preview-f16.gguf" --local-dir models
#huggingface-cli download ggml-org/gpt-oss-20b-GGUF --local-dir models
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF --local-dir models --include "Qwen3-30B-A3B-Q4_K_M.gguf"

# Make sure the script is executable
chmod +x ./scripts/compare-commits.sh

# Run the benchmark with CUDA enabled
log "Running benchmark..."
#GGML_CUDA=1 ./scripts/compare-commits.sh master patch test-backend-ops -o MUL_MAT_ID -p "type_a=[f16,f32].*" 2>&1 | tee benchmark.log
GGML_CUDA=1 ./scripts/compare-commits.sh master patch llama-bench -m models/Qwen3-30B-A3B-Q4_K_M.gguf -p 4096 -ub 1,2,4,8,16,32,256,512


log "Benchmark setup and execution completed!"
log "Results saved to results.out.txt"
log "Full benchmark log saved to benchmark.log"

