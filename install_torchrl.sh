#!/bin/bash

# TorchRL Installation for GRPO Training
echo "=== Installing TorchRL for GRPO Training ==="

# Install TorchRL and dependencies
pip install torchrl==0.3.0
pip install tensordict==0.3.0
pip install gymnasium==0.29.1

# Verify installation
python -c "
import torchrl
import tensordict
print('✅ TorchRL version:', torchrl.__version__)
print('✅ TensorDict version:', tensordict.__version__)
print('✅ Installation successful!')
"

echo "TorchRL installation completed!"
echo "Now you can run: python train_grpo_torchrl.py" 