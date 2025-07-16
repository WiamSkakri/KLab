#!/usr/bin/env python3

import torch
import ai3
import sys

print("="*60)
print("AI3 LIBRARY DIAGNOSTIC CHECK")
print("="*60)

# Check AI3 version
print(f"AI3 installation location: {ai3.__file__}")
if hasattr(ai3, '__version__'):
    print(f"AI3 version: {ai3.__version__}")
else:
    print("AI3 version: Not available (older installation?)")

# Check AI3 available functions and attributes
print(f"\nAI3 available attributes:")
ai3_attrs = [attr for attr in dir(ai3) if not attr.startswith('_')]
for attr in sorted(ai3_attrs):
    print(f"  - {attr}")

# Try to get supported algorithms
print(f"\nTesting algorithm support:")
algorithms_to_test = ['some', 'guess', 'gemm', 'winograd', 'direct', 'smm']

# Create a simple test model
test_model = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
print(f"Created test Conv2d layer: {test_model}")

for algorithm in algorithms_to_test:
    try:
        # Create a copy of the test model for each algorithm
        test_copy = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        ai3.swap_conv2d(test_copy, algorithm)
        print(f"  ✅ '{algorithm}' - SUPPORTED")
    except Exception as e:
        print(f"  ❌ '{algorithm}' - ERROR: {e}")

# Check PyTorch and CUDA info
print(f"\nEnvironment Info:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Check for cuDNN
print(f"  cuDNN available: {torch.backends.cudnn.is_available()}")
print(f"  cuDNN version: {torch.backends.cudnn.version()}" if torch.backends.cudnn.is_available(
) else "  cuDNN version: Not available")

print("="*60)
