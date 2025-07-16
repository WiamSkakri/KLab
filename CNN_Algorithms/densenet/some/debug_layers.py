#!/usr/bin/env python3

import torch
import torchvision.models as models
import ai3
import copy

print("="*80)
print("DENSENET LAYER-BY-LAYER SOME ALGORITHM COMPATIBILITY TEST")
print("="*80)

# Load DenseNet121 model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.eval()

print(f"Testing individual Conv2d layers for 'some' algorithm compatibility...")
print(
    f"Total Conv2d layers to test: {sum(1 for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d))}")
print()

compatible_layers = []
incompatible_layers = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        # Create a copy of the layer for testing
        test_layer = torch.nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None
        )

        # Copy weights and bias
        with torch.no_grad():
            test_layer.weight.copy_(module.weight)
            if module.bias is not None:
                test_layer.bias.copy_(module.bias)

        # Test 'some' algorithm on this specific layer
        try:
            ai3.swap_conv2d(test_layer, "some")
            compatible_layers.append({
                'name': name,
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            })
            status = "✅ COMPATIBLE"
        except Exception as e:
            incompatible_layers.append({
                'name': name,
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'error': str(e)
            })
            status = f"❌ INCOMPATIBLE: {e}"

        print(f"{name:50} | {status}")

print(f"\n{'='*80}")
print(f"COMPATIBILITY SUMMARY")
print(f"{'='*80}")
print(f"Compatible layers: {len(compatible_layers)}")
print(f"Incompatible layers: {len(incompatible_layers)}")

if incompatible_layers:
    print(f"\n🔍 DETAILED ANALYSIS OF INCOMPATIBLE LAYERS:")
    print(f"{'-'*80}")
    for layer in incompatible_layers:
        print(f"Layer: {layer['name']}")
        print(
            f"  Config: {layer['in_channels']}→{layer['out_channels']}, kernel={layer['kernel_size']}, stride={layer['stride']}, padding={layer['padding']}")
        print(f"  Error: {layer['error']}")
        print()

# Analyze patterns in incompatible layers
if incompatible_layers:
    print(f"🔍 PATTERN ANALYSIS:")
    print(f"{'-'*40}")

    # Group by kernel size
    kernel_sizes = {}
    for layer in incompatible_layers:
        k = str(layer['kernel_size'])
        kernel_sizes[k] = kernel_sizes.get(k, 0) + 1

    print(f"Incompatible by kernel size:")
    for k, count in kernel_sizes.items():
        print(f"  {k}: {count} layers")

    # Group by channel counts
    small_channels = sum(
        1 for layer in incompatible_layers if layer['in_channels'] < 64 or layer['out_channels'] < 64)
    large_channels = sum(
        1 for layer in incompatible_layers if layer['in_channels'] > 512 or layer['out_channels'] > 512)

    print(f"Incompatible by channel size:")
    print(f"  Small channels (<64): {small_channels} layers")
    print(f"  Large channels (>512): {large_channels} layers")

print(f"\n💡 RECOMMENDATIONS:")
if len(incompatible_layers) == 0:
    print("✅ All layers are compatible! The issue might be with the full model conversion process.")
    print("   Try using a different AI3 conversion approach or check for memory/device issues.")
elif len(incompatible_layers) < len(compatible_layers):
    print(f"⚠️  Some layers are incompatible. Consider using 'guess' algorithm instead.")
    print(f"   The 'guess' algorithm can automatically select optimal algorithms per layer.")
else:
    print(f"❌ Many layers are incompatible with 'some'. Try alternative algorithms:")
    print(f"   - 'guess': Auto-selects optimal algorithm per layer")
    print(f"   - 'gemm': General matrix multiplication (works with most layers)")
    print(f"   - 'winograd': Optimal for 3x3 convolutions")

print(f"{'='*80}")
