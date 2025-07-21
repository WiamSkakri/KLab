#!/usr/bin/env python3

import torch
import torchvision.models as models
import ai3
import time
import os
import csv
import random
from collections import defaultdict
import inspect


class LayerTimer:
    """Timer class to measure execution time of individual layers"""

    def __init__(self):
        self.layer_times = defaultdict(list)
        self.layer_info = {}  # Store layer dimensions and properties
        self.layer_input_sizes = {}  # Store actual input sizes for each layer
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks to measure layer execution times"""
        for name, module in model.named_modules():
            # Check if it's either a Conv2d or has 'algorithm' attribute (ai3.swap_torch.Conv2D)
            if (isinstance(module, torch.nn.Conv2d) or
                    hasattr(module, 'algorithm')):

                # Store layer dimensions
                if isinstance(module, torch.nn.Conv2d):
                    # For standard PyTorch Conv2d
                    self.layer_info[name] = {
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding,
                        'algorithm': 'pytorch_conv2d'
                    }
                elif hasattr(module, 'algorithm'):
                    # For ai3.swap_torch.Conv2D
                    if hasattr(module, 'weight'):
                        # Extract dimensions from weight tensor
                        weight_shape = module.weight.shape
                        self.layer_info[name] = {
                            'out_channels': weight_shape[0],
                            'in_channels': weight_shape[1],
                            'kernel_size': (weight_shape[2], weight_shape[3]),
                            'algorithm': module.algorithm if hasattr(module, 'algorithm') else 'ai3_unknown'
                        }
                        # Try to get other parameters if available
                        if hasattr(module, 'stride'):
                            self.layer_info[name]['stride'] = module.stride
                        if hasattr(module, 'padding'):
                            self.layer_info[name]['padding'] = module.padding

                # Pre-forward hook to record start time
                pre_hook = module.register_forward_pre_hook(
                    self._create_pre_hook(name))
                # Post-forward hook to record end time
                post_hook = module.register_forward_hook(
                    self._create_post_hook(name))
                self.hooks.append(pre_hook)
                self.hooks.append(post_hook)

    def _create_pre_hook(self, name):
        def hook(module, input):
            entry = {'start': time.time()}
            if input and len(input) > 0 and hasattr(input[0], 'shape') and len(input[0].shape) >= 4:
                entry['input_height'] = input[0].shape[2]
                entry['input_width'] = input[0].shape[3]
                entry['input_channels'] = input[0].shape[1]
                entry['batch_size'] = input[0].shape[0]
            self.layer_times[name].append(entry)
        return hook

    def _create_post_hook(self, name):
        def hook(module, input, output):
            # Find the latest entry for this layer and add end time
            if self.layer_times[name]:
                self.layer_times[name][-1]['end'] = time.time()
        return hook

    def get_execution_times(self):
        """Calculate execution times for each layer"""
        times = {}
        for layer_name, time_entries in self.layer_times.items():
            layer_times = []
            for entry in time_entries:
                if 'start' in entry and 'end' in entry:
                    # Convert to ms
                    execution_time = (entry['end'] - entry['start']) * 1000
                    layer_times.append(execution_time)
            if layer_times:
                times[layer_name] = {
                    'times': layer_times,
                    'average': sum(layer_times) / len(layer_times),
                    'min': min(layer_times),
                    'max': max(layer_times)
                }
        return times

    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset(self):
        """Reset all collected timing data"""
        self.layer_times.clear()


def print_model_structure(model, prefix=''):
    """Utility function to debug model structure and verify AI3 conversion"""
    print(f"\n{'='*60}")
    print("MODEL STRUCTURE ANALYSIS")
    print(f"{'='*60}")

    total_layers = 0
    ai3_layers = 0
    pytorch_layers = 0
    guess_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or hasattr(module, 'algorithm'):
            total_layers += 1
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(module, 'algorithm'):
                ai3_layers += 1
                algorithm = module.algorithm
                if algorithm == 'guess':
                    guess_layers += 1
                print(f"✓ AI3 Layer: {full_name}")
                print(f"  - Algorithm: {algorithm}")
                if hasattr(module, 'weight'):
                    print(f"  - Weight shape: {module.weight.shape}")
                    # Check algorithmic suitability for guess algorithm
                    if len(module.weight.shape) >= 4:
                        kernel_h, kernel_w = module.weight.shape[2], module.weight.shape[3]
                        is_3x3 = kernel_h == 3 and kernel_w == 3
                        is_1x1 = kernel_h == 1 and kernel_w == 1
                        print(f"  - 3x3 kernel (Winograd suitable): {is_3x3}")
                        print(f"  - 1x1 kernel (GEMM suitable): {is_1x1}")
            elif isinstance(module, torch.nn.Conv2d):
                pytorch_layers += 1
                print(f"⚠ PyTorch Layer: {full_name}")
                print(f"  - Type: {type(module).__name__}")
                print(
                    f"  - In/Out channels: {module.in_channels}/{module.out_channels}")

    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY:")
    print(f"  Total Conv2D layers: {total_layers}")
    print(f"  AI3 converted: {ai3_layers}")
    print(f"  Guess algorithm layers: {guess_layers}")
    print(f"  PyTorch remaining: {pytorch_layers}")
    print(
        f"  Conversion rate: {(ai3_layers/total_layers*100):.1f}%" if total_layers > 0 else "No layers found")
    print(f"{'='*60}\n")


def main():
    """Main function to run DenseNet121 Guess Algorithm performance testing with AI3"""
    print("="*80)
    print("DENSENET121 GUESS ALGORITHM IMPLEMENTATION WITH AI3 LIBRARY")
    print("="*80)

    # Configuration
    model_name = "DenseNet121"
    # Using Guess algorithm with AI3 (AI3 will choose best algorithm per layer)
    algorithm = "guess"
    device = "cuda"  # Using CUDA for GPU acceleration
    batch_size = 1
    iterations = 10
    # Generate random input sizes between 224 and 512 for comprehensive testing
    input_sizes = [random.randint(224, 512)
                   # Quick testing for device fix verification
                   for _ in range(100)]

    # CUDA initialization and verification
    if device == "cuda":
        if not torch.cuda.is_available():
            print("✗ CUDA is not available on this system!")
            print("Falling back to CPU...")
            device = "cpu"
        else:
            print("CUDA Device Information:")
            print(f"  ✓ CUDA is available")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")
            print(f"  ✓ Current GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

            # Initialize CUDA context properly
            torch.cuda.init()
            torch.cuda.empty_cache()  # Clear any existing cache

            # Set up cuDNN for optimal performance with guess algorithm
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.allow_tf32 = True  # Allow advanced algorithms
            print("  ✓ CUDA context initialized successfully")

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(
        f"  Algorithm: {algorithm} (AI3 will select optimal algorithm per layer)")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations per input size: {iterations}")
    print(f"  Input sizes to test: {len(input_sizes)}")
    print()

    results_dir = os.getcwd()  # Save in current directory

    # Set up CSV files for results
    overall_csv_file = os.path.join(
        results_dir, f"{model_name}_{algorithm}_{device}_overall.csv")
    layers_csv_file = os.path.join(
        results_dir, f"{model_name}_{algorithm}_{device}_layers.csv")

    # Initialize CSV files with headers
    with open(overall_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Algorithm', 'Device',
                        'Batch_Size', 'Input_Size', 'Execution_Time_ms'])

    with open(layers_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Layer', 'Algorithm', 'Device', 'Batch_Size', 'Input_Size',
                         'In_Channels', 'Out_Channels', 'Kernel_Size', 'Stride', 'Padding',
                         'Execution_Time_ms', 'Percentage_of_Total', 'Selected_Algorithm'])

    # Load DenseNet121 model
    print("Loading DenseNet121 model...")
    try:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.eval()
        print("✓ DenseNet121 model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading DenseNet121 model: {e}")
        return

    # Print original model structure
    print("\nOriginal model loaded. Analyzing structure...")
    original_conv_count = sum(1 for name, module in model.named_modules()
                              if isinstance(module, torch.nn.Conv2d))
    print(f"Found {original_conv_count} Conv2D layers in original model")

    # Apply AI3 Guess algorithm conversion
    print(f"\nApplying AI3 {algorithm} algorithm conversion...")
    ai3_conversion_successful = False

    try:
        ai3.swap_conv2d(model, algorithm)
        ai3_conversion_successful = True
        print("✓ AI3 conversion completed successfully")
    except Exception as e:
        print(f"✗ Error during AI3 conversion: {e}")
        print(
            "This might be due to AI3 library not being installed or configured properly.")
        print("Falling back to standard PyTorch implementation...")
        ai3_conversion_successful = False

    # Move model to device - but AI3 may keep some components on CPU by design
    try:
        if not ai3_conversion_successful:
            # Only move everything to device if AI3 conversion failed
            model = model.to(device)
            print(f"✓ Model moved to {device}")
        else:
            print(
                f"✓ AI3 conversion complete - AI3 manages device placement automatically")

    except Exception as e:
        print(f"✗ Error moving model to {device}: {e}")
        if device == "cuda":
            print("Falling back to CPU...")
            device = "cpu"
            model = model.to(device)

    # Print model structure to verify conversion
    print_model_structure(model)

    # For AI3, we need to determine the appropriate input device
    input_device = "cpu"  # Default to CPU for AI3 mixed device setup
    if not ai3_conversion_successful and device == "cuda":
        input_device = "cuda"  # Only use CUDA inputs if not using AI3

    # Analyze device placement after AI3 conversion
    print("Analyzing device placement after AI3 conversion...")
    if device == "cuda":
        cuda_params = sum(1 for p in model.parameters() if p.is_cuda)
        cpu_params = sum(1 for p in model.parameters() if not p.is_cuda)
        total_params = cuda_params + cpu_params

        print(f"✓ AI3 device distribution:")
        print(f"  - Parameters on CUDA: {cuda_params}/{total_params}")
        print(f"  - Parameters on CPU: {cpu_params}/{total_params}")
        if ai3_conversion_successful:
            print(f"  - Mixed device usage is expected with AI3")

        # Count AI3 layers by device
        ai3_cuda_layers = sum(1 for name, module in model.named_modules()
                              if hasattr(module, 'algorithm') and hasattr(module, 'weight') and module.weight.is_cuda)
        ai3_cpu_layers = sum(1 for name, module in model.named_modules()
                             if hasattr(module, 'algorithm') and hasattr(module, 'weight') and not module.weight.is_cuda)

        print(f"  - AI3 layers on CUDA: {ai3_cuda_layers}")
        print(f"  - AI3 layers on CPU: {ai3_cpu_layers}")
        print(f"  - Input tensors will be placed on: {input_device}")

    # Create timer and register hooks
    timer = LayerTimer()
    timer.register_hooks(model)

    print("\nStarting performance testing...")
    print("This will test the model with various input sizes to measure Guess algorithm performance.")
    print()

    total_start_time = time.time()

    for i, input_size in enumerate(input_sizes, 1):
        print(
            f"[{i}/{len(input_sizes)}] Testing with input size {input_size}x{input_size}")

        # Create input tensor
        try:
            input_tensor = torch.randn(batch_size, 3, input_size, input_size)

            # Place input tensor on appropriate device
            input_tensor = input_tensor.to(input_device)

            print(
                f"  ✓ Created input tensor: {input_tensor.shape} on {input_tensor.device}")

            # Print GPU memory if using CUDA
            if device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(
                    f"  GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        except Exception as e:
            print(f"  ✗ Error creating input tensor: {e}")
            continue

        # Warmup runs
        print("  Running warmup...")
        try:
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_tensor)
                    # Only synchronize if input is on CUDA
                    if input_tensor.is_cuda:
                        torch.cuda.synchronize()
            print("  ✓ Warmup completed")

            if device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(
                    f"  GPU Memory after warmup - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        except Exception as e:
            print(f"  ✗ Error during warmup: {e}")
            continue

        # Reset timer for actual measurements
        timer.reset()

        # Performance measurement
        print(f"  Measuring performance over {iterations} iterations...")
        iteration_times = []

        try:
            for iteration in range(iterations):
                if (iteration + 1) % 5 == 0 or iteration == 0:
                    print(f"    Iteration {iteration + 1}/{iterations}")

                start_time = time.time()

                with torch.no_grad():
                    output = model(input_tensor)
                    # Only synchronize if input is on CUDA
                    if input_tensor.is_cuda:
                        torch.cuda.synchronize()

                end_time = time.time()
                iteration_time = (end_time - start_time) * \
                    1000  # Convert to ms
                iteration_times.append(iteration_time)

            avg_time = sum(iteration_times) / len(iteration_times)
            print(f"  ✓ Average execution time: {avg_time:.2f} ms")

        except Exception as e:
            print(f"  ✗ Error during performance measurement: {e}")
            continue

        # Get layer timing data
        layer_times = timer.get_execution_times()
        total_layer_time = sum(times['average']
                               for times in layer_times.values())

        # Write overall results
        with open(overall_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, algorithm, device,
                            batch_size, input_size, avg_time])

        # Write layer-wise results
        with open(layers_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for layer_name, times in layer_times.items():
                layer_info = timer.layer_info.get(layer_name, {})

                # Get layer algorithm info
                selected_algorithm = layer_info.get('algorithm', 'unknown')

                # Calculate percentage
                percentage = (
                    times['average'] / total_layer_time * 100) if total_layer_time > 0 else 0

                # Extract layer parameters
                in_channels = layer_info.get('in_channels', 0)
                out_channels = layer_info.get('out_channels', 0)
                kernel_size = layer_info.get('kernel_size', (0, 0))
                stride = layer_info.get('stride', (1, 1))
                padding = layer_info.get('padding', (0, 0))

                # Format kernel_size, stride, and padding
                if isinstance(kernel_size, tuple):
                    kernel_str = f"{kernel_size[0]}x{kernel_size[1]}" if len(
                        kernel_size) >= 2 else str(kernel_size[0])
                else:
                    kernel_str = str(kernel_size)

                if isinstance(stride, tuple):
                    stride_str = f"{stride[0]}x{stride[1]}" if len(
                        stride) >= 2 else str(stride[0])
                else:
                    stride_str = str(stride)

                if isinstance(padding, tuple):
                    padding_str = f"{padding[0]}x{padding[1]}" if len(
                        padding) >= 2 else str(padding[0])
                else:
                    padding_str = str(padding)

                writer.writerow([
                    model_name, layer_name, algorithm, device, batch_size, input_size,
                    in_channels, out_channels, kernel_str, stride_str, padding_str,
                    times['average'], percentage, selected_algorithm
                ])

        # Print top slowest layers for this input size
        if layer_times:
            sorted_layers = sorted(layer_times.items(),
                                   key=lambda x: x[1]['average'], reverse=True)
            print("  Top 5 slowest layers:")
            for layer_name, times in sorted_layers[:5]:
                layer_info = timer.layer_info.get(layer_name, {})
                selected_algo = layer_info.get('algorithm', 'unknown')
                percentage = (
                    times['average'] / total_layer_time * 100) if total_layer_time > 0 else 0
                print(
                    f"    {layer_name}: {times['average']:.2f} ms ({percentage:.1f}%) [{selected_algo}]")

        print()

    # Clean up
    timer.clear_hooks()

    total_time = time.time() - total_start_time
    print("="*80)
    print("DENSENET121 GUESS ALGORITHM TESTING COMPLETED")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Input sizes tested: {len(input_sizes)}")
    print(f"Results saved to:")
    print(f"  - Overall: {overall_csv_file}")
    print(f"  - Layer-wise: {layers_csv_file}")

    if ai3_conversion_successful:
        ai3_layer_count = sum(
            1 for name, module in model.named_modules() if hasattr(module, 'algorithm'))
        guess_layer_count = sum(1 for name, module in model.named_modules()
                                if hasattr(module, 'algorithm') and module.algorithm == 'guess')
        print(f"  ✓ {ai3_layer_count} layers converted to AI3")
        print(
            f"  ✓ {guess_layer_count} layers using Guess algorithm (AI3 auto-selection)")
    else:
        print(f"  ⚠ AI3 conversion failed - using standard PyTorch")
    print(f"  ✓ Comprehensive layer-wise performance analysis with algorithm selection details")
    print()
    print("Next steps:")
    print("  1. Analyze the CSV files to compare AI3 Guess performance vs other algorithms")
    print("  2. Run the same test with different algorithms (gemm, implicit_gemm, winograd, etc.) for comparison")
    print("  3. Compare which algorithms AI3 selected for different layer types and sizes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
