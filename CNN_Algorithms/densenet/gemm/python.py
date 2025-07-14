import torch
import torchvision.models as models
import ai3
import time
import os
import csv
import random
from collections import defaultdict
import torch.nn as nn
import torch.backends.cudnn as cudnn
import inspect


def format_tuple_value(value):
    """Convert square tuples like (3, 3) to single values like 3"""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == value[1]:
        return value[0]
    return value


def print_model_structure(model, prefix=''):
    """Utility function to debug model structure"""
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"{full_name}: {type(module).__name__}")

        # Print whether it's a swapped Conv2D
        if hasattr(module, 'algorithm'):
            print(f"  - Swapped with algorithm: {module.algorithm}")

        # Print module signature
        if hasattr(module, 'forward'):
            sig = inspect.signature(module.forward)
            print(f"  - Signature: {sig}")

        # Recursively print children
        print_model_structure(module, full_name)


class LayerTimer:
    def __init__(self):
        self.layer_times = defaultdict(list)
        self.layer_info = {}
        self.hooks = []

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or hasattr(module, 'algorithm'):
                if isinstance(module, torch.nn.Conv2d):
                    self.layer_info[name] = {
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding
                    }
                elif hasattr(module, 'algorithm'):
                    if hasattr(module, 'weight'):
                        weight_shape = module.weight.shape
                        self.layer_info[name] = {
                            'out_channels': weight_shape[0],
                            'in_channels': weight_shape[1],
                            'kernel_size': (weight_shape[2], weight_shape[3]),
                            'algorithm': module.algorithm if hasattr(module, 'algorithm') else 'unknown'
                        }
                        if hasattr(module, 'stride'):
                            self.layer_info[name]['stride'] = module.stride
                        if hasattr(module, 'padding'):
                            self.layer_info[name]['padding'] = module.padding

                pre_hook = module.register_forward_pre_hook(
                    self._create_pre_hook(name))
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
            if name in self.layer_times and self.layer_times[name]:
                last_entry = self.layer_times[name][-1]
                if 'start' in last_entry:
                    last_entry['end'] = time.time()
                    last_entry['duration_ms'] = (
                        last_entry['end'] - last_entry['start']) * 1000

                    if hasattr(output, 'shape') and len(output.shape) >= 4:
                        last_entry['output_height'] = output.shape[2]
                        last_entry['output_width'] = output.shape[3]
                        last_entry['output_channels'] = output.shape[1]
        return hook

    def reset(self):
        for name in self.layer_times:
            self.layer_times[name] = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_average_times(self):
        results = {}
        for name, times in self.layer_times.items():
            durations = [entry.get('duration_ms', 0)
                         for entry in times if 'duration_ms' in entry]
            if durations:
                results[name] = sum(durations) / len(durations)
        return results

    def get_layer_dimensions(self):
        return self.layer_info

    def get_actual_layer_dimensions(self):
        results = {}
        for name, times in self.layer_times.items():
            if times:
                last_run = times[-1]
                results[name] = {
                    'actual_input_height': last_run.get('input_height', 'N/A'),
                    'actual_input_width': last_run.get('input_width', 'N/A'),
                    'actual_input_channels': last_run.get('input_channels', 'N/A'),
                    'actual_output_height': last_run.get('output_height', 'N/A'),
                    'actual_output_width': last_run.get('output_width', 'N/A'),
                    'actual_output_channels': last_run.get('output_channels', 'N/A'),
                    'batch_size': last_run.get('batch_size', 'N/A')
                }
        return results

    def get_actual_layer_input_sizes(self):
        """Get the actual input dimensions that each layer received during forward pass"""
        results = {}
        for name, times in self.layer_times.items():
            if times:
                last_run = times[-1]
                if 'input_height' in last_run and 'input_width' in last_run:
                    # Use actual input size if available
                    if last_run['input_height'] == last_run['input_width']:
                        # Single value for square
                        results[name] = last_run['input_height']
                    else:
                        results[name] = f"{last_run['input_height']}x{last_run['input_width']}"
                else:
                    results[name] = 'N/A'
        return results


def main():
    print("Starting computation")

    model_name = "DenseNet"
    algorithm = "gemm"
    device = "cuda"
    batch_size = 1
    iterations = 10
    input_sizes = [random.randint(224, 512) for _ in range(1)]

    results_dir = os.getcwd()

    overall_csv_file = os.path.join(
        results_dir, f"{model_name}_{algorithm}_{device}_overall.csv")
    layers_csv_file = os.path.join(
        results_dir, f"{model_name}_{algorithm}_{device}_layers.csv")

    with open(overall_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Algorithm', 'Device',
                        'Batch_Size', 'Input_Size', 'Execution_Time_ms'])

    with open(layers_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Layer', 'Algorithm', 'Device', 'Batch_Size', 'Input_Size',
                        'In_Channels', 'Out_Channels', 'Kernel_Size', 'Stride', 'Padding',
                         'Execution_Time_ms', 'Percentage_of_Total'])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This implementation requires a CUDA-capable GPU.")

    device_obj = torch.device(device)

    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.eval()

    # Don't force device placement before ai3 swap - let AI3 handle it
    # Count original Conv2d layers before swapping
    original_conv_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            original_conv_count += 1
    print(f"Original Conv2d layers found: {original_conv_count}")

    # Perform the ai3 swap
    print(f"Swapping Conv2d layers with ai3 {algorithm} algorithm...")
    ai3.swap_conv2d(model, algorithm)

    # Don't force device placement after ai3 swap - AI3 handles mixed device usage

    # Verify the swap worked
    ai3_conv_count = 0
    pytorch_conv_count = 0
    ai3_layers = []
    remaining_pytorch_layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pytorch_conv_count += 1
            remaining_pytorch_layers.append(name)
        elif hasattr(module, 'algorithm') and module.algorithm == algorithm:
            ai3_conv_count += 1
            ai3_layers.append(name)
        elif module.__class__.__name__ == "Conv2D":  # ai3 Conv2D class
            ai3_conv_count += 1
            ai3_layers.append(name)

    print(f"\nAfter ai3 swap:")
    print(f"  ai3 {algorithm} layers: {ai3_conv_count}")
    print(f"  Remaining PyTorch Conv2d layers: {pytorch_conv_count}")

    if ai3_conv_count == 0:
        print("⚠️  WARNING: No ai3 layers detected! The swap may have failed.")
        print("Available modules after swap:")
        for name, module in model.named_modules():
            if 'conv' in name.lower():
                print(f"  {name}: {type(module).__name__}")
    else:
        print(
            f"✅ SUCCESS: {ai3_conv_count} layers successfully swapped to ai3 {algorithm}")
        print(f"First few ai3 layers: {ai3_layers[:5]}")

    if pytorch_conv_count > 0:
        print(f"⚠️  NOTE: {pytorch_conv_count} PyTorch Conv2d layers remain:")
        print(f"  {remaining_pytorch_layers[:5]}...")

    # Verify algorithm attribute on ai3 layers
    verified_algorithm_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'algorithm'):
            if module.algorithm == algorithm:
                verified_algorithm_count += 1
            else:
                print(
                    f"⚠️  Layer {name} has algorithm '{module.algorithm}' instead of '{algorithm}'")

    if verified_algorithm_count > 0:
        print(
            f"✅ VERIFIED: {verified_algorithm_count} layers confirmed using {algorithm} algorithm")

    timer = LayerTimer()
    timer.register_hooks(model)

    print("Job is starting")

    for input_size in input_sizes:
        input_data = torch.randn(
            batch_size, 3, input_size, input_size, device=device_obj)

        # Warmup
        with torch.inference_mode():
            _ = model(input_data)
        timer.reset()

        overall_start_time = time.time()
        with torch.inference_mode():
            for i in range(iterations):
                _ = model(input_data)
        overall_end_time = time.time()

        overall_execution_time = (
            overall_end_time - overall_start_time) / iterations * 1000

        layer_times = timer.get_average_times()
        layer_dimensions = timer.get_layer_dimensions()
        actual_input_sizes = timer.get_actual_layer_input_sizes()

        with open(overall_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, algorithm, device,
                            batch_size, input_size, overall_execution_time])

        with open(layers_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for layer_name, avg_time in layer_times.items():
                percentage = (avg_time / overall_execution_time) * 100
                dimensions = layer_dimensions.get(layer_name, {})

                # Extract dimension information using format_tuple_value
                in_channels = dimensions.get('in_channels', 'N/A')
                out_channels = dimensions.get('out_channels', 'N/A')
                kernel_size = format_tuple_value(
                    dimensions.get('kernel_size', 'N/A'))
                stride = format_tuple_value(dimensions.get('stride', 'N/A'))
                padding = format_tuple_value(dimensions.get('padding', 'N/A'))

                # Use actual input size for this layer instead of model input size
                layer_input_size = actual_input_sizes.get(
                    layer_name, input_size)

                writer.writerow([
                    model_name, layer_name, algorithm, device, batch_size, layer_input_size,
                    in_channels, out_channels, kernel_size, stride, padding,
                    avg_time, percentage
                ])

    timer.remove_hooks()
    print("Job has ended")


if __name__ == "__main__":
    main()
