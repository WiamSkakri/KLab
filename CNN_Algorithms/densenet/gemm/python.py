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


def main():
    print("Starting computation")

    model_name = "DenseNet"
    algorithm = "gemm"
    device = "cuda"
    batch_size = 1
    iterations = 10
    input_sizes = [random.randint(224, 512) for _ in range(2)]

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

    # Count original Conv2d layers before swapping
    original_conv_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            original_conv_count += 1
    print(f"Original Conv2d layers found: {original_conv_count}")

    # Perform the ai3 swap
    print(f"Swapping Conv2d layers with ai3 {algorithm} algorithm...")
    ai3.swap_conv2d(model, algorithm)

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

    model = model.to(device_obj)

    timer = LayerTimer()
    timer.register_hooks(model)

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
        actual_dimensions = timer.get_actual_layer_dimensions()

        with open(overall_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, algorithm, device,
                            batch_size, input_size, overall_execution_time])

        with open(layers_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for layer_name, avg_time in layer_times.items():
                percentage = (avg_time / overall_execution_time) * 100
                dimensions = layer_dimensions.get(layer_name, {})
                actual_dims = actual_dimensions.get(layer_name, {})

                in_channels = dimensions.get('in_channels', 'N/A')
                out_channels = dimensions.get('out_channels', 'N/A')

                kernel_size_raw = dimensions.get('kernel_size', 'N/A')
                if kernel_size_raw != 'N/A' and isinstance(kernel_size_raw, (tuple, list)):
                    kernel_size = kernel_size_raw[0]
                else:
                    kernel_size = kernel_size_raw

                stride_raw = dimensions.get('stride', 'N/A')
                if stride_raw != 'N/A' and isinstance(stride_raw, (tuple, list)):
                    stride = stride_raw[0]
                else:
                    stride = stride_raw

                padding_raw = dimensions.get('padding', 'N/A')
                if padding_raw != 'N/A' and isinstance(padding_raw, (tuple, list)):
                    padding = padding_raw[0]
                else:
                    padding = padding_raw

                actual_input_h = actual_dims.get('actual_input_height', 'N/A')
                if actual_input_h != 'N/A':
                    layer_input_size = actual_input_h
                else:
                    layer_input_size = input_size

                writer.writerow([
                    model_name, layer_name, algorithm, device, batch_size, layer_input_size,
                    in_channels, out_channels, kernel_size, stride, padding,
                    avg_time, percentage
                ])

    timer.remove_hooks()
    print("Job is done")


if __name__ == "__main__":
    main()
