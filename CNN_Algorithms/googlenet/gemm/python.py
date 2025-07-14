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
            self.layer_times[name].append({'start': time.time()})
        return hook

    def _create_post_hook(self, name):
        def hook(module, input, output):
            if name in self.layer_times and self.layer_times[name]:
                last_entry = self.layer_times[name][-1]
                if 'start' in last_entry:
                    last_entry['end'] = time.time()
                    last_entry['duration_ms'] = (
                        last_entry['end'] - last_entry['start']) * 1000
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


class GEMMConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GEMMConv2d, self).__init__(in_channels, out_channels,
                                         kernel_size, stride, padding, dilation, groups, bias)
        self.algorithm = 'gemm'

    def forward(self, x):
        # Enable cuDNN autotuner for optimal GEMM performance
        with torch.backends.cudnn.flags(enabled=True, benchmark=True):
            return super(GEMMConv2d, self).forward(x)


def convert_to_gemm(model):
    """Convert all Conv2d layers to GEMM-optimized Conv2d layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new_conv = GEMMConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None
            )
            new_conv.weight.data = module.weight.data
            if module.bias is not None:
                new_conv.bias.data = module.bias.data
            setattr(model, name, new_conv)
        else:
            convert_to_gemm(module)
    return model


def main():
    # Configuration
    model_name = "GoogLeNet"
    algorithm = "gemm"
    device = "cuda"  # Using CUDA for GPU acceleration
    batch_size = 1
    iterations = 10
    input_sizes = [random.randint(224, 512) for _ in range(100)]

    results_dir = os.getcwd()

    # Set up CSV files
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

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This implementation requires a CUDA-capable GPU.")

    # Load model
    print(f"Loading {model_name}...")
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

    # Convert model to use GEMM-optimized Conv2d layers
    model = convert_to_gemm(model)

    # Move model to GPU and set to evaluation mode
    model = model.to(device)
    model.eval()
    print("Model loaded and converted to GEMM successfully")

    # Enable cuDNN benchmarking for optimal performance
    cudnn.benchmark = True

    # Create timer and register hooks
    timer = LayerTimer()
    timer.register_hooks(model)

    # Test with each input size
    for input_size in input_sizes:
        print(f"\nTesting with input size {input_size}x{input_size}")
        # Create input on CPU to avoid device mismatch with ai3
        input_data = torch.randn(batch_size, 3, input_size, input_size)

        # Warmup run
        print("Running warmup...")
        with torch.inference_mode():
            _ = model(input_data)
        timer.reset()
        print("Warmup completed")

        # Measure execution time
        print(f"Measuring performance over {iterations} iterations...")
        overall_start_time = time.time()
        with torch.inference_mode():
            for i in range(iterations):
                print(f"  Running iteration {i+1}/{iterations}")
                _ = model(input_data)
        overall_end_time = time.time()

        # Calculate overall execution time
        overall_execution_time = (
            overall_end_time - overall_start_time) / iterations * 1000
        print(f"Average execution time: {overall_execution_time:.2f} ms")

        # Get layer-wise timings and dimensions
        layer_times = timer.get_average_times()
        layer_dimensions = timer.get_layer_dimensions()

        # Print layer times and dimensions
        print("\nLayer-wise execution times:")
        for layer_name, avg_time in sorted(layer_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (avg_time / overall_execution_time) * 100
            dimensions = layer_dimensions.get(layer_name, {})
            print(f"{layer_name}: {avg_time:.2f} ms ({percentage:.2f}% of total)")

        # Save overall results to CSV
        with open(overall_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, algorithm, device,
                            batch_size, input_size, overall_execution_time])

        # Save per-layer results to CSV
        with open(layers_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for layer_name, avg_time in layer_times.items():
                percentage = (avg_time / overall_execution_time) * 100
                dimensions = layer_dimensions.get(layer_name, {})

                in_channels = dimensions.get('in_channels', 'N/A')
                out_channels = dimensions.get('out_channels', 'N/A')
                kernel_size = dimensions.get('kernel_size', 'N/A')
                stride = dimensions.get('stride', 'N/A')
                padding = dimensions.get('padding', 'N/A')

                writer.writerow([
                    model_name, layer_name, algorithm, device, batch_size, input_size,
                    in_channels, out_channels, kernel_size, stride, padding,
                    avg_time, percentage
                ])

    # Clean up
    timer.remove_hooks()
    print(
        f"\nGoogLeNet GEMM testing completed successfully. Results saved to:\n- {overall_csv_file}\n- {layers_csv_file}")


if __name__ == "__main__":
    main()
