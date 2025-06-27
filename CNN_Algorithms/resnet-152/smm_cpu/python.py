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
    def __init__(self):
        self.layer_times = defaultdict(list)
        self.layer_info = {}  # Store layer dimensions
        self.layer_input_sizes = {}  # Store actual input sizes for each layer
        self.hooks = []
        
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if (isinstance(module, torch.nn.Conv2d) or 
                hasattr(module, 'algorithm')):
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
            # Capture actual input tensor dimensions
            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
                if hasattr(input_tensor, 'shape') and len(input_tensor.shape) >= 4:
                    # For 4D tensors (batch, channels, height, width)
                    batch_size, channels, height, width = input_tensor.shape
                    self.layer_input_sizes[name] = height

            self.layer_times[name].append({'start': time.time()})
        return hook
    
    def _create_post_hook(self, name):
        def hook(module, input, output):
            if name in self.layer_times and self.layer_times[name]:
                last_entry = self.layer_times[name][-1]
                if 'start' in last_entry:
                    last_entry['end'] = time.time()
                    last_entry['duration_ms'] = (last_entry['end'] - last_entry['start']) * 1000
        return hook
    
    def reset(self):
        for name in self.layer_times:
            self.layer_times[name] = []
        self.layer_input_sizes = {}
            
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_average_times(self):
        results = {}
        for name, times in self.layer_times.items():
            durations = [entry.get('duration_ms', 0) for entry in times if 'duration_ms' in entry]
            if durations:
                results[name] = sum(durations) / len(durations)
        return results
        
    def get_layer_dimensions(self):
        return self.layer_info

    def get_layer_input_sizes(self):
        return self.layer_input_sizes

def format_square_tuple(value):
    """Convert square tuples like (3,3) to single values like 3"""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == value[1]:
        return value[0]
    return value

def main():
    print("Job is starting")

    # Configuration
    model_name = "ResNet-152"
    algorithm = "smm"
    device = "cpu"
    batch_size = 1
    iterations = 10
    input_sizes = [random.randint(224, 512) for _ in range(2)]  # Start with 2 for testing
    
    results_dir = os.getcwd()
    
    # Set up CSV files
    overall_csv_file = os.path.join(results_dir, f"{model_name}_{algorithm}_{device}_overall.csv")
    layers_csv_file = os.path.join(results_dir, f"{model_name}_{algorithm}_{device}_layers.csv")
    
    with open(overall_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Algorithm', 'Device', 'Batch_Size', 'Input_Size', 'Execution_Time_ms'])
    
    with open(layers_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Layer', 'Algorithm', 'Device', 'Batch_Size', 'Input_Size', 
                         'In_Channels', 'Out_Channels', 'Kernel_Size', 'Stride', 'Padding',
                         'Execution_Time_ms', 'Percentage_of_Total'])
    
    # Load model
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    model.eval()
    
    # Apply ai3 algorithm
    ai3.swap_conv2d(model, algorithm)
    
    # Create timer and register hooks
    timer = LayerTimer()
    timer.register_hooks(model)
    
    print("Writing to the csv file")

    # Test with each input size
    for input_size in input_sizes:
        input_data = torch.randn(batch_size, 3, input_size, input_size)
        
        # Warmup run
        with torch.inference_mode():
            _ = model(input_data)
        timer.reset()
        
        # Measure execution time
        overall_start_time = time.time()
        with torch.inference_mode():
            for _ in range(iterations):
                _ = model(input_data)
        overall_end_time = time.time()
        
        # Calculate overall execution time
        overall_execution_time = (overall_end_time - overall_start_time) / iterations * 1000
        
        # Get layer-wise timings and dimensions
        layer_times = timer.get_average_times()
        layer_dimensions = timer.get_layer_dimensions()
        layer_input_sizes = timer.get_layer_input_sizes()
        
        # Save overall results to CSV
        with open(overall_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, algorithm, device, batch_size, input_size, overall_execution_time])
        
        # Save per-layer results to CSV
        with open(layers_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for layer_name, avg_time in layer_times.items():
                percentage = (avg_time / overall_execution_time) * 100
                dimensions = layer_dimensions.get(layer_name, {})
                actual_input_size = layer_input_sizes.get(layer_name, 'N/A')
                
                in_channels = dimensions.get('in_channels', 'N/A')
                out_channels = dimensions.get('out_channels', 'N/A')
                kernel_size = format_square_tuple(
                    dimensions.get('kernel_size', 'N/A'))
                stride = format_square_tuple(dimensions.get('stride', 'N/A'))
                padding = format_square_tuple(dimensions.get('padding', 'N/A'))
                
                writer.writerow([
                    model_name, layer_name, algorithm, device, batch_size, actual_input_size,
                    in_channels, out_channels, kernel_size, stride, padding,
                    avg_time, percentage
                ])

    # Clean up
    timer.remove_hooks()
    print("Job has ended")

if __name__ == "__main__":
    main()
