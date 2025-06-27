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
        self.hooks = []
        
    def register_hooks(self, model):
        # First, collect all modules and their names
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
                        'padding': module.padding
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
                            'algorithm': module.algorithm if hasattr(module, 'algorithm') else 'unknown'
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
        
        pass
            
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
            durations = [entry.get('duration_ms', 0) for entry in times if 'duration_ms' in entry]
            if durations:
                results[name] = sum(durations) / len(durations)
        return results
        
    def get_layer_dimensions(self):
        return self.layer_info

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

def main():
    # Configuration
    model_name = "VGG-16"
    algorithm = "direct"
    device = "cpu"
    batch_size = 1
    iterations = 10
    # Generate random input sizes between 224 and 512
    input_sizes = [random.randint(224, 512)
                   for _ in range(100)]

    results_dir = os.getcwd()  # Save in current directory

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

    # Load model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.eval()

    # Apply ai3 algorithm
    ai3.swap_conv2d(model, algorithm)

    # Create timer and register hooks
    timer = LayerTimer()
    timer.register_hooks(model)

    print("Job is starting")
    
    # Test with each input size
    for input_size in input_sizes:
        # Generate input data for this size
        input_data = torch.randn(batch_size, 3, input_size, input_size)

        # Warmup run
        with torch.inference_mode():
            _ = model(input_data)
        timer.reset()  # Reset timers after warmup

        # Measure execution time
        overall_start_time = time.time()
        with torch.inference_mode():
            for i in range(iterations):
                _ = model(input_data)
        overall_end_time = time.time()

        # Calculate overall execution time
        # Convert to milliseconds
        overall_execution_time = (
            overall_end_time - overall_start_time) / iterations * 1000

        # Get layer-wise timings and dimensions
        layer_times = timer.get_average_times()
        layer_dimensions = timer.get_layer_dimensions()
        actual_input_sizes = timer.get_actual_layer_input_sizes()
        
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
                
                # Extract dimension information
                in_channels = dimensions.get('in_channels', 'N/A')
                out_channels = dimensions.get('out_channels', 'N/A')
                kernel_size = format_tuple_value(dimensions.get('kernel_size', 'N/A'))
                stride = format_tuple_value(dimensions.get('stride', 'N/A'))
                padding = format_tuple_value(dimensions.get('padding', 'N/A'))
                
                # Use actual input size for this layer instead of model input size
                layer_input_size = actual_input_sizes.get(layer_name, input_size)
                
                writer.writerow([
                    model_name, layer_name, algorithm, device, batch_size, layer_input_size,
                    in_channels, out_channels, kernel_size, stride, padding,
                    avg_time, percentage
                ])

    # Clean up
    timer.remove_hooks()
    print("Job has ended")

if __name__ == "__main__":
    main()