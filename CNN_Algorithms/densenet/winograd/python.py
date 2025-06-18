import torch
import torchvision.models as models
import ai3
import time
import os
import csv
import random
from collections import defaultdict
import torch.nn as nn

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
                    last_entry['duration_ms'] = (last_entry['end'] - last_entry['start']) * 1000
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

class WinogradConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(WinogradConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.algorithm = 'winograd'
        
        # Winograd F(2x2, 3x3) transformation matrices
        self.G = torch.tensor([
            [1, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        self.B = torch.tensor([
            [1, 0, -1, 0],
            [0, 1, 1, 0],
            [0, -1, 1, 0],
            [0, 1, 0, -1]
        ], dtype=torch.float32)
        
        # Modified A matrix for 2x2 output
        self.A = torch.tensor([
            [1, 1, 1, 0],
            [0, 1, -1, -1]
        ], dtype=torch.float32)
    
    def winograd_2x2_3x3(self, x, weight):
        """
        Implement Winograd F(2x2, 3x3) convolution
        This implementation is for 3x3 kernels with 2x2 output tiles
        """
        batch_size, channels, height, width = x.shape
        out_channels, in_channels, kh, kw = weight.shape
        
        # Transform the weight matrix
        g = self.G.to(x.device)
        g_t = g.t()
        U = g @ weight.view(out_channels, in_channels, 3, 3) @ g_t
        U = U.view(out_channels, in_channels, 4, 4)
        
        # Pad input if necessary
        pad_h = (4 - height % 2) % 2
        pad_w = (4 - width % 2) % 2
        x_padded = torch.nn.functional.pad(x, (1, pad_w + 1, 1, pad_h + 1))
        
        # Process input in 4x4 tiles
        tiles_h = (height + pad_h) // 2
        tiles_w = (width + pad_w) // 2
        
        # Initialize output
        output = torch.zeros(batch_size, out_channels, tiles_h * 2, tiles_w * 2, device=x.device)
        
        b = self.B.to(x.device)
        b_t = b.t()
        a = self.A.to(x.device)
        a_t = a.t()
        
        # Process each tile
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Extract 4x4 tile
                tile = x_padded[:, :, i*2:i*2+4, j*2:j*2+4]
                
                # Transform input tile
                V = b @ tile.reshape(-1, 4, 4) @ b_t
                V = V.view(batch_size, channels, 4, 4)
                
                # Element-wise multiplication and sum using einsum
                M = torch.einsum('bchw,oihw->bohw', V, U)
                
                # Inverse transform
                # First transform along height dimension
                temp = torch.einsum('bohw,mw->bohm', M, a)
                # Then transform along width dimension
                Y = torch.einsum('bohm,nw->bonm', temp, a)
                
                # Place the result in the output tensor
                output[:, :, i*2:i*2+2, j*2:j*2+2] = Y
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output[:, :, :height, :width]
    
    def forward(self, x):
        if self.kernel_size == (3, 3) and self.stride == (1, 1) and self.dilation == (1, 1):
            try:
                return self.winograd_2x2_3x3(x, self.weight)
            except Exception as e:
                print(f"Warning: Winograd convolution failed ({str(e)}), falling back to standard convolution")
                return super(WinogradConv2d, self).forward(x)
        else:
            return super(WinogradConv2d, self).forward(x)

def convert_to_winograd(model):
    """Convert all Conv2d layers to Winograd-optimized Conv2d layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new_conv = WinogradConv2d(
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
            convert_to_winograd(module)
    return model

def main():
    # Configuration
    model_name = "DenseNet"
    algorithm = "winograd"
    device = "cpu"  # This doesn't affect ai3's operation since it uses cuDNN internally
    batch_size = 1
    iterations = 10
    input_sizes = [random.randint(224, 512) for _ in range(2)]  # Testing with 2 sizes initially
    
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
    print(f"Loading {model_name}...")
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.eval()
    print("Model loaded successfully")
    
    # Apply ai3 algorithm
    print(f"Applying {algorithm} algorithm...")
    ai3.swap_conv2d(model, algorithm)
    print("Algorithm applied successfully")
    
    # Create timer and register hooks
    timer = LayerTimer()
    timer.register_hooks(model)
    
    # Test with each input size
    for input_size in input_sizes:
        print(f"\nTesting with input size {input_size}x{input_size}")
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
        overall_execution_time = (overall_end_time - overall_start_time) / iterations * 1000
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
            writer.writerow([model_name, algorithm, device, batch_size, input_size, overall_execution_time])
        
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
    print(f"\nDenseNet Winograd testing completed successfully. Results saved to:\n- {overall_csv_file}\n- {layers_csv_file}")

if __name__ == "__main__":
    main()
