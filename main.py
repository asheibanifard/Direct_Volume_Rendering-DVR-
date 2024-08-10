# Module Docstring
"""
This module processes volumetric data by applying an HU filter, performing Direct Volume Rendering
(DVR), and visualizing the result using Matplotlib. The module includes decorators to log input
and output, check device compatibility, and inspect the outer scope for debugging purposes.
"""
import functools
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_volume, process_volume
from utils import hu_filter, dvr, cumprod_exclusive

# Decorator to log input arguments, output, and outer scope variables
def input_output_decorator(func):
    """
    A decorator that logs the input arguments, output of the function it wraps,
    and variables in the outer scope.

    Args:
    - func (Callable): The function to be wrapped.

    Returns:
    - wrapper (Callable): The wrapped function with logging.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log the input arguments
        print(f"Input arguments for {func.__name__}: {args}, {kwargs}")

        # Log outer scope variables
        outer_scope_vars = {key: value for key, value in func.__globals__.items() if not
                            key.startswith('__')}
        print(f"Outer scope variables before {func.__name__}: {outer_scope_vars}")

        # Call the original function and store the result
        result = func(*args, **kwargs)

        # Log the output
        print(f"Output from {func.__name__}: {result}")

        # Log outer scope variables again (in case of modifications)
        outer_scope_vars_after = {key: value for key, value in func.__globals__.items() if not
                                  key.startswith('__')}
        print(f"Outer scope variables after {func.__name__}: {outer_scope_vars_after}")

        return result
    return wrapper

# Decorator to ensure proper device selection (CPU/GPU)
def device_check_decorator(func):
    """
    A decorator that checks and adjusts the device (CPU/GPU) for the function it wraps.

    Args:
    - func (Callable): The function to be wrapped.

    Returns:
    - wrapper (Callable): The wrapped function with device checking.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the device from the function's arguments (defaults to CPU if not provided)
        device = kwargs.get('device', torch.device('cpu'))
        # Check if CUDA is requested but not available
        if not torch.cuda.is_available() and device == 'cuda:0':
            print("CUDA not available, switching to CPU")
            # Switch to CPU if CUDA is unavailable
            kwargs['device'] = torch.device('cpu')
        # Call the original function with possibly modified arguments
        return func(*args, **kwargs)
    return wrapper

@input_output_decorator
def apply_hu_filter(vol, window_center, window_width):
    """
    Apply the HU filter to the input data.

    Args:
    - vol (np.ndarray): The input volume data.
    - window_center (int): The center of the HU window.
    - window_width (int): The width of the HU window.

    Returns:
    - filtered_data (np.ndarray): The data after applying the HU filter.
    """
    return hu_filter(vol, window_center, window_width)

@input_output_decorator
@device_check_decorator
def process_dvr(volume, vol_depth, cumprod_func, device, dim):
    """
    Perform Direct Volume Rendering (DVR) on the input tensor.

    Args:
    - volume (torch.Tensor): The input volume as a tensor.
    - vol_depth (int): The depth of the volume.
    - cumprod_func (Callable): The cumulative product function.
    - device (str): The device to perform the rendering on (CPU/GPU).
    - dim (int): The dimension along which to sum the DVR result.

    Returns:
    - depth_map (torch.Tensor): The depth map resulting from DVR.
    - rgb_map (torch.Tensor): The RGB map resulting from DVR.
    - acc_map (torch.Tensor): The accumulated opacity map resulting from DVR.
    """
    return dvr(volume.cpu().detach().numpy(), vol_depth, cumprod_func, device, dim)

@input_output_decorator
def visualize_rgb_map(rgb_map):
    """
    Visualize the RGB map resulting from DVR.

    Args:
    - rgb_map (torch.Tensor): The RGB map resulting from DVR.

    Returns:
    - None
    """
    plt.imshow(rgb_map, cmap='gray')
    plt.title('RGB Map Visualization')
    plt.show()

# Set the Matplotlib style to grayscale
plt.style.use('grayscale')

# Define the path to your volumetric data file
PATH = '/home/armin/Downloads/vis_male_128x256x256_uint8.raw'

# Define the dimensions of the volume
dims = (128, 256, 256)  # Assuming the volume is 128x256x256

# Define the data type of the volume
DType = np.uint8  # Assuming the data is 8-bit unsigned integers

# Load the volume data
load_vol = load_volume(PATH, dims, DType)

# Process the volume to convert it to a PyTorch tensor
load_vol = process_volume(load_vol,
                          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Visualize a specific slice of the volume

# Assuming `volume` is already loaded in the previous code
data = load_vol

# Apply the HU filter to the data
HU = apply_hu_filter(data, 218, 254)

# Perform Direct Volume Rendering (DVR)
depth, rgb, acc = process_dvr(HU, 128, cumprod_exclusive, 'cuda:0', 1)

# Visualize the RGB map
visualize_rgb_map(rgb.cpu().detach())
