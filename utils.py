# Module Docstring
"""
This module contains three main functions: hu_filter, cumprod_exclusive, and dvr.
These functions perform operations on images and tensors, with utility decorators
to log inputs/outputs and check for device compatibility.

Decorators:
- `input_output_decorator`: Logs the input arguments and output of a function.
- `device_check_decorator`: Ensures the device (CPU/GPU) is correctly selected
  based on availability.

Functions:
- `hu_filter`: Applies a window filter to an image based on specified center and width.
- `cumprod_exclusive`: Computes the cumulative product of a tensor with exclusive mode.
- `dvr`: Implements a simple direct volume rendering function.
"""
from functools import wraps
import torch

# Decorator to log input arguments and output
def input_output_decorator(func):
    """
    A decorator that logs the input arguments and output of the function it wraps.

    Args:
    - func (Callable): The function to be wrapped.

    Returns:
    - wrapper (Callable): The wrapped function with logging.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log the input arguments
        print(f"Input arguments: {args}, {kwargs}")
        # Call the original function and store the result
        result = func(*args, **kwargs)
        # Log the output
        print(f"Output: {result}")
        # Return the original function's result
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
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the device from the function's arguments (defaults to CPU if not provided)
        device = kwargs.get('device', torch.device('cpu'))
        # Check if CUDA is requested but not available
        if not torch.cuda.is_available() and device.type == 'cuda':
            print("CUDA not available, switching to CPU")
            # Switch to CPU if CUDA is unavailable
            kwargs['device'] = torch.device('cpu')
        # Call the original function with possibly modified arguments
        return func(*args, **kwargs)
    return wrapper

# Function to apply a window filter to an image
@input_output_decorator
def hu_filter(image, window_center, window_width):
    """
    Applies a Hounsfield Unit (HU) window filter to an image, clipping its values
    to a specified range based on the window center and width.

    Args:
    - image (np.ndarray): The input image to be filtered.
    - window_center (int/float): The center of the window range.
    - window_width (int/float): The width of the window range.

    Returns:
    - window_image (np.ndarray): The filtered image with values clipped to the specified range.
    """
    # Calculate the minimum value of the window range
    img_min = window_center - window_width // 2
    # Calculate the maximum value of the window range
    img_max = window_center + window_width // 2
    # Create a copy of the input image to avoid modifying the original
    window_image = image.clone()
    # Clip the image values below the minimum to the minimum value
    window_image[window_image < img_min] = img_min
    # Clip the image values above the maximum to the maximum value
    window_image[window_image > img_max] = img_max
    # Return the filtered image
    return window_image

# Function to compute cumulative product with exclusive mode
@input_output_decorator
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """
    Mimics the functionality of TensorFlow's tf.math.cumprod with exclusive mode,
    computing the cumulative product of a tensor along the last dimension.

    Args:
    - tensor (torch.Tensor): The input tensor for which the cumprod is computed.

    Returns:
    - cumprod (torch.Tensor): The cumulative product tensor with exclusive mode.
    """
    # Set the dimension along which to compute the cumulative product (fixed at 1)
    dim = 1
    # Compute the cumulative product of the tensor along the specified dimension
    cumprod = torch.cumprod(tensor, dim)
    # Shift the computed values by 1 position along the specified dimension
    cumprod = torch.roll(cumprod, 1, dim)
    # Set the first value in the rolled dimension to 0, mimicking exclusive mode
    cumprod[..., 0] = 0.
    # Return the modified cumulative product tensor
    return cumprod

# Function for direct volume rendering
@device_check_decorator
@input_output_decorator
def dvr(data, depth, cumprod_func, device, dim):
    """
    Implements a simple Direct Volume Rendering (dvr) algorithm using PyTorch.

    Args:
    - data (np.ndarray): The input 3D volume data to be rendered.
    - depth (int): The number of depth samples to use in the rendering process.
    - cumprod_func (Callable): A function to compute the cumulative product, typically
    'cumprod_exclusive.
    - device (torch.device): The device (CPU/GPU) to perform the computations on.
    - dim (int): The dimension along which the rendering is performed.

    Returns:
    - depth_map (torch.Tensor): The rendered depth map.
    - rgb_map (torch.Tensor): The rendered RGB map.
    - acc_map (torch.Tensor): The accumulated weights map.
    """
    # Normalize the data to be between 0 and 1
    dense_grid = (data - data.min()) / (data.max() - data.min())
    # Create a tensor of depth values, evenly spaced between 0 and depth-1
    depth_values = torch.linspace(0, depth-1, depth).to(device)
    # Create a large constant tensor to represent an "infinite" distance
    one_e_10 = torch.tensor([1e10], dtype=torch.float32, device=device)
    # Convert the input data into a torch tensor and move it to the specified device
    intensity_grid = torch.from_numpy(data).to(device)

    # Calculate the distances between consecutive depth values and append a large distance
    dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                       one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

    # Convert the normalized grid to a torch tensor and move it to the specified device
    density = torch.from_numpy(dense_grid).to(device)
    del dense_grid  # Free memory by deleting the original dense_grid
    # Compute the alpha values (opacity) based on the density and distances
    alpha = 1. - torch.exp(-density * dists)
    del density  # Free memory by deleting the original density tensor
    # Compute the weights for each sample by multiplying alpha with the cumulative product
    weights = alpha * cumprod_func(1. - alpha + 1e-10)
    # Compute the RGB map by summing the weighted intensities along the specified dimension
    rgb_map = (weights * intensity_grid).sum(dim=dim)
    # Compute the accumulated weights map
    acc_map = weights.sum(dim=dim)
    # Compute the depth map by summing the weighted depth values
    depth_map = (weights * depth_values).sum(dim=dim)
    del weights  # Free memory by deleting the weights tensor
    # Return the depth map, RGB map, and accumulated weights map
    return depth_map, rgb_map, acc_map
