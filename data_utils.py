# Module Docstring
"""
This module reads a volumetric data file, processes it into a 3D array, converts
it to a PyTorch tensor, and visualizes a slice of the volume using Matplotlib.

The code handles the following tasks:
- Loading a raw binary file into a 3D numpy array.
- Processing and reshaping the array.
- Converting the numpy array to a PyTorch tensor for further computations.
- Visualizing a slice of the 3D volume using Matplotlib.

Decorators:
- `input_output_decorator`: Logs the input arguments and output of a function.
- `device_check_decorator`: Ensures the device (CPU/GPU) is correctly selected
  based on availability.
"""
from functools import wraps
import numpy as np
import torch
import matplotlib.pyplot as plt


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

# Function to read and process the volumetric data
@input_output_decorator
def load_volume(vol_path, vol_dims, vol_dtype):
    """
    Loads a volumetric data file into a numpy array and reshapes it.

    Args:
    - path (str): Path to the raw volumetric data file.
    - dims (tuple): Dimensions of the volume (width, height, depth).
    - dtype (np.dtype): Data type of the volume.

    Returns:
    - volume (np.ndarray): Loaded and reshaped 3D numpy array.
    """
    with open(vol_path, 'rb') as f:
        # Read the binary file and reshape it into the specified dimensions
        vol = np.frombuffer(f.read(), dtype=vol_dtype).reshape(vol_dims)
    return vol

# Function to convert numpy array to PyTorch tensor and process it
@input_output_decorator
@device_check_decorator
def process_volume(vol, device=torch.device('cpu')):
    """
    Converts a numpy array volume to a PyTorch tensor and processes it.

    Args:
    - vol (np.ndarray): The input volume as a 3D numpy array.
    - device (torch.device): The device to process the tensor on (CPU/GPU).

    Returns:
    - vol (torch.Tensor): The processed volume as a PyTorch tensor.
    """
    # Crop the volume to a smaller region (e.g., select the first 128x128 pixels in the last two
    # dimensions)
    volume = vol[:, :128, :128]

    # Convert the numpy array to a PyTorch tensor and cast it to float32
    volume = torch.from_numpy(volume).to(torch.float32).to(device)

    # Permute the tensor dimensions (if necessary)
    # Note: This line is a placeholder; typically, you would specify how to permute dimensions.
    volume = volume.permute(0, 1, 2)  # No change in dimensions, but maintaining the line for
    # completeness.
    return volume

# Function to visualize a slice of the volume
@input_output_decorator
def visualize_slice(vol, slice_idx=70):
    """
    Visualizes a specific slice of the 3D volume using Matplotlib.

    Args:
    - vol (torch.Tensor): The 3D volume tensor.
    - slice_idx (int): The index of the slice to visualize.

    Returns:
    - None
    """
    plt.imshow(vol[slice_idx].cpu(), cmap='gray')
    plt.title(f'Slice {slice_idx} of the Volume')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

