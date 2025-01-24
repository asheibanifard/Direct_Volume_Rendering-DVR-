import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import torch
from data_utils import process_volume
from utils import dvr, cumprod_exclusive
# Rotation Matrices
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
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/rendered.png')

# Set the Matplotlib style to grayscale
plt.style.use('grayscale')
def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def combined_rotation_matrix(phi, theta, psi):
    """
    Combine rotations around X, Y, and Z axes.
    :param phi: Rotation angle around the X-axis (in radians).
    :param theta: Rotation angle around the Y-axis (in radians).
    :param psi: Rotation angle around the Z-axis (in radians).
    :return: Combined rotation matrix.
    """
    rx = rotation_matrix_x(phi)
    ry = rotation_matrix_y(theta)
    rz = rotation_matrix_z(psi)
    return rz @ ry @ rx  # Matrix multiplication in Z-Y-X order

# Compute camera view matrix
def compute_view_matrix(camera_position, target, up_vector, rotation_matrix):
    """
    Compute the camera's view matrix after applying a rotation.
    """
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    # Apply rotation to camera axes
    rotated_forward = rotation_matrix @ forward
    rotated_up = rotation_matrix @ up
    rotated_right = rotation_matrix @ right

    # View matrix
    view_matrix = np.array([
        [rotated_right[0], rotated_up[0], -rotated_forward[0], 0],
        [rotated_right[1], rotated_up[1], -rotated_forward[1], 0],
        [rotated_right[2], rotated_up[2], -rotated_forward[2], 0],
        [-np.dot(rotated_right, camera_position),
         -np.dot(rotated_up, camera_position),
         np.dot(rotated_forward, camera_position), 1]
    ])
    return view_matrix

# Generate a random 3D volume
def generate_volume(shape=(64, 64, 64)):
    """
    Generate a random 3D volume with values between 0 and 1.
    """
    return np.random.random(shape)

# Rotate the volume
def rotate_volume(volume, rotation_matrix):
    """
    Rotate the volume using an affine transformation.
    """
    center = np.array(volume.shape) / 2
    offset = center - np.dot(rotation_matrix, center)
    rotated_volume = affine_transform(volume, rotation_matrix, offset=offset, order=1)
    return rotated_volume

# Visualize slices of the volume
def visualize_volume_slices(volume, title="Volume Slices"):
    """
    Visualize slices of the volume along three axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(volume[volume.shape[0] // 2, :, :], cmap='gray')
    axes[0].set_title('Slice along X-axis')
    axes[1].imshow(volume[:, volume.shape[1] // 2, :], cmap='gray')
    axes[1].set_title('Slice along Y-axis')
    axes[2].imshow(volume[:, :, volume.shape[2] // 2], cmap='gray')
    axes[2].set_title('Slice along Z-axis')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Initial camera setup
    camera_position = np.array([0, 0, 5])  # Camera position
    target = np.array([0, 0, 0])           # Target point
    up_vector = np.array([0, 1, 0])        # Up direction

    # Generate a random 3D volume
    volume = torch.load('./data/preprocessed/target_volume.pth').cpu().detach().numpy()
    visualize_volume_slices(volume, title="Original Volume")

    # Define rotation (e.g., rotate 45 degrees around Y-axis)
    theta_x = np.radians(45)
    theta_y = np.radians(0)
    theta_z = np.radians(45)
    rotation_matrix = combined_rotation_matrix(theta_x, theta_y, theta_z)
    
    # Compute the new view matrix
    view_matrix = compute_view_matrix(camera_position, target, up_vector, rotation_matrix)
    print("View Matrix After Rotation:\n", view_matrix)

    # Rotate the volume
    rotated_volume = rotate_volume(volume, rotation_matrix)
    load_vol = process_volume(rotated_volume,
                          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Visualize a specific slice of the volume

    # Assuming `volume` is already loaded in the previous code
    data = load_vol

    # Apply the HU filter to the data
    # HU = apply_hu_filter(data, 218, 254)

    # Perform Direct Volume Rendering (DVR)
    depth, rgb, acc = process_dvr(data, 64, cumprod_exclusive, 'cuda:0', 1)

    # Visualize the RGB map
    visualize_rgb_map(rgb.cpu().detach())
    visualize_volume_slices(rotated_volume, title="Rotated Volume")
