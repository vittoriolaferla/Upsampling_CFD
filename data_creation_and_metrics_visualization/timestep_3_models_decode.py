import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# Global Normalization Parameters
# ----------------------------

# Define global min and max based on expected velocity ranges
GLOBAL_UMIN_5TH = -0.97893
GLOBAL_UMAX_95TH = 0.83819
GLOBAL_VMIN_5TH = -0.97893
GLOBAL_VMAX_95TH = 0.83819
GLOBAL_WMIN_5TH = -0.97893
GLOBAL_WMAX_95TH = 0.83819

# Define channel target parameters used in sigmoid-based normalization
channel_target_mean = [0.537238078213845, 0.53890670991845, 0.5364345268330284]
channel_target_steep = [12, 12, 12]  # Corrected spelling from "steap" to "steep"

# ----------------------------
# Decoding Functions
# ----------------------------

def invert_sigmoid(norm, k, c=0.5, epsilon=1e-16):
    """
    Invert the sigmoid-based soft clipping used during normalization.
    
    Parameters:
    - norm: np.ndarray, normalized data in [0,1]
    - k: float, steepness parameter used in the sigmoid function during normalization
    - c: float, midpoint parameter used in the sigmoid function during normalization
    - epsilon: float, small value to prevent log(0) and division by zero
    
    Returns:
    - scaled: np.ndarray, scaled data before sigmoid-based soft clipping
    """
    # Clip to avoid division by zero or log(0)
    norm = np.clip(norm, epsilon, 1 - epsilon)
    
    # Invert sigmoid-based soft clipping
    scaled = (np.log(norm / (1 - norm)) + k * c) / k
    
    return scaled

def decode_component(norm_image, vmin, vmax, k, c):
    """
    Decode a normalized velocity component back to its original scale.
    
    Parameters:
    - norm_image: np.ndarray, normalized image data in [0,1]
    - vmin: float, minimum velocity value used in min-max scaling
    - vmax: float, maximum velocity value used in min-max scaling
    - k: float, steepness parameter used in sigmoid during normalization
    - c: float, midpoint parameter used in sigmoid during normalization
    
    Returns:
    - component: np.ndarray, decoded velocity component
    """
    # Invert sigmoid-based soft clipping
    scaled = invert_sigmoid(norm_image, k, c)
    
    # Invert min-max scaling
    component = scaled * (vmax - vmin) + vmin
    
    return component

def decode_velocities(u_dir, v_dir, w_dir, scalar_output_dir):
    """
    Decode U, V, W velocity components from grayscale images and compute scalar magnitude.
    
    Parameters:
    - u_dir: str, directory containing U velocity images
    - v_dir: str, directory containing V velocity images
    - w_dir: str, directory containing W velocity images
    - scalar_output_dir: str, directory to save scalar magnitude images
    """
    # Ensure scalar output directory exists
    os.makedirs(scalar_output_dir, exist_ok=True)
    
    # Get list of filenames (assuming same filenames in all directories)
    u_files = sorted([f for f in os.listdir(u_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    v_files = sorted([f for f in os.listdir(v_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    w_files = sorted([f for f in os.listdir(w_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Check that all directories have the same files
    if not (u_files == v_files == w_files):
        raise ValueError("Filenames in U, V, W directories do not match.")
    
    # Iterate over all files
    for filename in tqdm(u_files, desc="Decoding Velocities"):
        # Paths to U, V, W images
        u_path = os.path.join(u_dir, filename)
        v_path = os.path.join(v_dir, filename)
        w_path = os.path.join(w_dir, filename)
        
        # Load images and convert to numpy arrays
        u_image = np.array(Image.open(u_path).convert('L'))
        v_image = np.array(Image.open(v_path).convert('L'))
        w_image = np.array(Image.open(w_path).convert('L'))
        
        # Normalize images from [0,255] to [0,1]
        u_norm = u_image.astype(np.float32) / 255.0
        v_norm = v_image.astype(np.float32) / 255.0
        w_norm = w_image.astype(np.float32) / 255.0
        
        # Decode each component back to original scale
        u = decode_component(u_norm, GLOBAL_UMIN_5TH, GLOBAL_UMAX_95TH, channel_target_steep[0], channel_target_mean[0])
        v = decode_component(v_norm, GLOBAL_VMIN_5TH, GLOBAL_VMAX_95TH, channel_target_steep[1], channel_target_mean[1])
        w = decode_component(w_norm, GLOBAL_WMIN_5TH, GLOBAL_WMAX_95TH, channel_target_steep[2], channel_target_mean[2])
        
        # Compute scalar magnitude
        scalar_magnitude = np.sqrt(u**2 + v**2 + w**2)
        
        # Save scalar magnitude image with specified colormap
        scalar_save_path = os.path.join(scalar_output_dir, filename)
        plt.imsave(scalar_save_path, scalar_magnitude, cmap='gist_rainbow_r', vmin=0, vmax=0.56, origin='lower')
        # Optional: Save scalar magnitude as grayscale image without colormap
        # plt.imsave(scalar_save_path, scalar_magnitude, cmap='gray', vmin=0, vmax=0.56, origin='lower')
        
        print(f"Saved Scalar Magnitude image: {scalar_save_path}")

def visualize_scalar_magnitude_images(scalar_output_dir, num_images=5):
    """
    Display a few scalar magnitude images from the save directory.
    
    Parameters:
    - scalar_output_dir: str, directory where scalar magnitude images are saved
    - num_images: int, number of images to display
    """
    import matplotlib.pyplot as plt
    
    saved_files = sorted([f for f in os.listdir(scalar_output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(saved_files))):
        img_path = os.path.join(scalar_output_dir, saved_files[i])
        img = Image.open(img_path)
        
        plt.subplot(1, num_images, i+1)
        plt.imshow(img, cmap='gist_rainbow_r', vmin=0, vmax=0.56)
        plt.title(os.path.basename(img_path))
        plt.axis('off')
    
    plt.show()

def main():
    # Directories containing decoded U, V, W images
    u_dir = '/home/vittorio/Scrivania/KAIR/results/test_U_component_4'
    v_dir = '/home/vittorio/Scrivania/KAIR/results/test_V_Component_4'
    w_dir = '/home/vittorio/Scrivania/KAIR/results/test_W_component_4'
    
    # Directory to save scalar magnitude images
    scalar_output_dir = '/home/vittorio/Scrivania/ResShift_4_scale/data/3_Models/Scalar_Magnitude_SwinIR_10k_4'
    
    # Decode velocities and save scalar magnitude images
    decode_velocities(
        u_dir=u_dir,
        v_dir=v_dir,
        w_dir=w_dir,
        scalar_output_dir=scalar_output_dir
    )
    
    # Optional: Visualize some saved scalar magnitude images
    print("\nDisplaying some saved scalar magnitude images:")
    visualize_scalar_magnitude_images(scalar_output_dir, num_images=5)

if __name__ == "__main__":
    main()
