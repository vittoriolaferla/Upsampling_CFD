import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure
import matplotlib.patches as mpatches
import os
from scipy.ndimage import convolve
import imageio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

channel_target_mean = [ 0.537238078213845, 0.53890670991845,0.5364345268330284]
channel_target_steap = [12,12,12]

#0.1= 0.20278, 99.9=-0.20801
#0.05, 99.95
GLOBAL_UMIN_5TH=  -0.97893 #-0.76154
GLOBAL_UMAX_95TH= 0.83819
#0.2 = -0.16683, 99.8 = 0.17921
#0.05 =  -0.24492, 99.95= 0.25438, 
#0.1= -0.20424, 99.9=0.216
GLOBAL_VMIN_5TH= -0.97893 #-0.70624
GLOBAL_VMAX_95TH=  0.83819 #0.76394
#0.1=-0.40764, 99.9=0.20972
GLOBAL_WMIN_5TH= -0.97893
GLOBAL_WMAX_95TH=  0.83819 #0.44072


def decode_rgb_to_scalar(rgb_image, k=15, c=0.5, epsilon=1e-16):
    """
    Decode an RGB image back to scalar magnitude by inverting soft clipping and min-max scaling.

    Parameters:
    - rgb_image: np.ndarray, shape (H, W, 3), the RGB image to decode
    - k: float, steepness parameter used in the sigmoid function during normalization
    - c: float, midpoint parameter used in the sigmoid function during normalization
    - epsilon: float, small value to prevent log(0) and division by zero

    Returns:
    - scalar_magnitude: np.ndarray, shape (H, W), the reconstructed scalar magnitude
    """
    # Ensure the image is in float format and normalized to [0, 1]
    if rgb_image.dtype != np.float32 and rgb_image.dtype != np.float64:
        rgb_image = rgb_image.astype(np.float32) / 255.0

    # Decode normalized components
    u_norm = rgb_image[:, :, 0]
    v_norm = rgb_image[:, :, 1]
    w_norm = rgb_image[:, :, 2]

    # Clamp normalized values to avoid numerical issues
    u_norm_clamped = np.clip(u_norm, epsilon, 1 - epsilon)
    v_norm_clamped = np.clip(v_norm, epsilon, 1 - epsilon)
    w_norm_clamped = np.clip(w_norm, epsilon, 1 - epsilon)

    # Invert sigmoid-based soft clipping
    def invert_sigmoid(norm, k ,c=0.5):
        return (np.log(norm / (1 - norm)) + k * c) / k

    u_scaled = invert_sigmoid(u_norm_clamped, k = channel_target_steap[0], c=channel_target_mean[0])
    v_scaled = invert_sigmoid(v_norm_clamped, k= channel_target_steap[1], c=channel_target_mean[1])
    w_scaled = invert_sigmoid(w_norm_clamped, k = channel_target_steap[2], c=channel_target_mean[2])

    # Invert Min-Max Scaling
    u = u_scaled * (GLOBAL_UMAX_95TH - GLOBAL_UMIN_5TH) + GLOBAL_UMIN_5TH
    v = v_scaled * (GLOBAL_VMAX_95TH - GLOBAL_VMIN_5TH) + GLOBAL_VMIN_5TH
    w = w_scaled * (GLOBAL_WMAX_95TH - GLOBAL_WMIN_5TH) + GLOBAL_WMIN_5TH

    # Compute scalar magnitude
    scalar_magnitude = np.sqrt(u**2 + v**2 + w**2)

    return scalar_magnitude


def decode_and_save_scalar_images(rgb_save_dir, decoded_scalar_save_dir, flip=False,
                                 colormap='gist_rainbow_r', upsample=False, upsample_factor=2):
    """
    Decode RGB images back to scalar magnitude images, optionally upsample them, 
    flip them vertically, and save them.

    Parameters:
    - rgb_save_dir (str): Directory containing RGB images.
    - decoded_scalar_save_dir (str): Directory to save decoded scalar images.
    - vmax (float): Maximum value for normalization in saved images.
    - vmin (float): Minimum value for normalization in saved images.
    - colormap (str): Colormap to use when saving scalar images.
    - upsample (bool): If True, upsample images before decoding.
    - upsample_factor (int): Factor by which to upsample the images.
    """
    os.makedirs(decoded_scalar_save_dir, exist_ok=True)
    saved_decoded_scalar_images = []

    # Get list of RGB images
    rgb_image_files = [f for f in os.listdir(rgb_save_dir) if f.endswith('.png')]

    for filename in rgb_image_files:
        rgb_image_path = os.path.join(rgb_save_dir, filename)
        decoded_image_path = os.path.join(decoded_scalar_save_dir, filename)

        # Read the RGB image
        rgb_image = plt.imread(rgb_image_path)

        # Upsample the image if the flag is set
        if upsample:
            # Convert the numpy array to a PIL Image
            pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8)) if rgb_image.dtype != np.uint8 else Image.fromarray(rgb_image)
            original_size = pil_image.size
            new_size = (original_size[0] * upsample_factor, original_size[1] * upsample_factor)
            pil_image = pil_image.resize(new_size, Image.BICUBIC)
            # Convert back to numpy array and normalize if necessary
            rgb_image = np.array(pil_image) / 255.0 if pil_image.mode in ['RGB', 'RGBA'] else np.array(pil_image)

            print(f"Upsampled image {filename} from {original_size} to {new_size}")

        # Decode the scalar magnitude
        scalar_magnitude = decode_rgb_to_scalar(rgb_image)
        # Skip saving and loading images

        # Flip the scalar magnitude image vertically
        if flip:
            scalar_magnitude = np.flipud(scalar_magnitude)

        # Save the decoded scalar image using the specified vmin and vmax
        plt.imsave(decoded_image_path, scalar_magnitude, cmap=colormap, vmin=0, vmax=0.56, origin='lower')
        saved_decoded_scalar_images.append(decoded_image_path)
        print(f"Saved decoded scalar image: {decoded_image_path}")

    return saved_decoded_scalar_images

if __name__ == "__main__":
    saved_decoded_scalar_images = decode_and_save_scalar_images(
        rgb_save_dir= "//home/vittorio/Documenti/Upsampling_CFD/results/test_DAT_x2_Indoor_umass_0.35_Vector/visualization/test_set_physic",
        decoded_scalar_save_dir="/home/vittorio/Documenti/Upsampling_CFD/results/test_DAT_x2_Indoor_umass_0.35_Vector/visualization/decoded",
        colormap='gist_rainbow_r',
        flip=True,
        upsample=False
    )
