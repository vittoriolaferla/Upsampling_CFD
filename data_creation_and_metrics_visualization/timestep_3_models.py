import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure
import matplotlib.patches as mpatches
import os
from scipy.ndimage import convolve
import imageio
from tqdm import tqdm
from PIL import Image
import seaborn as sns  # Ensure seaborn is installed: pip install seaborn

# Define global min and max based on expected velocity ranges
GLOBAL_UMIN = -0.97893
GLOBAL_UMAX = 0.83819
GLOBAL_VMIN = -0.97893
GLOBAL_VMAX = 0.83819
GLOBAL_WMIN = -0.97893
GLOBAL_WMAX = 0.83819

# Additional percentiles for robust scaling (if needed)
GLOBAL_UMIN_5TH = -0.97893
GLOBAL_UMAX_95TH = 0.83819
GLOBAL_VMIN_5TH = -0.97893
GLOBAL_VMAX_95TH = 0.83819
GLOBAL_WMIN_5TH = -0.97893
GLOBAL_WMAX_95TH = 0.83819

# Define channel target parameters
channel_target_mean = [0.537238078213845, 0.53890670991845, 0.5364345268330284]
channel_target_steap = [12, 12, 12]  # Note: "steap" likely meant "steep"

def soft_clipping_normalize_component(component, vmin, vmax, k=15, c=0.5):
    """
    Normalize the component with soft clipping using a sigmoid function.

    Parameters:
    - component: np.ndarray, the data to normalize
    - vmin: float, minimum value for min-max scaling
    - vmax: float, maximum value for min-max scaling
    - k: float, steepness parameter for the sigmoid function
    - c: float, midpoint for the sigmoid function

    Returns:
    - component_norm: np.ndarray, normalized data in [0, 1] with soft clipping
    """
    # Min-Max Scaling
    component_scaled = (component - vmin) / (vmax - vmin)
    
    # Apply sigmoid-based soft clipping
    component_soft_clipped = 1 / (1 + np.exp(-k * (component_scaled - c)))
    
    return component_soft_clipped

def save_grayscale_image(image_array, save_path):
    """
    Save a normalized grayscale image.

    Parameters:
    - image_array: np.ndarray, normalized image data in [0, 1]
    - save_path: str, path to save the image
    """
    # Convert to 8-bit grayscale
    image_uint8 = (image_array * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8, mode='L')  # 'L' mode for (8-bit pixels, black and white)
    image_pil.save(save_path)

def getSlice(casenum, step, axis, location, reduce_):
    """
    Extract a 2D slice from the CSV data corresponding to a specific case, step, and axis,
    and apply sigmoid-based normalization.

    Parameters:
    - casenum: int or str, identifier for the case
    - step: int or str, time step identifier
    - axis: int, axis along which to slice (0, 1, or 2)
    - location: float, location along the axis to slice
    - reduce_: int, factor by which to reduce the resolution

    Returns:
    - slice_u_norm, slice_v_norm, slice_w_norm: np.ndarray, sigmoid-normalized velocity components
    - slice_mag: np.ndarray, scalar magnitude
    """
    csv_path = f"/home/vittorio/Scrivania/ResShift_4_scale/data/Final_Steps_47/{casenum}_{step}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    working_csv = pd.read_csv(csv_path, skiprows=[0]).to_numpy()
    vels = working_csv[:, 4:7]  # u, v, w
    coords = working_csv[:, 0:3]
    new_working = np.concatenate((coords, vels), axis=1)

    try:
        final_working = new_working.reshape((48, 48, 48, 6), order='C')
    except ValueError as e:
        raise ValueError(f"Reshape error for casenum={casenum}, step={step}: {e}")

    tolerance = 1e-5
    if axis == 0:
        coords_axis = final_working[:, 0, 0, 0]
    elif axis == 1:
        coords_axis = final_working[0, :, 0, 1]
    elif axis == 2:
        coords_axis = final_working[0, 0, :, 2]
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    index = np.argmin(np.abs(coords_axis - location))
    coord_at_index = coords_axis[index]

    if not np.isclose(coord_at_index, location, atol=tolerance):
        raise ValueError(f"No slice found for location={location} on axis={axis}. Closest coordinate is {coord_at_index}")

    if axis == 0:
        slice_u = final_working[index, :, :, 3]
        slice_v = final_working[index, :, :, 4]
        slice_w = final_working[index, :, :, 5]
    elif axis == 1:
        slice_u = final_working[:, index, :, 3]
        slice_v = final_working[:, index, :, 4]
        slice_w = final_working[:, index, :, 5]
    elif axis == 2:
        slice_u = final_working[:, :, index, 3]
        slice_v = final_working[:, :, index, 4]
        slice_w = final_working[:, :, index, 5]

    # Reduce the resolution by subsampling
    slice_u = slice_u[::reduce_, ::reduce_]
    slice_v = slice_v[::reduce_, ::reduce_]
    slice_w = slice_w[::reduce_, ::reduce_]

    # Compute the magnitude of the velocity vector (not normalized)
    slice_mag = np.sqrt(slice_u**2 + slice_v**2 + slice_w**2)

    # Apply sigmoid-based normalization
    slice_u_norm = soft_clipping_normalize_component(
        slice_u, GLOBAL_UMIN_5TH, GLOBAL_UMAX_95TH, k=channel_target_steap[0], c=channel_target_mean[0]
    )
    slice_v_norm = soft_clipping_normalize_component(
        slice_v, GLOBAL_VMIN_5TH, GLOBAL_VMAX_95TH, k=channel_target_steap[1], c=channel_target_mean[1]
    )
    slice_w_norm = soft_clipping_normalize_component(
        slice_w, GLOBAL_WMIN_5TH, GLOBAL_WMAX_95TH, k=channel_target_steap[2], c=channel_target_mean[2]
    )

    print(f"Max U: {np.max(slice_u)}, Min U: {np.min(slice_u)}")
    print(f"Max V: {np.max(slice_v)}, Min V: {np.min(slice_v)}")
    print(f"Max W: {np.max(slice_w)}, Min W: {np.min(slice_w)}")
    global_min = min(np.min(slice_u), np.min(slice_v), np.min(slice_w))
    global_max = max(np.max(slice_u), np.max(slice_v), np.max(slice_w))
    print(f"Global Min Velocity: {global_min}, Global Max Velocity: {global_max}")

    return slice_u_norm, slice_v_norm, slice_w_norm, slice_mag

def process_and_save_separate_datasets(
    total_cases=128,
    steps_range=range(31, 41),
    reduce_=1,
    axis=1,
    location=0.21277,
    u_save_dir='/home/vittorio/Scrivania/ResShift_4_scale/data/3_Models/U_Images_Soft_4',
    v_save_dir='/home/vittorio/Scrivania/ResShift_4_scale/data/3_Models/V_Images_Soft_4',
    w_save_dir='/home/vittorio/Scrivania/ResShift_4_scale/data/3_Models/W_Images_Soft_4'):
    
    """
    Process and save separate datasets for u, v, and w velocity components using sigmoid-based normalization.

    Parameters:
    - total_cases: int, total number of cases
    - steps_range: iterable, range of steps to process
    - reduce_: int, subsampling factor
    - axis: int, axis along which to slice
    - location: float, location along the axis to slice
    - u_save_dir, v_save_dir, w_save_dir: str, directories to save grayscale images for each component
    """

    # Create directories if they don't exist
    os.makedirs(u_save_dir, exist_ok=True)
    os.makedirs(v_save_dir, exist_ok=True)
    os.makedirs(w_save_dir, exist_ok=True)

    saved_u_images = []
    saved_v_images = []
    saved_w_images = []

    # Initialize lists to collect normalized components
    all_u_norm = []
    all_v_norm = []
    all_w_norm = []

    # Also collect raw (non-normalized) components if needed
    all_u = []
    all_v = []
    all_w = []

    for c in tqdm(range(total_cases), desc="Processing Cases"):
        for s in steps_range:
            casenum = c
            step = s
            try:
                # Get the normalized slices
                slice_u_norm, slice_v_norm, slice_w_norm, slice_mag = getSlice(casenum, step, axis, location, reduce_)

                # Append normalized data to the lists
                all_u_norm.append(slice_u_norm.flatten())
                all_v_norm.append(slice_v_norm.flatten())
                all_w_norm.append(slice_w_norm.flatten())

            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping casenum={casenum}, step={step}: {e}")
                continue  # Skip to the next iteration if there's an error

            # Define the base filename
            base_filename = f'XZ_{casenum}_{step}_{reduce_}.png'

            # Define the save paths
            u_save_path = os.path.join(u_save_dir, base_filename)
            v_save_path = os.path.join(v_save_dir, base_filename)
            w_save_path = os.path.join(w_save_dir, base_filename)

            # Save each velocity component as a separate grayscale image
            save_grayscale_image(slice_u_norm, u_save_path)
            saved_u_images.append(u_save_path)
            print(f"Saved U image: {u_save_path}")

            save_grayscale_image(slice_v_norm, v_save_path)
            saved_v_images.append(v_save_path)
            print(f"Saved V image: {v_save_path}")

            save_grayscale_image(slice_w_norm, w_save_path)
            saved_w_images.append(w_save_path)
            print(f"Saved W image: {w_save_path}")

    # Concatenate all normalized components into single arrays
    if all_u_norm and all_v_norm and all_w_norm:
        all_u_norm = np.concatenate(all_u_norm)
        all_v_norm = np.concatenate(all_v_norm)
        all_w_norm = np.concatenate(all_w_norm)
    else:
        print("No normalized data collected.")
        all_u_norm = np.array([])
        all_v_norm = np.array([])
        all_w_norm = np.array([])

    print("Component U mean:", all_u_norm.mean() if all_u_norm.size > 0 else "N/A")
    print("Component V mean:", all_v_norm.mean() if all_v_norm.size > 0 else "N/A")
    print("Component W mean:", all_w_norm.mean() if all_w_norm.size > 0 else "N/A")

    print("Finished processing and saving separate datasets for U, V, and W velocity components.")

    # Return the collected normalized components along with saved image paths
    return saved_u_images, saved_v_images, saved_w_images, all_u_norm, all_v_norm, all_w_norm

def main():
    # Parameters
    total_cases = 127  # From 0 to 127
    steps_range = range(31, 41)  # Time steps from 31 to 40
    reduce_ = 4 # Subsampling factor
    axis = 1
    location = 0.21277

    # Define directories for separate datasets
    u_save_dir = '/home/vittorio/Scrivania/ResShift_4_scale/data/3_Models/U_Images'
    v_save_dir = '/home/vittorio/Scrivania/ResShift_4_scale/data/3_Models/V_Images'
    w_save_dir = '/home/vittorio/Scrivania/ResShift_4_scale/data/3_Models/W_Images'

    # Proceed to process and save separate datasets
    saved_u_images, saved_v_images, saved_w_images, all_u_norm, all_v_norm, all_w_norm = process_and_save_separate_datasets(
        total_cases=total_cases,
        steps_range=steps_range,
        reduce_=reduce_,
        axis=axis,
        location=location,
        u_save_dir=u_save_dir,
        v_save_dir=v_save_dir,
        w_save_dir=w_save_dir
    )



if __name__ == "__main__":
    # Step 1: Process and save grayscale images for U, V, and W velocity components
    main()
