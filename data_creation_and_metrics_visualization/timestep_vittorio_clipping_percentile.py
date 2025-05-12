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

# Define global vmin and vmax based on expected velocity ranges
#GLOBAL_VMIN = -0.97893   # Minimum velocity (non-negative)
#GLOBAL_VMAX =  0.83819  # Adjusted maximum component velocity

GLOBAL_UMIN=  -0.97893  #-0.76154
GLOBAL_UMAX= 0.83819 #0.83819
GLOBAL_VMIN = -0.97893 #-0.70624
GLOBAL_VMAX=  0.83819 # 0.76394
GLOBAL_WMIN = -0.97893
GLOBAL_WMAX =  0.83819  #0.44072


# Define directories for saving images
RGB_SAVE_DIR = '/home/vittorio/Scrivania/ResShift_4_scale/data/RGB_Images_soft_4'
SCALAR_SAVE_DIR = '/home/vittorio/Scrivania/ResShift_4_scale/data/Scalar_Images'
DECODED_SCALAR_SAVE_DIR = '/home/vittorio/Scrivania/ResShift_4_scale/data/Decoded_Scalar_Images'

# Create directories if they don't exist
os.makedirs(RGB_SAVE_DIR, exist_ok=True)
os.makedirs(SCALAR_SAVE_DIR, exist_ok=True)
os.makedirs(DECODED_SCALAR_SAVE_DIR, exist_ok=True)


#channel_target_mean = [ 0.4743525824257478,  0.48059976378879715, 0.686624102700547]
 # Example values for R, G, B
channel_target_mean = [  0.537238078213845, 0.53890670991845,0.5364345268330284]
channel_target_steap = [10,10,10]

#0.1= 0.20278, 99.9=-0.20801
#0.05, 99.95
GLOBAL_UMIN_5TH= -0.97893 #-0.76154 
GLOBAL_UMAX_95TH= 0.83819
#0.2 = -0.16683, 99.8 = 0.17921
#0.05 =  -0.24492, 99.95= 0.25438, 
#0.1= -0.20424, 99.9=0.216
GLOBAL_VMIN_5TH=  -0.97893 #-0.70624
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
    
    return component_scaled


def encode_velocity_components(u_norm, v_norm, w_norm):
    """
    Encode normalized velocity components into RGB channels.

    Parameters:
    - u_norm, v_norm, w_norm: np.ndarray, normalized velocity components in [0, 1]

    Returns:
    - encoded_image: np.ndarray, shape (H, W, 3), encoded RGB image
    """
    # Directly use the normalized components without additional clipping
    R = u_norm
    G = v_norm
    B = w_norm

    encoded_image = np.stack((R, G, B), axis=2)
    return encoded_image




def getSlice(casenum, step, axis, location, reduce_):
    """
    Extract a 2D slice from the CSV data corresponding to a specific case, step, and axis.

    Parameters:
    - casenum: int or str, identifier for the case
    - step: int or str, time step identifier
    - axis: int, axis along which to slice (0, 1, or 2)
    - location: float, location along the axis to slice
    - reduce_: int, factor by which to reduce the resolution

    Returns:
    - enhanced_image: np.ndarray, shape (H, W, 3), encoded RGB image
    - slice_u, slice_v, slice_w: np.ndarray, velocity components
    - slice_mag: np.ndarray, scalar magnitude
    """
    # Read the CSV file for the given case number and time step
    csv_path = f"/home/vittorio/Scrivania/ResShift_4_scale/data/Final_Steps_47/{casenum}_{step}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    working_csv = pd.read_csv(csv_path, skiprows=[0]).to_numpy()

    # Extract the velocity components (u, v, w)
    vels = working_csv[:, 4:7]  # u, v, w

    # Extract the coordinate components (x, y, z)
    coords = working_csv[:, 0:3]

    # Combine coordinates and velocities
    new_working = np.concatenate((coords, vels), axis=1)

    # Reshape the data into a 4D array with dimensions (48, 48, 48, 6)
    try:
        final_working = new_working.reshape((48, 48, 48, 6), order='C')
    except ValueError as e:
        raise ValueError(f"Reshape error for casenum={casenum}, step={step}: {e}")

    # Use tolerances in floating-point comparisons
    tolerance = 1e-5

    # Depending on the specified axis, extract the 2D slice at the given location
    if axis == 0:
        x_coords = final_working[:, 0, 0, 0]
        index = np.argmin(np.abs(x_coords - location))
        coord_at_index = x_coords[index]
    elif axis == 1:
        y_coords = final_working[0, :, 0, 1]
        index = np.argmin(np.abs(y_coords - location))
        coord_at_index = y_coords[index]
    elif axis == 2:
        z_coords = final_working[0, 0, :, 2]
        index = np.argmin(np.abs(z_coords - location))
        coord_at_index = z_coords[index]
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    # Check if the coordinate at the index is within the tolerance
    if not np.isclose(coord_at_index, location, atol=tolerance):
        raise ValueError(f"No slice found for location={location} on axis={axis}. Closest coordinate is {coord_at_index}")

    # Extract the 2D slice of velocities based on the axis
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

    # Normalize each component using soft clipping
    slice_u_norm = soft_clipping_normalize_component(slice_u, GLOBAL_UMIN_5TH, GLOBAL_UMAX_95TH,k = channel_target_steap[0], c=channel_target_mean[0])
    slice_v_norm = soft_clipping_normalize_component(slice_v, GLOBAL_VMIN_5TH, GLOBAL_VMAX_95TH, k = channel_target_steap[1], c=channel_target_mean[1])
    slice_w_norm = soft_clipping_normalize_component(slice_w, GLOBAL_WMIN_5TH, GLOBAL_WMAX_95TH, k = channel_target_steap[2], c=channel_target_mean[2])

    # Encode velocity components with directionality
    encoded_image = encode_velocity_components(slice_u_norm, slice_v_norm, slice_w_norm)

    # Enhance edges (optional)
    enhanced_image = encoded_image  # edge_enhancement_selective(encoded_image, sigma=1.0, amount=0.5, edge_threshold=0.1)

    print(f"Max U: {np.max(slice_u)}, Min U: {np.min(slice_u)}")
    print(f"Max V: {np.max(slice_v)}, Min V: {np.min(slice_v)}")
    print(f"Max W: {np.max(slice_w)}, Min W: {np.min(slice_w)}")
    global_min = min(np.min(slice_u), np.min(slice_v), np.min(slice_w))
    global_max = max(np.max(slice_u), np.max(slice_v), np.max(slice_w))
    print(f"Global Min Velocity: {global_min}, Global Max Velocity: {global_max}")

    return enhanced_image, slice_u, slice_v, slice_w, slice_mag



import seaborn as sns  # Make sure to install seaborn if you haven't: pip install seaborn

def process_and_save_images(total_cases=128, steps_range=range(31, 41), reduce_=1, axis=1, location=0.21277,
                           rgb_save_dir=RGB_SAVE_DIR, scalar_save_dir=SCALAR_SAVE_DIR):
    """
    Process and save RGB-encoded velocity images and scalar velocity magnitude images from CSV data.
    Additionally, collects normalized velocity components for distribution plotting.
    """
    os.makedirs(rgb_save_dir, exist_ok=True)
    os.makedirs(scalar_save_dir, exist_ok=True)
    saved_rgb_images = []
    saved_scalar_images = []
    
    # Initialize lists to collect normalized components
    all_u_norm = []
    all_v_norm = []
    all_w_norm = []

    all_u = []
    all_v = []
    all_w = []
    
    # Variables to accumulate total difference sum and total number of elements
    total_difference_sum = 0.0
    total_num_elements = 0

    for c in tqdm(range(total_cases), desc="Processing Cases"):  # Cases 0 to 127
        for s in steps_range:     # Steps 31 to 40
            casenum = c
            step = s
            try:
                # Get the slice data (enhanced RGB image and scalar magnitude)
                enhanced_image, slice_u, slice_v, slice_w, slice_mag = getSlice(casenum, step, axis, location, reduce_)

                # Append not normalized data to the lists
                all_u.append(slice_u.flatten())
                all_v.append(slice_v.flatten())
                all_w.append(slice_w.flatten())
                
                # Normalize the components
                slice_u_norm = soft_clipping_normalize_component(slice_u, GLOBAL_UMIN_5TH, GLOBAL_UMAX_95TH,k = channel_target_steap[0], c=channel_target_mean[0])
                slice_v_norm = soft_clipping_normalize_component(slice_v, GLOBAL_VMIN_5TH, GLOBAL_VMAX_95TH, k = channel_target_steap[1], c=channel_target_mean[1])
                slice_w_norm = soft_clipping_normalize_component(slice_w, GLOBAL_WMIN_5TH, GLOBAL_WMAX_95TH, k= channel_target_steap[2], c=channel_target_mean[2])
                
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
            rgb_save_path = os.path.join(rgb_save_dir, base_filename)
            scalar_save_path = os.path.join(scalar_save_dir, base_filename)

            # Save the RGB image using plt.imsave
            plt.imsave(rgb_save_path, enhanced_image, origin='lower')
            saved_rgb_images.append(rgb_save_path)
            print(f"Saved RGB image: {rgb_save_path}")

            # Decode the scalar magnitude from the enhanced image
            decoded_scalar_magnitude = decode_rgb_to_scalar(enhanced_image)
            difference = np.abs(decoded_scalar_magnitude - slice_mag)
            max_difference = np.max(difference)
            print(f"Maximum difference between original and decoded scalar magnitudes: {max_difference}")

            # Accumulate total difference sum and number of elements
            total_difference_sum += np.sum(difference)
            total_num_elements += difference.size

            # Save the scalar magnitude image using specified colormap and vmax matching the original data range
            plt.imsave(scalar_save_path, slice_mag, cmap='gist_rainbow_r', vmin=0, vmax=0.56, origin='lower')
            saved_scalar_images.append(scalar_save_path)
            print(f"Saved Scalar Magnitude image: {scalar_save_path}")

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

    # Calculate the overall mean difference across all data
    if total_num_elements > 0:
        overall_mean_difference = total_difference_sum / total_num_elements
        print(f"Overall mean difference across all data: {overall_mean_difference}")
    else:
        print("No data was processed, so the overall mean difference cannot be calculated.")
    
    print("Compent U mean: ", all_u_norm.mean())
    print("Compent U mean: ", all_v_norm.mean())
    print("Compent U mean: ", all_w_norm.mean())

    print("Finished processing and saving RGB and Scalar Magnitude images.")
    
    # Return the collected normalized components along with saved image paths
    return saved_rgb_images, saved_scalar_images, all_u_norm, all_v_norm, all_w_norm, all_u, all_v, all_w





def compute_global_robust_scaling(total_cases, steps_range, axis, location, 
                                  lower_percentile=0.05, upper_percentile=99.95):
    """
    Compute the global lower and upper percentiles for velocity components u, v, w
    across all data for robust scaling.

    Parameters:
    - total_cases: Total number of cases.
    - steps_range: Iterable of steps to process.
    - axis: Axis parameter (unused in this function, but kept for compatibility).
    - location: Location parameter (unused in this function, but kept for compatibility).
    - lower_percentile: Lower percentile for scaling (default=5).
    - upper_percentile: Upper percentile for scaling (default=95).

    Returns:
    - A dictionary containing lower and upper percentiles for each component.
      Example:
      {
          'u': {'lower': value, 'upper': value},
          'v': {'lower': value, 'upper': value},
          'w': {'lower': value, 'upper': value}
      }
    """
    
    # Initialize lists to collect all velocity component values
    u_values = []
    v_values = []
    w_values = []
    
    # Iterate through all cases and steps
    for c in tqdm(range(total_cases), desc="Processing cases"):
        for s in steps_range:
            casenum = c
            step = s
            csv_path = f"/home/vittorio/Scrivania/ResShift_4_scale/data/Final_Steps_47/{casenum}_{step}.csv"
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}. Skipping.")
                continue

            try:
                # Read the CSV file, skipping the first row if it's a header
                working_csv = pd.read_csv(csv_path, skiprows=[0]).to_numpy()

                # Extract the velocity components (u, v, w)
                vels = working_csv[:, 4:7]  # Assuming columns 4,5,6 are u, v, w

                # Append to respective lists
                u_values.append(vels[:, 0])
                v_values.append(vels[:, 1])
                w_values.append(vels[:, 2])
            except Exception as e:
                print(f"Error processing {csv_path}: {e}. Skipping.")
                continue

    # Concatenate all collected values
    if u_values:
        all_u = np.concatenate(u_values)
    else:
        all_u = np.array([])
    if v_values:
        all_v = np.concatenate(v_values)
    else:
        all_v = np.array([])
    if w_values:
        all_w = np.concatenate(w_values)
    else:
        all_w = np.array([])

    # Compute the lower and upper percentiles for each component
    scaling_params = {}
    for component, data in zip(['u', 'v', 'w'], [all_u, all_v, all_w]):
        if data.size == 0:
            print(f"No data collected for component {component}.")
            scaling_params[component] = {'lower': None, 'upper': None}
            continue
        lower = np.percentile(data, lower_percentile)
        upper = np.percentile(data, upper_percentile)
        scaling_params[component] = {'lower': lower, 'upper': upper}
        print(f"Computed Global {component.upper()} {lower_percentile}th Percentile: {lower}, {upper_percentile}th Percentile: {upper}")

    return scaling_params

def plot_normalized_distributions(u_norm, v_norm, w_norm, bins=100):
    """
    Plot the distribution of velocity components in a style similar to the example.

    Parameters:
    - u_norm, v_norm, w_norm: 1D NumPy arrays of normalized components.
    - bins: Number of bins for the histogram.
    """
    # Create a figure
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # Define velocity components and their labels
    components = [u_norm, v_norm, w_norm]
    labels = ['U Velocities', 'V Velocities', 'W Velocities']

    # Loop over components to create subplots
    for i, (comp, label) in enumerate(zip(components, labels)):
        axs[i].hist(comp, bins=bins, color='blue', edgecolor='black')
        axs[i].set_yscale('log')  # Logarithmic scale
        axs[i].set_title(label, fontsize=14)
        axs[i].set_ylabel('Frequency ', fontsize=12)

    # Set a shared X label
    axs[-1].set_xlabel('Velocity Values', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_violin_distributions(u_norm, v_norm, w_norm):
    """
    Plot violin plots of normalized velocity components.

    Parameters:
    - u_norm, v_norm, w_norm: 1D NumPy arrays of normalized components.
    """
    data = pd.DataFrame({
        'Component': ['u_norm'] * len(u_norm) + ['v_norm'] * len(v_norm) + ['w_norm'] * len(w_norm),
        'Value': np.concatenate([u_norm, v_norm, w_norm])
    })
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Component', y='Value', data=data, palette=['r', 'g', 'b'])
    plt.title('Violin Plots of Normalized Velocity Components')
    plt.xlabel('Velocity Component')
    plt.ylabel('Normalized Value')
    plt.tight_layout()
    plt.show()
    
    # Optionally, save the violin plot
    # plt.savefig('/path/to/save/violin_plots.png')




def main():
    # Parameters
    total_cases = 127  # From 0 to 127
    steps_range = range(31, 41)  # Time steps from 30 to 40
    reduce_ = 1
    axis = 1
    location = 0.21277

    #scaling_params = compute_global_robust_scaling(total_cases, steps_range, axis, location)

    
    # Proceed to process and save images
    saved_rgb_images, saved_scalar_images, all_u_norm, all_v_norm, all_w_norm, all_u, all_v, all_w = process_and_save_images(
        total_cases=total_cases, steps_range=steps_range, reduce_=reduce_, axis=axis, location=location)
    
    
    
    # Check if normalized data was collected
    if all_u_norm.size == 0 and all_v_norm.size == 0 and all_w_norm.size == 0:
        print("No normalized data available for plotting.")
        return




if __name__ == "__main__":
    # Step 1: Process and save RGB and Scalar Magnitude images
    main()


