import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# Define your folder containing HDF5 files
folder_path = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_obstacles/hdf-files"
output_folder = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_obstacles/2d_slices"
output_csv_folder = os.path.join(output_folder, "csv_data")
output_geom_folder = os.path.join(output_folder, "geometry_slices") # Optional: separate folder for geometry

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_csv_folder, exist_ok=True)
os.makedirs(output_geom_folder, exist_ok=True) # Create geometry output folder

# --- IMPORTANT: Replace with the actual key for your geometry data ---
GEOMETRY_DATA_KEY = "data_A"  # <-- CHANGE THIS KEY NAME AS NEEDED

# limit the range of velocities if needed
vmin_vel, vmax_vel = 0, 0.56
# Optional: Define vmin/vmax for geometry visualization if needed
vmin_geom, vmax_geom = 0, 1 # Example: Assuming geometry is like a mask (0 or 1) or SDF

# Second pass: Generate and save images
filenames = [f for f in os.listdir(folder_path) if f.endswith(".h5")]
for filename in tqdm(filenames, desc="Processing HDF5 files"):
    file_path = os.path.join(folder_path, filename)
    file_number = os.path.splitext(filename)[0] # Extract numeric part

    with h5py.File(file_path, "r") as f:
        # --- Load Velocity Data ---
        data_B = f["data_B"][:]

        # --- Load Geometry Data ---
        if GEOMETRY_DATA_KEY in f:
            geometry_data = f[GEOMETRY_DATA_KEY][:]
            has_geometry = True
        else:
            print(f"Warning: Geometry key '{GEOMETRY_DATA_KEY}' not found in {filename}")
            has_geometry = False
            geometry_data = None # Set to None if not found

    # --- Process Velocity Data ---
    # Reorder the data and calculate the magnitude of velocity
    velocity_magnitude = np.linalg.norm(data_B.reshape((64, 64, 64, 3), order='C'), axis=-1)

    # Get center slices for velocity
    vel_center_x = velocity_magnitude[velocity_magnitude.shape[0] // 2, :, :]
    vel_center_y = velocity_magnitude[:, velocity_magnitude.shape[1] // 2, :]
    vel_center_z = velocity_magnitude[:, :, velocity_magnitude.shape[2] // 2]

    # --- Process Geometry Data (if loaded) ---
    if has_geometry:
        # Assuming geometry_data is also a 3D grid (e.g., 64x64x64)
        # Adjust slicing if geometry_data has a different shape
        if len(geometry_data.shape) == 3 and geometry_data.shape[0] >= 2 and geometry_data.shape[1] >= 2 and geometry_data.shape[2] >= 2 :
             geom_center_x = geometry_data[geometry_data.shape[0] // 2, :, :]
             geom_center_y = geometry_data[:, geometry_data.shape[1] // 2, :]
             geom_center_z = geometry_data[:, :, geometry_data.shape[2] // 2]
        elif len(geometry_data.shape) == 2: # If it's already 2D, maybe use it directly?
             print(f"Warning: Geometry data in {filename} is 2D, handling might need adjustment.")
             # Decide how to handle 2D geometry data if necessary
             geom_center_x, geom_center_y, geom_center_z = None, None, None # Placeholder
        else:
             print(f"Warning: Unexpected geometry data shape {geometry_data.shape} in {filename}")
             geom_center_x, geom_center_y, geom_center_z = None, None, None # Placeholder


    # --- Function to save velocity images and CSV data ---
    def save_velocity_image_and_csv(slice_data, plane, reduction_factor):
        if reduction_factor == 1:
            # No downsampling, save the full 64x64 slice
            output_path_img = os.path.join(output_folder, f"{file_number}_vel_{plane}_{reduction_factor}.jpg")
            plt.imsave(output_path_img, slice_data, cmap='gist_rainbow_r', origin="lower", vmin=vmin_vel, vmax=vmax_vel)

            # Save corresponding CSV file
            output_path_csv = os.path.join(output_csv_folder, f"{file_number}_vel_{plane}_{reduction_factor}.csv")
            with open(output_path_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(slice_data)

        elif reduction_factor > 1:
            downsampled = slice_data[::reduction_factor, ::reduction_factor] # Downsampling
            output_path_img = os.path.join(output_folder, f"{file_number}_vel_{plane}_{reduction_factor}.jpg")
            plt.imsave(output_path_img, downsampled, cmap='gist_rainbow_r', origin="lower", vmin=vmin_vel, vmax=vmax_vel)

    # --- Function to save low-res velocity images ---
    def save_low_res_velocity_image(slice_data, plane, reduction_factor):
         if slice_data is not None and reduction_factor > 1:
            downsampled = slice_data[::reduction_factor, ::reduction_factor]
            output_path_img = os.path.join(output_folder, f"{file_number}_vel_{plane}_{reduction_factor}.jpg")
            plt.imsave(output_path_img, downsampled, cmap='gist_rainbow_r', origin="lower", vmin=vmin_vel, vmax=vmax_vel)

    # --- Function to save geometry images (Example: saving full-res only) ---
    def save_geometry_image(slice_data, plane):
         if slice_data is not None:
            output_path_img = os.path.join(output_geom_folder, f"{file_number}_{plane}.jpg")
            # Use a different colormap if desired (e.g., 'gray' for masks/SDF)
            plt.imsave(output_path_img, slice_data, cmap='gray', origin="lower", vmin=vmin_geom, vmax=vmax_geom)


    # --- Save Velocity Images and CSV data ---
    save_velocity_image_and_csv(vel_center_x, "x", 1)
    save_velocity_image_and_csv(vel_center_y, "y", 1)
    save_velocity_image_and_csv(vel_center_z, "z", 1)

    # Optionally save lower resolution velocity images
    for reduction in [2, 4]:
        save_low_res_velocity_image(vel_center_x, "x", reduction)
        save_low_res_velocity_image(vel_center_y, "y", reduction)
        save_low_res_velocity_image(vel_center_z, "z", reduction)

    # --- Save Geometry Images (if loaded and processed) ---
    if has_geometry:
        save_geometry_image(geom_center_x, "x")
        save_geometry_image(geom_center_y, "y")
        save_geometry_image(geom_center_z, "z")
        # Add logic here if you want downsampled geometry images or geometry CSVs

print("Processing complete. Images and CSV files saved.")