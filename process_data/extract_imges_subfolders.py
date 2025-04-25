import os
import shutil

def organize_images_by_suffix(source_dir):
    """
    Creates subdirectories (1, 2, 4) within the source directory,
    copies images into them based on their suffix (_1, _2, _4)
    before the file extension, and removes the suffix from the copied filename.

    Args:
        source_dir (str): The path to the directory containing the images.
    """

    # Define the suffixes and their corresponding subfolder names
    suffix_map = {
        "_1": "1",
        "_2": "2",
        "_4": "4"
    }

    # Create the subfolders if they don't exist
    for folder_name in suffix_map.values():
        os.makedirs(os.path.join(source_dir, folder_name), exist_ok=True)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, filename)):
            name, ext = os.path.splitext(filename)  # Split filename into name and extension
            copied = False
            for suffix, folder_name in suffix_map.items():
                if name.endswith(suffix):
                    new_filename = name[:-len(suffix)] + ext  # Remove the suffix
                    source_path = os.path.join(source_dir, filename)
                    destination_path = os.path.join(source_dir, folder_name, new_filename)
                    try:
                        shutil.copy2(source_path, destination_path)
                        print(f"Copied '{filename}' as '{new_filename}' to folder '{folder_name}'")
                        copied = True
                        break  # Move to the next file once copied
                    except Exception as e:
                        print(f"Error copying '{filename}': {e}")
            if not copied:
                print(f"Skipped '{filename}': Suffix not recognized for organization.")

def organize_subfolder_images(parent_dir):
    """
    Divides images within subfolders (1, 2, 4) of the parent directory
    into new subfolders (X, Y, Z) based on their new suffix (_x, _y, _z),
    and moves the images to the corresponding new subfolder.

    Args:
        parent_dir (str): The path to the parent directory containing the subfolders 1, 2, and 4.
    """

    target_subfolders = ["geometry_slices"]
    suffix_map = {
        "_x": "X",
        "_y": "Y",
        "_z": "Z"
    }

    for subfolder_name in target_subfolders:
        subfolder_path = os.path.join(parent_dir, subfolder_name)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder_name}")

            # Create the destination subfolders (X, Y, Z) if they don't exist within the current subfolder
            for new_folder_name in suffix_map.values():
                os.makedirs(os.path.join(subfolder_path, new_folder_name), exist_ok=True)

            # Iterate through all files in the current subfolder
            for filename in os.listdir(subfolder_path):
                if os.path.isfile(os.path.join(subfolder_path, filename)):
                    name, ext = os.path.splitext(filename)
                    moved = False
                    for suffix, new_folder_name in suffix_map.items():
                        if name.endswith(suffix):
                            source_path = os.path.join(subfolder_path, filename)
                            destination_path = os.path.join(subfolder_path, new_folder_name, filename)
                            try:
                                shutil.move(source_path, destination_path)
                                print(f"Moved '{filename}' from '{subfolder_name}' to '{new_folder_name}' in '{subfolder_name}'")
                                moved = True
                                break  # Move to the next file once moved
                            except Exception as e:
                                print(f"Error moving '{filename}' from '{subfolder_name}': {e}")
                    if not moved:
                        print(f"Skipped '{filename}' in '{subfolder_name}': Suffix not recognized for organization (X, Y, Z).")
        else:
            print(f"Subfolder '{subfolder_name}' not found in the parent directory.")

if __name__ == "__main__":
    parent_directory = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_obstacles/2d_slices" # **<-- MODIFY THIS LINE**
    organize_subfolder_images(parent_directory)
    print("Second-level image organization complete.")