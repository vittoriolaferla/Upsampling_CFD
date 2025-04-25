import os
import shutil
import random

def create_paired_dataset_with_geometry(hr_dir, lw_dir, csv_dir, geometry_dir, output_dir, split=0.8):
    """
    Creates a paired image, CSV, and geometry dataset with train and test splits.

    Args:
        hr_dir (str): Path to the directory containing all HR images.
        lw_dir (str): Path to the directory containing all LW images.
        csv_dir (str): Path to the directory containing corresponding CSV files.
        geometry_dir (str): Path to the directory containing geometry information files (images).
        output_dir (str): Path to the directory where the train and test folders will be created.
                           It will contain HR, LW, CSV, and Geometry subdirectories within train and test.
        split (float): The proportion of data to use for the training set (default: 0.8).
    """

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    # Create subdirectories for HR, LW, CSV, and Geometry in train and test
    os.makedirs(os.path.join(train_dir, "HR"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "LW"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "CSV"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "Geometry"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "HR"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "LW"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "CSV"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "Geometry"), exist_ok=True)

    hr_filenames = set(os.listdir(hr_dir))
    lw_filenames = set(os.listdir(lw_dir))
    csv_filenames = set(os.listdir(csv_dir))
    geometry_filenames = set(os.listdir(geometry_dir))

    # Assuming CSV and Geometry filenames have the same base name as image filenames
    image_basenames = {os.path.splitext(f)[0] for f in hr_filenames.intersection(lw_filenames)}
    csv_basenames = {os.path.splitext(f)[0] for f in csv_filenames}
    geometry_basenames = {os.path.splitext(f)[0] for f in geometry_filenames}

    common_basenames = list(image_basenames.intersection(csv_basenames).intersection(geometry_basenames))

    if not common_basenames:
        print("Error: No common base filenames found between HR/LW images, CSV files, and Geometry files.")
        return

    random.shuffle(common_basenames)
    num_train = int(len(common_basenames) * split)
    train_basenames = common_basenames[:num_train]
    test_basenames = common_basenames[num_train:]

    print(f"Found {len(common_basenames)} common base filenames (image pairs, CSVs, and Geometry).")
    print(f"Splitting into {len(train_basenames)} training sets and {len(test_basenames)} testing sets.")

    def copy_files(basenames, source_hr_dir, source_lw_dir, source_csv_dir, source_geometry_dir,
                   dest_hr_dir, dest_lw_dir, dest_csv_dir, dest_geometry_dir, dataset_type):
        for basename in basenames:
            hr_filename = basename + ".jpg"  # Assuming .jpg for HR/LW
            lw_filename = basename + ".jpg"
            csv_filename = basename + ".csv"  # Assuming .csv for CSV
            geometry_filename = basename + ".jpg"  # Assuming .jpg for Geometry (was .npy)

            hr_src_path = os.path.join(source_hr_dir, hr_filename)
            lw_src_path = os.path.join(source_lw_dir, lw_filename)
            csv_src_path = os.path.join(source_csv_dir, csv_filename)
            geometry_src_path = os.path.join(source_geometry_dir, geometry_filename)

            hr_dest_path = os.path.join(dest_hr_dir, hr_filename)
            lw_dest_path = os.path.join(dest_lw_dir, lw_filename)
            csv_dest_path = os.path.join(dest_csv_dir, csv_filename)
            geometry_dest_path = os.path.join(dest_geometry_dir, geometry_filename)

            try:
                shutil.copy2(hr_src_path, hr_dest_path)
                shutil.copy2(lw_src_path, lw_dest_path)
                shutil.copy2(csv_src_path, csv_dest_path)
                shutil.copy2(geometry_src_path, geometry_dest_path)
                print(f"Copied {dataset_type} set: {basename}")
            except FileNotFoundError as e:
                print(f"Error: Could not find one or more files for {basename}: {e}")
            except Exception as e:
                print(f"Error copying {dataset_type} set {basename}: {e}")

    print("\nCopying training data...")
    copy_files(train_basenames, hr_dir, lw_dir, csv_dir, geometry_dir,
               os.path.join(train_dir, "HR"), os.path.join(train_dir, "LW"), os.path.join(train_dir, "CSV"),
               os.path.join(train_dir, "Geometry"), "train")

    print("\nCopying testing data...")
    copy_files(test_basenames, hr_dir, lw_dir, csv_dir, geometry_dir,
               os.path.join(test_dir, "HR"), os.path.join(test_dir, "LW"), os.path.join(test_dir, "CSV"),
               os.path.join(test_dir, "Geometry"), "test")

    print("\nPaired image, CSV, and Geometry dataset creation complete.")

if __name__ == "__main__":
    # ** SET YOUR DIRECTORY PATHS HERE **
    hr_images_directory = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_obstacles/2d_slices/No_Downsample/Y"
    lw_images_directory = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_obstacles/2d_slices/2_Downsample/Y"
    csv_files_directory = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_obstacles/2d_slices/csv_data/Y"
    geometry_directory = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_obstacles/2d_slices/geometry_slices/Y"   # Add this line
    output_dataset_directory = "/home/vittorio/Scrivania/ETH/Upsampling/DAT/datasets/dataset_csv_geometry_Y"  # Change this line
    train_test_split = 0.85  # You can change the split ratio here



    create_paired_dataset_with_geometry(hr_images_directory, lw_images_directory, csv_files_directory, geometry_directory, output_dataset_directory, train_test_split)
