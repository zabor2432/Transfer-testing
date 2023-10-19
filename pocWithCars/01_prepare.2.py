import os
import shutil
import random
from tqdm import tqdm


def transform_data_structure(source_dir, target_dir):
    # Create target directories if they don't exist
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'testTT'), exist_ok=True)

    class_folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]

    for class_folder in class_folders:

        class_path = os.path.join(source_dir, class_folder)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_images = len(images)

        random.shuffle(images)  # Shuffle the list of images

        train_split = int(0.6 * num_images)
        val_split = int(0.2 * num_images)
        test_split = int(0.2 * num_images)  # Smaller test set 

        all_images = images[:train_split + val_split + test_split] # List slicing, [od:do:krok (def 1)]

        random.shuffle(all_images)  # Shuffle the images

        train_images = all_images[:train_split]
        val_images = all_images[train_split:train_split + val_split]
        test_images = all_images[train_split + val_split:]

        if class_folder != "white":
            class_target_dir = os.path.join(target_dir, 'train', class_folder)
            os.makedirs(class_target_dir, exist_ok=True)
            print(f"preparing data for train split for {class_folder} subset")
            for img in tqdm(train_images):
                shutil.copy2(os.path.join(class_path, img), class_target_dir)

            class_target_dir = os.path.join(target_dir, 'test', class_folder)
            os.makedirs(class_target_dir, exist_ok=True)
            print(f"preparing data for test split for {class_folder} subset")
            for img in tqdm(test_images):
                shutil.copy2(os.path.join(class_path, img), class_target_dir)

            os.makedirs(os.path.join(target_dir, 'testTT', class_folder), exist_ok=True)

        class_target_dir = os.path.join(target_dir, 'val', class_folder)
        os.makedirs(class_target_dir, exist_ok=True)
        print(f"preparing data for val split for {class_folder} subset")
        for img in tqdm(val_images):
            shutil.copy2(os.path.join(class_path, img), class_target_dir)

        if class_folder == "white":
            class_target_dir = os.path.join(target_dir, 'testTT', class_folder)
            os.makedirs(class_target_dir, exist_ok=True)
            print(f"preparing data for test split for {class_folder} subset")
            for img in tqdm(test_images):
                shutil.copy2(os.path.join(class_path, img), class_target_dir)
            
            os.makedirs(os.path.join(target_dir, 'test', class_folder), exist_ok=True)

        


if __name__ == "__main__":
    random.seed(2137)

    source_directory = "/home/macierz/s177788/projektBadawczy/Transfer-testing/pocWithCars/Transfer_testing_db/TT_DB"  # Update this to your source directory
    target_directory = "data"  # Update this to your desired target directory
    transform_data_structure(source_directory, target_directory)