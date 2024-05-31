import os
import shutil
import random

def prepare_data(src, dest, split_ratio=0.8):
    categories = ['Cat', 'Dog']
    
    # Create directories for training and validation sets
    for category in categories:
        os.makedirs(os.path.join(dest, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(dest, 'validation', category), exist_ok=True)

    for category in categories:
        category_path = os.path.join(src, category)
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(files)
        split_point = int(len(files) * split_ratio)
        
        train_files = files[:split_point]
        val_files = files[split_point:]
        
        for file in train_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(dest, 'train', category, file))
        
        for file in val_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(dest, 'validation', category, file))

if __name__ == "__main__":
    src_dir = 'PetImages'  # Source directory containing 'Cat' and 'Dog' subdirectories
    dest_dir = 'data'      # Destination directory for training and validation sets
    prepare_data(src_dir, dest_dir)
