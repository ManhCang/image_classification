from PIL import Image
import os

def remove_corrupted_images(directory):
    for category in ['Cat', 'Dog']:
        folder_path = os.path.join(directory, category)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify that it is, in fact, an image
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted file: {file_path}. Error: {e}")
                os.remove(file_path)

# Run the script for both training and validation directories
remove_corrupted_images('data/train')
remove_corrupted_images('data/validation')
