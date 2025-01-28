import os
import shutil
from pathlib import Path
import random

data_dir = Path('/Users/muzafarov/Desktop/Datasets/SafetyHelmetDetection')

images_dir = data_dir / 'images'
labels_dir = data_dir / 'labels'

train_images_dir = data_dir / 'train' / 'images'
val_images_dir = data_dir / 'val' / 'images'
test_images_dir = data_dir / 'test' / 'images'
train_labels_dir = data_dir / 'train' / 'labels'
val_labels_dir = data_dir / 'val' / 'labels'
test_labels_dir = data_dir / 'test' / 'labels'

for directory in [train_images_dir, val_images_dir, test_images_dir, train_labels_dir, val_labels_dir, test_labels_dir]:
    os.makedirs(directory, exist_ok=True)

image_files = list(images_dir.glob('*.png'))  # Измените расширение, если необходимо

print(f"Найдено {len(image_files)} изображений.")

random.shuffle(image_files)

train_files = image_files[:4000]
val_files = image_files[4000:4500]
test_files = image_files[4500:]

def copy_files(file_list, dest_images_dir, dest_labels_dir):
    for image_file in file_list:
        shutil.copy(image_file, dest_images_dir)
        
        label_file = labels_dir / (image_file.stem + '.txt')
        if label_file.exists():
            shutil.copy(label_file, dest_labels_dir)
        else:
            print(f"Аннотация для {image_file.name} не найдена.")

copy_files(train_files, train_images_dir, train_labels_dir)
copy_files(val_files, val_images_dir, val_labels_dir)
copy_files(test_files, test_images_dir, test_labels_dir)

print(f"Данные разделены: {len(train_files)} для обучения, {len(val_files)} для валидации, {len(test_files)} для тестирования.") 