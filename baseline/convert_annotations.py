import os
import xml.etree.ElementTree as ET
from pathlib import Path

data_dir = Path('/Users/muzafarov/Desktop/Datasets/SafetyHelmetDetection')
annotations_dir = data_dir / 'annotations'
images_dir = data_dir / 'images'
output_dir = data_dir / 'labels' 
os.makedirs(output_dir, exist_ok=True)

classes = ['helmet', 'head', 'person']

def convert_annotation(file_path):
    """Конвертирует XML аннотацию в YOLO формат."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)
    
    yolo_annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

for xml_file in annotations_dir.glob('*.xml'):
    yolo_annotations = convert_annotation(xml_file)
    if yolo_annotations:
        output_file = output_dir / (xml_file.stem + '.txt')
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_annotations)) 