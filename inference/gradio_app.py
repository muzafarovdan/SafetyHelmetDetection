import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
from pathlib import Path

weights_path = Path('weights/yolov8n_baseline.pt')

model = YOLO(weights_path)

class_names = ['helmet', 'head', 'person']

def predict_and_draw(image):
    if image is None or image.size == 0:
        raise ValueError("Изображение не загружено. Пожалуйста, проверьте файл и попробуйте снова.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = f"{class_names[cls]}: {conf:.2f}"
            
            if class_names[cls] == 'helmet':
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)
    
    return image

iface = gr.Interface(
    fn=predict_and_draw,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Safety Helmet Detection",
    description="Загрузите изображение, чтобы получить предсказания модели YOLOv8n с отрисованными bounding box."
)

if __name__ == "__main__":
    iface.launch(share=True)