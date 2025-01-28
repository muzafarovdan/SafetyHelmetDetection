from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

app = Flask(__name__)

weights_path = Path('weights/yolov8n_baseline.pt')

model = YOLO(weights_path)

class_names = ['helmet', 'head', 'person']

def run_inference(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    predictions = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            predictions.append({
                "class": class_names[cls],
                "confidence": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    predictions = run_inference(image)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)