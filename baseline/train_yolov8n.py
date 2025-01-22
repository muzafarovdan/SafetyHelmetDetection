from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="baseline/dataset.yaml", epochs=20, imgsz=415)

results.save("baseline/results") 