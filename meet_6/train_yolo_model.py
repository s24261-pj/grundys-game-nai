from ultralytics import YOLO

model = YOLO("yolov5n.pt")

model.info()

results = model.train(data="./dataset/data.yaml", epochs=150, imgsz=640, project=".", name="runs")
