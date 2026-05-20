from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
)