from ultralytics import YOLO

# Load a model
model = YOLO("yolo12m.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)