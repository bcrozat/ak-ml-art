from ultralytics import YOLO

# Load model
model = YOLO("models/yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolon11.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="dataset/Bubbles.v1i.yolov11/data.yaml", epochs=100, imgsz=640)
# results = model.train(data="<yourdatasetfolder/data.yaml>", epochs=100, imgsz=640)