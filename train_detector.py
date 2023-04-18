from ultralytics import YOLO

# train YOLO
model = YOLO("pre/yolov8m.pt")  # load a pretrained model (recommended for training)
results = model.train(data="datautils/coco128.yaml", sub=False, epochs=300)
metrics = model.val(data="datautils/coco128.yaml", sub=False, save=True)