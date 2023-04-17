from ultralytics import YOLO

model = YOLO("pre/yolov8m.pt")  # load a pretrained model (recommended for training)
results = model.train(data="coco.yaml", epochs=300)
metrics = model.val(data="coco.yaml", save=True)