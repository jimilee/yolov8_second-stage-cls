import os

import cv2

from ultralytics import YOLO



model = YOLO("best.pt")  # load a pretrained model (recommended for training)

path = '/home/jml/_workspace/yolov8_emb/ultralytics/yolo/data/datasets/EMB_dataset/images/' # test image folder.
for x,y,z in os.walk(path):
    for f in z:
        img_path = f'{x}/{f}'
        if img_path.endswith('.jpg'):
            results = model(img_path, conf=0.35, save=True)  # train the model
            for r in results:
                boxes = r.boxes
                cls = r.cls
                res = r.res

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(len(results[0]))
            # if len(results[0]) != 0:
            #     src = cv2.imread(img_path)
            #     src = cv2.resize(src, (640,480))
            #     print(type(results[0]))
            #     cv2.imshow("src ", src)
            #     # cv2.imshow("res ",results[0])
            #     cv2.waitKey(0)