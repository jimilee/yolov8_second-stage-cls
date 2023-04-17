from ultralytics import YOLO

# model = YOLO("pre/yolov8m.pt")  # load a pretrained model (recommended for training)
# results = model.train(data="emb.yaml", epochs=500)  # train the model
# metrics = model.val(data="emb.yaml", save=True)

import os

import cv2

from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

def get_labelfile(path):
    label_path = (path.split('/'))
    label_path[-3] = 'labels'
    label_path[-1] = label_path[-1][:-4] + '.txt'
    return '/'.join(label_path)

class Detector:
    def __init__(self, weight):
        self.res_path = '/runs/detect/'
        if os.path.isdir(self.res_path + 'predict/'):
            try:
                for file in os.scandir(self.res_path + 'predict/'):
                    os.remove(file.path)
                os.rmdir(self.res_path + 'predict/')
            except OSError as e:
                print("Error: %s : %s" % (self.res_path + 'predict/', e.strerror))

        self.model = YOLO(weight)  # load a model

    def infer_detect(self, path ='./dataset/EMB_dataset', target_labels=[]): # ./dataset/EMB_dataset
        if os.path.isdir(self.res_path+'predict/'):
            try:
                for file in os.scandir(self.res_path+'predict/'):
                    os.remove(file.path)
                os.rmdir(self.res_path+'predict/')
            except OSError as e:
                print("Error: %s : %s" % (self.res_path+'predict/', e.strerror))

        total = 0
        cnt_true = 0
        for x,y,z in tqdm(os.walk(f"{path}/images")): # iter per image
            for f in z:
                img_path = f'{x}/{f}'
                label_path = get_labelfile(img_path)
                if img_path.endswith(('.jpg', '.JPG','.jpeg')):
                    # label txt 확인
                    if os.path.isfile(label_path):
                        results = self.model(img_path, conf=0.5, save=False, verbose=False, target_labels=target_labels)  # infer model
                        for r in results:
                            if r.category in target_labels:
                                total += 1
                                cnt_true += r.result

        # for x,y,z in tqdm(os.walk(res_path)):
        #     for f in z:
        #         img_path = f'{x}/{f}'
        #         if img_path.endswith(('.jpg', '.JPG','.jpeg')):
        #             cnt_false += 1
        print(f'Total test images : {total} , Total False : {cnt_true}, Acc : {cnt_true/total}')
        return cnt_true/total

if __name__ == '__main__':
    # opt = parse_opt()
    detector = Detector(weight='weight.pt')
    detector.infer_detect()
