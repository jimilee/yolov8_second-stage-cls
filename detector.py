from ultralytics import YOLO

import os

import cv2

from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from ultralytics.yolo.utils import yaml_load
def get_labelfile(path):
    label_path = (path.split('/'))
    label_path[-3] = 'labels'
    label_path[-1] = label_path[-1][:-4] + '.txt'
    return '/'.join(label_path)

class Detector:
    def __init__(self, weight, cfg):
        self.res_path = '/runs/detect/'
        if os.path.isdir(self.res_path + 'predict/'):
            try:
                for file in os.scandir(self.res_path + 'predict/'):
                    os.remove(file.path)
                os.rmdir(self.res_path + 'predict/')
            except OSError as e:
                print("Error: %s : %s" % (self.res_path + 'predict/', e.strerror))

        self.model = YOLO(weight)  # load a model
        self.cfg = yaml_load(cfg)


    def infer_detect(self, path ='./sub_dataset/', target_labels=[]): # ./sub_dataset/EMB_dataset
        if os.path.isdir(self.res_path+'predict/'):
            try:
                for file in os.scandir(self.res_path+'predict/'):
                    os.remove(file.path)
                os.rmdir(self.res_path+'predict/')
            except OSError as e:
                print("Error: %s : %s" % (self.res_path+'predict/', e.strerror))

        total = 0
        cnt_true = 0
        for x,y,z in tqdm(os.walk(f"{path}")): # iter per image
            for f in z:
                img_path = f'{x}/{f}'
                if img_path.endswith(('.jpg', '.JPG','.jpeg')):
                    results = self.model(img_path, conf=0.5, save=True, verbose=False,
                                         target_labels=target_labels,
                                         sub_names=self.cfg['sub_names'],
                                         sub_data=self.cfg['sub_data'],
                                         sub_model=self.cfg['sub_model'])  # infer model
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
    detector = Detector(weight='yolov8m.pt', cfg='ultralytics/yolo/cfg/stanford_dogs.yaml')
    detector.infer_detect(path='datasets/stanford_dogs/Images', target_labels=[16])
