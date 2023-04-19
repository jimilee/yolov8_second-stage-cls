import os
import cv2
import glob
from tqdm import tqdm
import xml.etree.ElementTree as elemTree
import pandas as pd
import numpy as np
import shutil

def prepare_dataset(path, split=0.2):
    try:
        path_l, labels = [], set()
        for p in glob.glob(path+'/*/*.jpg'):
            labels.add(str(p).split('/')[-2])
            path_l.append(p)

        total_len = len(path_l)
        print(f'Total dataset len : {total_len}, label len : {len(labels)}')

        df = pd.DataFrame(path_l)
        df1 = df.iloc[np.random.permutation(df.index)].reset_index(drop=True)

        x_train, x_valid = df1[:int(total_len*(1-split))].reset_index(drop=True), df1[int(total_len*(1-split)):].reset_index(drop=True)
        print(f'--- train data len : {len(x_train)}, valid data len : {len(x_valid)}')
        train_path = f'{path}/train/'
        valid_path = f'{path}/valid/'

        if not os.path.isdir(train_path) or not os.path.isdir(valid_path):
            os.mkdir(train_path)
            os.mkdir(valid_path)
        for l in labels: # make label folders
            os.mkdir(train_path+l)
            os.mkdir(valid_path+l)

        for i, t_p in x_train.iterrows():
            l_f, f = t_p[0].split('/')[-2:]
            if not os.path.isdir(os.path.join(train_path, l_f)):
                os.makedirs(os.path.join(train_path, l_f), exist_ok = True)

            shutil.move(t_p[0], os.path.join(train_path, l_f, f))

        for i, t_p in x_valid.iterrows():
            l_f, f = t_p[0].split('/')[-2:]
            if not os.path.isdir(os.path.join(valid_path, l_f)):
                os.makedirs(os.path.join(valid_path, l_f), exist_ok=True)

            shutil.move(t_p[0], os.path.join(valid_path, l_f, f))

        for l in labels:  # rm label folders
            try:
                os.rmdir(f'{path}/{l}/')
            except:
                continue
        return True
    except:
        return False
        # l_f = t_p.split('/')[-2]
        # if not os.path.isdir(os.path.join(train_path, l_f)):
        #     os.mkdir(os.path.join(train_path, l_f))
        # print(l_f)



def make_cropped_dataset(annot_path, image_path, result_path):
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    else:
        shutil.rmtree(result_path)
    for _label in tqdm(os.listdir(image_path)):
        if _label[:2] == 'n0': _label = _label[10:]
        save_p = os.path.join(result_path, _label)
        if not os.path.isdir(save_p):
            os.mkdir(save_p)

        for _imp,_anp in zip(os.listdir(image_path+_label), os.listdir(annot_path+_label)):
            imp = os.path.join(image_path,_label,_imp)
            anp = os.path.join(annot_path,_label,_imp[:-4])

            tree = elemTree.fromstring(open(anp,'r').read())
            anot = tree.findall("object")
            objects = [[x.find('bndbox').findtext(t) for t in ['xmin', 'ymin', 'xmax', 'ymax']] for x in anot]

            image = cv2.imread(imp)
            for i, bbox in enumerate(objects):
                x1,y1, x2,y2 = map(int, bbox)
                crop = image[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(save_p, _imp[:-4]+f'_{i}.jpg'), crop)

        if _label[:2]=='n0':
            os.rename(os.path.join(image_path,_label), os.path.join(image_path,_label[10:]))
            os.rename(os.path.join(annot_path,_label), os.path.join(annot_path,_label[10:]))


if __name__ == '__main__':
    annot_path = '../datasets/stanford_dogs/Annotation/'
    image_path = '../datasets/stanford_dogs/Images/'
    result_path = '../datasets/stanford_dogs/Cropped/'
    make_cropped_dataset(annot_path, image_path, result_path)
    _ = prepare_dataset(result_path)
    print(f"prepare dataset done. (status:{_})")