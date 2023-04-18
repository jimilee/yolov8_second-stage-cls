import os

import torch.utils.data
import numpy as np
import pandas as pd
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import transforms as transforms_
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skimage import io, transform, color

import cv2
#
# from train_Oxy import OxyClassifier

def img_Contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

def get_black_label(image, x, y):
    h,w,_ = image.shape
    image = cv2.resize(image, (x, y))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    out = gray.copy()
    out = 255 - out
    out = np.where(out > 160, 255, 0)

    out = out.astype("uint8")
    # 모폴로지 침식
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.dilate(out, k,2)

    # 결과 출력
    # merged = np.hstack((out, erosion))
    out = erosion
    # cv2.imshow('Erode', merged)
    # cv2.waitKey(0)

    ret, img_binary = cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for cnt in contours:
    #     cv2.drawContours(image, [cnt], 0, (255, 0, 0), 3)

    result = []
    scores = []
    for cnt in contours:
        epsilon = 0.2 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # cv2.drawContours(image, [cnt], 0, (255, 0, 0), 2)
        if len(approx) >= 2:
            rect = cv2.minAreaRect(cnt)
            # print(rect[1])
            if rect[1][0] > x and rect[1][1] > y:
                # print('+'*100)
                # print(len(approx))
                box = cv2.boxPoints(rect)  # 중심점과 각도를 4개의 꼭지점 좌표로 변환
                box = np.int0(box)  # 정수로 변환
                width = int(rect[1][0])
                height = int(rect[1][1])
                src_pts = box.astype("float32")
                # coordinate of the points in box points after the rectangle has been
                # straightened
                dst_pts = np.array([[0, height - 2],
                                    [0, 0],
                                    [width - 2, 0],
                                    [width - 2, height - 2]], dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                # directly warp the rotated rectangle to get the straightened rectangle

                tmp = cv2.warpPerspective(out, M, (width, height))
                tmp = cv2.resize(tmp, (15, 15))

                # tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)
                # print((tmp==0).sum())
                if (tmp==0).sum() < 60:
                    # cv2.imshow(str((tmp == 0).sum()), tmp)
                    # cv2.waitKey(0)
                    warped = cv2.warpPerspective(image, M, (width, height))
                    if warped.shape[0] > warped.shape[1]:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    warped = img_Contrast(warped)
                    # warped = cv2.resize(warped, (w * 2, h * 2))
                    result.append(warped)
                    scores.append((tmp==0).sum())
    if len(result) > 1 : fin = result[int(scores.index(min(scores)))]
    elif len(result) == 0 : fin = None
    else : fin = result[0]
    # print(type(fin))
    # cv2.imshow('fin', fin)
    # cv2.waitKey(0)
    return fin

def data_transforms_img(img_size):
    data_transforms = {
        "train":
            A.Compose([
            # A.HueSaturationValue(
            #     hue_shift_limit=0.2,
            #     sat_shift_limit=0.2,
            #     val_shift_limit=0.2,
            #     p=0.5
            # ),
            # A.CLAHE(always_apply=False, p=0.5, clip_limit=(1, 5), tile_grid_size=(8, 8)),
            # A.GridDistortion(always_apply=False, p=0.5, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=0,
            #                border_mode=0, value=(0, 0, 0), mask_value=None, normalized=False),
            A.LongestMaxSize(max_size=int(img_size * 1.0)),
            # min_size보다 작으면 pad
            A.PadIfNeeded(min_height=int(img_size * 1.0), min_width=int(img_size * 1.0),
                          border_mode=cv2.BORDER_CONSTANT),
            # A.Resize(img_size, img_size),
            # A.RandomSizedCrop(always_apply=False, p=0.5, min_max_height=(img_size-20, img_size), height=img_size, width=img_size, w2h_ratio=1.0,
            #                 interpolation=0),
            # A.Resize(img_size, img_size),
            # # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지.
            # A.LongestMaxSize(max_size=int(img_size * 1.0)),
            # # min_size보다 작으면 pad
            # A.PadIfNeeded(min_height=int(img_size * 1.0), min_width=int(img_size * 1.0),
            #               border_mode=cv2.BORDER_CONSTANT),
            # # A.ToGray(p=1),
            # A.ShiftScaleRotate(p=1.0, shift_limit=(-0.05, 0.05), scale_limit=(-0.4, 0.0), rotate_limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms_.ToTensorV2()
        ]),
        "valid": A.Compose([
            # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지.
            A.LongestMaxSize(max_size=int(img_size * 1.0)),
            # min_size보다 작으면 pad
            A.PadIfNeeded(min_height=int(img_size * 1.0), min_width=int(img_size * 1.0),
                          border_mode=cv2.BORDER_CONSTANT),
            # A.ToGray(p=1),
            # A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms_.ToTensorV2()
        ]),
        "save": A.Compose([
            # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지.
            A.LongestMaxSize(max_size=int(img_size * 1.0)),
            # min_size보다 작으면 pad
            A.PadIfNeeded(min_height=int(img_size * 1.0), min_width=int(img_size * 1.0),
                          border_mode=cv2.BORDER_CONSTANT),
            # A.ToGray(p=1),
            # A.Resize(img_size, img_size),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #transforms_.ToTensorV2()
        ])
    }
    return data_transforms

def make_df(target_data):
    cnt = 0
    # 현재 위치(.)의 파일을 모두 가져온다.
    path = f"./datasets/{target_data}/images/"
    label_path = f"./datasets/{target_data}/labels/"

    set_cnt = {}
    train_df, valid_df = None, None
    for filename in os.listdir(path):
        # print(path + filename +"/")
        if '옥시' in filename:
            label,x_l,y_l,w_l,h_l = [],[],[],[],[]
            img_path = []
            txt_path = []
            for img_name in os.listdir(path + filename + "/"):
                if img_name.endswith(("jpg","png")):
                    tmp = img_name[:-4]+'.txt'
                    try:
                        f = open(f'{label_path}{filename}/{tmp}', 'r')
                        while True:
                            line, x,y,w,h = list(map(float, f.readline().split()))
                            if line in [45, 47, 49]: # Oxy 레이블
                                label.append(int([45, 47, 49].index(line)))
                                x_l.append(x)
                                y_l.append(y)
                                w_l.append(w)
                                h_l.append(h)
                                img_path.append(f'{path}{filename}/{img_name}')
                                txt_path.append(f'{label_path}{filename}/{tmp}')
                                # src = cv2.imread(f'{path}{filename}/{img_name}')
                                # cv2.imwrite(f'/home/jml/_workspace/yolov5/data/images/{line}_{img_name}', src)

                        f.close()
                    except:
                        continue
            if len(label) > 0:
                df = pd.DataFrame({'label': label, 'img_path': img_path, 'label_path': txt_path,'x':x_l,'y':y_l, 'w':w_l, 'h':h_l})

                set_cnt[filename]=len(df)
                train, valid = train_test_split(df, test_size=0.2, shuffle=True)
                if cnt == 0:
                    train_df = train
                    valid_df = valid
                    cnt +=1
                else:
                    train_df = pd.concat([train_df, train], axis=0)
                    valid_df = pd.concat([valid_df, valid], axis=0)

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    # print(train_df)
    # print(valid_df)
    print(set_cnt)
    return train_df, valid_df
    # cnt_df = pd.DataFrame.from_dict(set_cnt, orient='index')

    # plt.bar(cnt_df.columns, cnt_df.values)  #
    # plt.xticks(rotation=90)
    # plt.show()

# make_df('drug_V2')
class BBoxCrop(object):
    """ Operator that crops according to the given bounding box coordinates. """
    def __call__(self, image, x, y, w, h):
        src_h, src_w = image.shape[:2]
        left = int((x - (w/2))*src_w)
        top = int((y - (h/2))*src_h)

        new_w = int(w*src_w)
        new_h = int(h*src_h)
        image = image[top: top + new_h, left: left + new_w]

        return image


class OxyDataset(torch.utils.data.Dataset):
    """ Dataset containing emotion categories (Daily, Gender, Embellishment). """
    def __init__(self, df, imgsz, type):
        self.df = df
        self.bbox_crop = BBoxCrop()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.to_pil = transforms.ToPILImage()
        self.data_transforms = data_transforms_img(imgsz)
        self.type = type
        self.le = OneHotEncoder(sparse=False)
        self.le.fit(df[['label']])
        self.oh = self.le.transform(df[['label']])
        self.th_w = 20
        self.th_h = 20
        # print(self.oh.shape)

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = io.imread(sample['img_path'])
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)

        bbox_xmin = sample['x']
        bbox_ymin = sample['y']
        bbox_xmax = sample['w']
        bbox_ymax = sample['h']
        label = sample['label']

        image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)

        # results = get_black_label(image, i, 16, 20)
        result = get_black_label(image, self.th_w, self.th_h)
        if result is None: result = image
        result = self.data_transforms[self.type](image=result)["image"]

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = background(image, None)
        # img = image.detach().cpu().numpy()
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        ret = {}
        ret['image'] = result
        ret['label'] = label
        ret['oh'] = self.oh[i]
        return ret

    def __len__(self):
        return len(self.df)
