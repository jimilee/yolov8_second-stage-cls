import argparse
import os
import random
import time

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
import timm
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from ultralytics.yolo.data.sub_datautil import OxyDataset, make_df

target_data = 'drug_V2'


Train_config = {'batch_size' : 16,
               'lr' : 0.0001,
               'epochs' : 100,
               'Model_name' : 'last_v3',
               'image_size' : 224}

trans = {'Train': A.Compose([
    A.Resize(Train_config['image_size'], Train_config['image_size']),
    A.OneOf([A.Rotate(limit=10),
             A.RandomBrightness(),
             A.CoarseDropout(always_apply=False, p=0.5, max_holes=20,
                             max_height=15, max_width=15, min_holes=1,
                             min_height=8, min_width=8),
             A.Cutout(num_holes=8, max_h_size=1, max_w_size=1, fill_value=1),
             ], p=1.0),
    A.GaussNoise(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()]),

    'Valid': A.Compose([A.Resize(Train_config['image_size'], Train_config['image_size']),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()])
}


class EMBClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(EMBClassifier, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes= num_classes)

    def forward(self,x):
        out = self.model(x)
        return out

class OxyClassifier(nn.Module):
    def __init__(self, model_name, pretrained =True):
        super(OxyClassifier, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        n_features = self.model.get_classifier().in_features
        self.model.reset_classifier(num_classes=0, global_pool="avg")

        self.linear = nn.Linear(n_features, 3)

    def forward(self,x):
        x = self.model(x)
        out = self.linear(x)
        return out

parser = argparse.ArgumentParser()
# tf_efficientnetv2_b1.in1k (8.14-240) # swinv2_cr_tiny_ns_224(	28.33p-224 ) / tf_efficientnet_b4_ns / swin_tiny_patch4_window7_224 /
# tinynet_b.in1k(3.73p-188)
parser.add_argument("--version", type=str, default='tinynet_b')
parser.add_argument("--name", type=str, default='tinynet_b')
parser.add_argument('--imgsz', default=	188, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

a = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_test_metrics(y_true, y_pred, verbose=True):
    """
    :return: asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi
    """
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    # if verbose:
    #     print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)

    return top_1, cs_accuracy.mean()

@torch.inference_mode()
def valid_infer(net, val_dataloader):
    with torch.no_grad():
        net.eval()
        gt_list = np.array([])
        pred_list = np.array([])

        bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for j, sample in bar:
            for key in sample.keys():
                sample[key] = sample[key].to(DEVICE)
            out = net(sample['image'])

            gt = np.array(sample['label'].cpu())
            gt_list = np.concatenate([gt_list, gt], axis=0)

            _, indx = out.max(1)
            pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)

        top_1, acsa = get_test_metrics(gt_list, pred_list)
        print("------------------------------------------------------")
        print(
            "Score: Top-1=%.5f, ACSA=%.5f" % (top_1, acsa))
        print("------------------------------------------------------")
    return top_1, acsa


@torch.inference_mode()
def valid_infer2(net, val_dataloader):
    with torch.no_grad():
        net.eval()
        gt_list = np.array([])
        pred_list = np.array([])

        bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for j, batches in bar:
            for b in range(len(batches)):
                sample = batches[int(b)]
                for key in sample:
                    if key == 'label': sample[key] = torch.tensor([sample[key]],dtype=torch.int16).to(DEVICE)
                    sample[key] = torch.Tensor(sample[key]).to(DEVICE)
                gt = np.array([sample['label'].cpu()])
                out = None
                for img in sample['image']:
                    img = img.view(1, 1, a.imgsz, a.imgsz)
                    img = img.repeat(1, 3, 1, 1)
                    if out is None : out = net(img)
                    else : out+= net(img)

                _, indx = out.max(1)
                pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)
                gt_list = np.concatenate([gt_list, gt], axis=0)

        top_1, acsa = get_test_metrics(gt_list, pred_list)
        print("------------------------------------------------------")
        print(
            "Score: Top-1=%.5f, ACSA=%.5f" % (top_1, acsa))
        print("------------------------------------------------------")
    return top_1, acsa
# def main():
#     """ The main function for model training. """
#     if os.path.exists('sub_models') is False:
#         os.makedirs('sub_models')
#
#     save_path = 'sub_models/' + a.version
#     if os.path.exists(save_path) is False:
#         os.makedirs(save_path)
#
#     net = OxyClassifier(model_name=a.name).to(DEVICE)
#     # print(net.daily_linear)
#     # trained_weights = torch.load(f'./sub_models/{a.name}/model_100_0.99.pkl', map_location=DEVICE)
#     # net.load_state_dict(trained_weights)
#     train_df, valid_df = make_df(target_data)# 데이터셋 경로
#     print(f'len of train df : {len(train_df)} , valid : {len(valid_df)}')
#     train_dataset = OxyDataset(train_df, imgsz=a.imgsz, type='train')
#     print(len(train_dataset[0]), len(train_dataset[1]))
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=8)
#
#     valid_dataset = OxyDataset(valid_df, imgsz=a.imgsz, type='valid')
#     valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=a.batch_size, shuffle=True, num_workers=8)
#
#     optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
#     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
#
#     criterion = nn.CrossEntropyLoss().to(DEVICE)
#
#     total_step = len(train_dataloader)
#     step = 0
#     t0 = time.time()
#
#     best_top1 = 0
#     for epoch in range(a.epochs):
#         net.train()
#         # bar = next(iter(train_dataloader))
#         # i = 0
#         bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
#         for i, sample in bar:
#             optimizer.zero_grad()
#             step += 1
#             for key in sample:
#                 # if key == 'label': sample[key] = torch.IntTensor([sample[key]]).to(DEVICE)
#                 sample[key] = torch.Tensor(sample[key]).to(DEVICE)
#             # print(sample['label'], sample['oh'])
#             # out = None
#             out = net(sample['image'])
#             # else : out += net(img)
#             # print(out)
#             # print("sample['label'] : ", sample['label'])
#             # print("sample['oh'] : ", sample['oh'])
#             # print(sample['label'].shape)
#             # print(out[0].shape)
#             loss = criterion(out, sample['label'])
#             # loss = tmp_loss
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#
#             bar.set_postfix_str('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
#                   'Time : {:2.3f}'
#                   .format(epoch + 1, a.epochs, i + 1, total_step, loss.item(), time.time() - t0))
#
#             t0 = time.time()
#             i += 1
#         top_1, acsa = valid_infer(net, valid_dataloader)
#
#         if best_top1 < top_1 :
#             print('Saving Best Model....')
#             torch.save(net.state_dict(), f'{save_path}/best.pkl')
#             best_top1 = top_1
#             print('OK.')
#         if ((epoch + 1) % 10 == 0):
#             print('Saving Model....')
#             torch.save(net.state_dict(), f'{save_path}/model_{str(epoch + 1)}_{top_1:.2f}.pkl')
#             print('OK.')
#

    # ### Test train set.
    # test.run(version=a.version, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=a.seed)
    # run.finish()

# 학습 할 때.
# def main():
#     """ The main function for model training. """
#     if os.path.exists('sub_models') is False:
#         os.makedirs('sub_models')
#
#     save_path = 'sub_models/' + a.version
#     if os.path.exists(save_path) is False:
#         os.makedirs(save_path)
#
#     net = OxyClassifier(model_name=a.name).to(DEVICE)
#     # print(net.daily_linear)
#
#     train_df, valid_df = make_df(target_data)# 데이터셋 경로
#     print(f'len of train df : {len(train_df)} , valid : {len(valid_df)}')
#     train_dataset = OxyDataset(train_df, imgsz=a.imgsz, type='train')
#     print(len(train_dataset[0]), len(train_dataset[1]))
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=8,collate_fn=lambda x: x is not None)
#
#     valid_dataset = OxyDataset(valid_df, imgsz=a.imgsz, type='valid')
#     valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=a.batch_size, shuffle=True, num_workers=8,collate_fn=lambda x: x is not None)
#
#     optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
#     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
#
#     criterion = nn.CrossEntropyLoss().to(DEVICE)
#
#     total_step = len(train_dataloader)
#     step = 0
#     t0 = time.time()
#
#     best_top1 = 0
#     for epoch in range(a.epochs):
#         net.train()
#         # bar = next(iter(train_dataloader))
#         # i = 0
#         bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
#         for i, batches in bar:
#             for b in range(len(batches)):
#                 sample = batches[int(b)]
#                 optimizer.zero_grad()
#                 step += 1
#                 for key in sample:
#                     if key == 'label': sample[key] = torch.IntTensor([sample[key]]).to(DEVICE)
#                     sample[key] = torch.Tensor(sample[key]).to(DEVICE)
#                 # print(sample.keys())
#                 # out = None
#                 for img in sample['image']:
#                     img = img.view(1, 1, a.imgsz, a.imgsz)
#                     img = img.repeat(1, 3, 1, 1)
#                     # print(img.shape)
#                     out = net(img)
#                     # else : out += net(img)
#                     # print(out)
#                     # print("sample['label'] : ", sample['label'])
#                     # print("sample['oh'] : ", sample['oh'])
#                     # print(sample['label'].shape)
#                     # print(out[0].shape)
#                     loss = criterion(out[0], sample['oh'])
#                     # loss = tmp_loss
#                     loss.backward()
#                     optimizer.step()
#                     scheduler.step()
#
#             bar.set_postfix_str('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
#                   'Time : {:2.3f}'
#                   .format(epoch + 1, a.epochs, i + 1, total_step, loss.item(), time.time() - t0))
#
#             t0 = time.time()
#             i += 1
#         top_1, acsa = valid_infer2(net, valid_dataloader)
#
#         if best_top1 < top_1 :
#             print('Saving Best Model....')
#             torch.save(net.state_dict(), f'{save_path}/best.pkl')
#             best_top1 = top_1
#             print('OK.')
#         if ((epoch + 1) % 10 == 0):
#             print('Saving Model....')
#             torch.save(net.state_dict(), f'{save_path}/model_{str(epoch + 1)}_{top_1:.2f}.pkl')
#             print('OK.')
#
#
#     # ### Test train set.
#     # test.run(version=a.version, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=a.seed)
#     # run.finish()
#

# if __name__ == '__main__':
#     main()
