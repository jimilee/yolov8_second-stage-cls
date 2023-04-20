## Yolov8 with Pytorch Image Models (timm) second stage Models

### Sub classification setting
modify ultralytics/yolo/cfg/default.yaml file.
```
'''

# Sub classification setting
sub: True

target_labels: [0, 1] # target label index for sub_classification (i.e. category )
sub_names: [ 'tf_efficientnet_b0', 'tf_efficientnet_b0'] # timm model name

sub_train:  ['dataset/0',#  dataset path for classifier(cropped images)'
            'dataset/1']                                        # dataset/0/ (category)
                                                                   #  ㄴtrain/
                                                                   #    ㄴ0/...11/ ...*.jpg (class folder)
                                                                   #  ㄴvalid/
                                                                   #    ㄴ0/...11/ ... (class folder)
sub_model: ['runs/sub_models/model1.pt',
            'runs/sub_models/model2.pt'] # path to timm model file,

sub_data : ['',''] # path to timm model class names

'''
```

### Train sub model
```commandline
python train_submodel.py
```

train_submodel.py --args
```
    parser.add_argument('--data', type=str,
                        default='ultralytics/yolo/cfg/default.yaml',
                        help='*.yaml path') #
    parser.add_argument('--test_path', type=str,
                        default='datasets',
                        help='test dataset path for detector(!None cropped images! just original detection dataset path)')
    parser.add_argument('--det_w', type=str,
                        default="best.pt",
                        help='trained detector weight path.')
    parser.add_argument('--epoch', type=int, default=3, help='train epochs')
    parser.add_argument('--name', type=str, default='tf_efficientnet_b0', help='timm model name')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=224, help='inference size h,w')
    parser.add_argument('--lr', type=float, default=0.0001, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
```

## [Ex] Train with Stanford Dogs Dataset
Download dataset from homepage \
[Image](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) \
[Annotations](http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar)

unzip into  'datasets/stanford_dogs/'
```
/datasets/stanford_dogs/
    ㄴImages/
      ㄴn02085620-Chihuahua/ ...
    ㄴAnnotation
      ㄴn02085620-Chihuahua/ ...
```
prepare dataset for train
```commandline
python datautils/prepare_StanfordDog.py 
```
modify .yaml file \
parse args & train stanford Dogs
```commandline
python train_submodel.py --data ultralytics/yolo/cfg/stanford_dogs.yaml --test_path datasets/stanford_dogs/Images --det_w yolov8m.pt
```
for detection, you can use pre-trained detection yolov8 model  

this repository based on https://github.com/ultralytics/ultralytics