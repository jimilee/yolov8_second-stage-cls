# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ yolo task=... mode=predict  model=s.pt --source 0                         # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ yolo task=... mode=predict --weights yolov8n.pt          # PyTorch
                                    yolov8n.torchscript        # TorchScript
                                    yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                    yolov8n_openvino_model     # OpenVINO
                                    yolov8n.engine             # TensorRT
                                    yolov8n.mlmodel            # CoreML (macOS-only)
                                    yolov8n_saved_model        # TensorFlow SavedModel
                                    yolov8n.pb                 # TensorFlow GraphDef
                                    yolov8n.tflite             # TensorFlow Lite
                                    yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov8n_paddle_model       # PaddlePaddle
    """
import json
import pickle
import os.path
import platform
import random
from collections import defaultdict
from datetime import time
from pathlib import Path

import cv2
import torch
import timm

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadPilAndNumpy, LoadScreenshots, LoadStreams
from ultralytics.yolo.data.sub_datautil import get_black_label, data_transforms_img
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.plotting import save_one_box
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.engine.results import SubResults
class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f"{self.args.mode}"
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False

        # Usable if setup is done
        self.model = None
        self.sub_model = {}
        self.labels = {}
        # self.target_labels = self.args.target_labels # target labels
        self.data = self.args.data  # data_dict
        self.bs = None
        self.imgsz = None
        self.device = None
        self.classes = self.args.classes
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        self.data_transforms = data_transforms_img(528) # sub_image size


    def preprocess(self, img):
        pass

    def get_annotator(self, img):
        raise NotImplementedError("get_annotator function needs to be implemented")

    def write_results(self, results, batch, print_string):
        raise NotImplementedError("print_results function needs to be implemented")

    def second_stage_process(self, idx, results, batch, classes=None):
        # sub models.
        p, im, im0 = batch
        det = results[idx].boxes
        sub_str = "None"
        _cat, _lidx, _bbox, isTrue = None, None, None, True
        # write
        # if len(list(reversed(det))) > 1:
        #     for d in list(det):
        #         cls, conf = d.cls.squeeze(), d.conf.squeeze()
        #         print(cls, conf)
        for d in list(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            c = int(cls)  # integer class
            # sub_str = c
            # Second-stage classifier
            if c in self.args.target_labels:
                imc = im0.copy()
                _bbox = d.xyxy.squeeze()
                imsub = save_one_box(_bbox, imc, BGR=True, save=False)  # make crop bbox img
                _bbox = _bbox.tolist()
                _cat = c
                # # Save cropped image
                # im1 = self.data_transforms[1]['save'](image=imsub)["image"]
                # test_cl = str(p).split('/')[-2:]
                # if not os.path.isdir(f"runs/test_for_class/{c}/{test_cl[0]}"):
                #     os.mkdir(f"runs/test_for_class/{c}/{test_cl[0]}")
                # test_cl[-1] = 'e_'+test_cl[-1]
                # test_cl = '/'.join(test_cl)
                # cv2.imwrite(f"runs/test_for_class/{c}/{test_cl}", im1)
                # check dir name
                _d_n = str(p).split('/')[-2]
                try:#
                    if _d_n.isdigit(): # if digit
                        _correct = int(_d_n)
                    else:
                        _correct = _d_n # if string
                        # _correct = str(p).split('/')[-2][10:]
                except:
                    _correct = -1

                im1 = self.data_transforms['valid'](image=imsub)["image"]
                im1 = torch.unsqueeze(im1, 0).to(self.device)
                with torch.no_grad():
                    self.sub_model[c].eval()
                    out = self.sub_model[c](im1)
                    # print(out[0])
                    # print(c, len(out[0]), out.argmax(dim=1).item(), max(out[0]), out[0][out.argmax(dim=1).item()])
                    _idx = out.argmax(dim=1).item()
                    _lidx = self.labels[c][_idx].split('_')
                    #
                    # print(c, _idx, str(_lidx[1]), _correct, str(_lidx[1]) == _correct)
                    if _correct != -1 : # if folder name is digit. --> check result.
                        if str(_correct).isdigit():
                            sub_str = f'[ {str(_lidx[0])} ]_{str(_lidx[1])} / {str(int(_lidx[1]) == _correct)}'
                            isTrue = int(_lidx[1]) == _correct
                        else:
                            sub_str = f'{str(_lidx[1])} / {str((_lidx[1]) == _correct)}'
                            isTrue = str(_lidx[1]) == _correct
                        # print(isTrue)

                    else:
                        sub_str = f'[ {str(_lidx[0])} ]_{str(_idx)}'
            break

        return sub_str, SubResults(boxes=_bbox, category=_cat, classes=_lidx, result=isTrue), isTrue

    def setup_source(self, source=None):
        if not self.model:
            raise Exception("setup model before setting up source!")
        # source
        source, webcam, screenshot, from_img = self.check_source(source)
        # model
        stride, pt = self.model.stride, self.model.pt
        imgsz = check_imgsz(self.args.imgsz, stride=stride, min_dim=2)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            self.args.show = check_imshow(warn=True)
            self.dataset = LoadStreams(source,
                                       imgsz=imgsz,
                                       stride=stride,
                                       auto=pt,
                                       transforms=getattr(self.model.model, 'transforms', None),
                                       vid_stride=self.args.vid_stride)
            bs = len(self.dataset)
        elif screenshot:
            self.dataset = LoadScreenshots(source,
                                           imgsz=imgsz,
                                           stride=stride,
                                           auto=pt,
                                           transforms=getattr(self.model.model, 'transforms', None))
        elif from_img:
            self.dataset = LoadPilAndNumpy(source,
                                           imgsz=imgsz,
                                           stride=stride,
                                           auto=pt,
                                           transforms=getattr(self.model.model, 'transforms', None))
        else:
            self.dataset = LoadImages(source,
                                      imgsz=imgsz,
                                      stride=stride,
                                      auto=pt,
                                      transforms=getattr(self.model.model, 'transforms', None),
                                      vid_stride=self.args.vid_stride)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.webcam = webcam
        self.screenshot = screenshot
        self.from_img = from_img
        self.imgsz = imgsz
        self.bs = bs

    @smart_inference_mode()
    def __call__(self, source=None, model=None, stream=False):
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self):
        # Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode
        gen = self.stream_inference()
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def stream_inference(self, source=None, model=None):
        self.run_callbacks("on_predict_start")
        # setup model
        if not self.model:
            self.setup_model(model)
        if not self.sub_model and self.args.sub:
            print('warm up models')
            self.setup_submodel()

        # setup source. Run every time predict is called
        self.setup_source(source)
        # check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        for batch in self.dataset:
            self.run_callbacks("on_predict_batch_start")
            self.batch = batch
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)

            # postprocess
            with self.dt[2]:
                self.results = self.postprocess(preds, im, im0s, self.classes)

            self.sub_results = []
            for i in range(len(im)):
                p, im0 = (path[i], im0s[i]) if self.webcam or self.from_img else (path, im0s)
                p = Path(p)
                sub_str = ''
                if self.args.sub: # Sub Classification
                    sub_str, sub_res, isTrue = self.second_stage_process(i, self.results, (p, im, im0), self.classes)
                    self.sub_results.append(sub_res)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0), sub_str)

                if self.args.show:
                    self.show(p)

                if self.args.save:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))

            self.run_callbacks("on_predict_batch_end")
            yield from self.sub_results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f"{s}{'' if len(preds) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if (self.args.save_txt or self.args.save) and self.args.verbose:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")


    # def _stream_inference(self, source=None, model=None): # original code
    #     self.run_callbacks("on_predict_start")
    #     # setup model
    #     if not self.model:
    #         self.setup_model(model)
    #     # setup source. Run every time predict is called
    #     self.setup_source(source)
    #     # check if save_dir/ label file exists
    #     if self.args.save or self.args.save_txt:
    #         (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
    #     # warmup model
    #     if not self.done_warmup:
    #         self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.bs, 3, *self.imgsz))
    #         self.done_warmup = True
    #
    #     self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
    #     for batch in self.sub_dataset:
    #         self.run_callbacks("on_predict_batch_start")
    #         self.batch = batch
    #         path, im, im0s, vid_cap, s = batch
    #         visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False
    #         with self.dt[0]:
    #             im = self.preprocess(im)
    #             if len(im.shape) == 3:
    #                 im = im[None]  # expand for batch dim
    #
    #         # Inference
    #         with self.dt[1]:
    #             preds = self.model(im, augment=self.args.augment, visualize=visualize)
    #
    #         # postprocess
    #         with self.dt[2]:
    #             self.results = self.postprocess(preds, im, im0s, self.classes)
    #         for i in range(len(im)):
    #             p, im0 = (path[i], im0s[i]) if self.webcam or self.from_img else (path, im0s)
    #             p = Path(p)
    #
    #             if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
    #                 s += self.write_results(i, self.results, (p, im, im0))
    #
    #             if self.args.show:
    #                 self.show(p)
    #
    #             if self.args.save:
    #                 self.save_preds(vid_cap, i, str(self.save_dir / p.name))
    #
    #         self.run_callbacks("on_predict_batch_end")
    #         yield from self.results
    #
    #         # Print time (inference-only)
    #         if self.args.verbose:
    #             LOGGER.info(f"{s}{'' if len(preds) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")
    #
    #     # Print results
    #     if self.args.verbose and self.seen:
    #         t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
    #         LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms postprocess per image at shape '
    #                     f'{(1, 3, *self.imgsz)}' % t)
    #     if self.args.save_txt or self.args.save:
    #         nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
    #         s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
    #         LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
    #
    #     self.run_callbacks("on_predict_end")

    def setup_model(self, model):
        device = select_device(self.args.device)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)
        self.device = device
        self.model.eval()

    def setup_submodel(self):
        sub_weights_paths = self.args.sub_model # model path.
        sub_labels_paths = self.args.sub_data
        model_name = self.args.sub_names
        for i, lid in enumerate(self.args.target_labels):
            wp, jp = sub_weights_paths[i], sub_labels_paths[i]
            if not os.path.isfile(jp):
                save_dir = '/'.join(wp.split('/')[:-1])
                jp = save_dir + f'/label_data_{lid}.pkl' # default pkl path
            # self.labels[lid] = []
            with open(jp, 'rb') as f:
                label_ = pickle.load(f)
                self.labels[lid] = label_

            nc = len(label_)
            # print('setup :',model_name[i], nc)
            model = timm.create_model(model_name=model_name[i], num_classes=nc)

            if os.path.isfile(wp):
                checkpoint = torch.load(wp, map_location=self.device)
                model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()

            self.sub_model[lid]=model

            # trained_weights = torch.load(sub_weights, map_location=device)
            # self.sub_model.load_state_dict(trained_weights)
            # self.sub_model.eval()

    def check_source(self, source):
        source = source if source is not None else self.args.source
        webcam, screenshot, from_img = False, False, False
        if isinstance(source, (str, int, Path)):  # int for local usb carame
            source = str(source)
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            is_url = source.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://'))
            webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
            screenshot = source.lower().startswith('screen')
            if is_url and is_file:
                source = check_file(source)  # download
        else:
            from_img = True
        return source, webcam, screenshot, from_img

    def show(self, p):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        im0 = self.annotator.result()
        # save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)
