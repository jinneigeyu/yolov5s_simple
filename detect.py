import torch
import os
import cv2
import numpy as np
from utils.metrics import box_iou
from utils.augmentations import letterbox
from models.yolo import YOLOV5S, Detect
from utils.plots import Annotator, Colors
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
    generate_anchors
)
import yaml
from utils.torch_utils import select_device, time_sync

device = "cuda"
TEST_FOLDER = "data\\test\\"
IMG_SIZE = 640

colors = Colors()
# names = ["head", "back", "circle"]
# names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def compute_pre(
    model,
    input,
    src,
    conf_thres=0.25,
    iou_thres=0.15,
    names=None,
    classes=None,
    agnostic_nms=False,
    max_det=20,
):
    t2 = time_sync()
    pre, _ = model(input)
    print(time_sync() - t2)
    pred = non_max_suppression(
        pre, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
    )
       
    annotator = Annotator(src, line_width=2, example=names)
    
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(input.shape[2:], det[:, :4], src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label=names[c]
                annotator.box_label(xyxy, label, color=colors(c, True))
        im0 = annotator.result()
        cv2.imshow(str("detect"), im0)
        cv2.waitKey(0)  # 1 millisecond


@torch.no_grad()
def main():
    run_folder = 'run/2022-06-09-155158'
    with open( os.path.join( run_folder,'hyp.yaml'), errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    # device = "cuda"
    device = "cuda"
    anchors= generate_anchors(hyp['auto_anchors'],device)
    model = YOLOV5S(IMG_SIZE, hyp['num_class'], anchors=anchors)
    
    if device != 'cpu':
        model.half()
        model.cuda()
        
    model_pth = os.path.join( run_folder,'mode_120_.pt')
    model.load_state_dict(torch.load(model_pth))
    model.eval()

    stride, names = model.head[-1].stride, ["head", "back", "person"]
    
    dataset = LoadImages("./test_imgs", img_size=IMG_SIZE, stride=stride[2])
    for path, im, im0s, vid_cap, s in dataset:
        img = cv2.imread(path)
        # img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        src = img.copy()
        img = letterbox(img,(IMG_SIZE,IMG_SIZE),stride=32,auto=False)[0]
       
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im_t = torch.from_numpy(img).to(device)
        im_t = im_t.float()
        im_t /= 255
        im_t = im_t[None]
        
        if device !='cpu':
            im_t=im_t.half()
        
        compute_pre(model, im_t, src,names=hyp['classnames'])


if __name__ == "__main__":
    main()
