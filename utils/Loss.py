from pickletools import read_uint1
import torch
import torch.nn as nn
from models.yolo import Detect
from utils.metrics import box_iou, bbox_iou


class Loss:
    def __init__(self, anchors, hyp=None, device="cuda"):
        # iou loss
        # obj loss
        # cls loss
        self.hyp=hyp
        self.device = device
        self.nlayres = 3
        self.na = 3
        self.anchors = anchors  # shape :  [3 ,2]
        self.nc = 3
        self.BCE_cls = nn.BCEWithLogitsLoss()
        self.BCE_obj = nn.BCEWithLogitsLoss()

    def set_tg(self, gti, bn_id, anc_id, gridij, xywh, cls):

        gti[bn_id, anc_id, gridij[1], gridij[0], :4] = xywh  # [-0.5,1.5]
        gti[bn_id, anc_id, gridij[1], gridij[0], 4] = 1  # 标记此处为有物体
        gti[
            bn_id, anc_id, gridij[1], gridij[0], (cls.long() + 5).item()
        ] = 1  # 类别one-hot

    def generate_gtojb_anchors_indices(self, gt_i, anchors_i):
        # a layer have 3 anchor size
        obj_ids = gt_i[:, :, :, :, 4] > 0  # 获取有真值物体的索引
        bn, ln, h, w, _ = gt_i.shape
        anshape = anchors_i.shape
        ancs = anchors_i.repeat(bn, h, w, 1, 1)
        ancs = ancs.permute(0, 3, 1, 2, 4)
        return obj_ids, ancs

    def _build_target(self, pre, targets):
        # pre_0 ： scale  1/8  [batch ,  anchors_of_layer_0=3 , 80 , 80 , nc+5]
        # pre_1 ： scale  1/16 [batch ,  anchors_of_layer_1=3 , 40 , 40 , nc+5]
        # pre_2 ： scale  1/30 [batch ,  anchors_of_layer_2=3 , 20 , 20 , nc+5]

        # targets: 要求 shape (batch * 3,6) ; 3 : a image have 3 layers , 6: [anchor_i , class , x1 ,y1 , x2 , y2 ]

        batch_size = pre[0].shape[0]
        nt = targets.shape[0]  # number of anchors , targets

        gt_list = []
        for li in range(self.nlayres):  # 对每一个输出层 简历一个gt
            # loop each out layers
            pi = pre[li]
            pi_size = torch.tensor(pi.shape[2:4], device=self.device)  # ny,nx
            gti = torch.zeros_like(pi, device=self.device)
            if nt:
                # 这个batch 里面有真值
                for ti in targets:
                    # 对于每个gt 找到它属于哪个 batch_index , 它的类别 和 box
                    # ti.shape =(1,6)
                    batch_index = ti[0].long()
                    cls = ti[1].long()
                    box = ti[2:]
                    # ti = torch.zeros_like(pi, device=pi.device)
                    for a_id, anchor in enumerate(self.anchors[li]):
                        # 对当前layers 的每一个anchor 创建对应的 gt
                        # r= box[2:]/anchor
                        # j=torch.max(r, 1 / r).max(2)[0] <4

                        xy = box[:2] * pi_size  #  0~size
                        wh = box[2:] * pi_size

                        gridij = xy.long()
                        self.set_tg(gti,batch_index,a_id,gridij,torch.cat((xy - gridij, wh), dim=0),cls,)

                        # 开始正采样 区相与gt的相邻网格为gt
                        judge = (xy - gridij) - 0.5

                        grid_add = gridij.clone()

                        if judge[0] < 0:

                            grid_add[0] = (gridij[0] - 1).clamp(0)
                            self.set_tg(gti,batch_index,a_id,grid_add,torch.cat((xy - grid_add, wh), dim=0),cls,)

                            grid_add[0] = gridij[0]
                            if judge[1] < 0:
                                grid_add[1] = (gridij[1] - 1).clamp(0)
                            elif judge[1] > 0:
                                grid_add[1] = (gridij[1] + 1).clamp(0, pi_size[0])
                            self.set_tg(gti,batch_index,a_id,grid_add,torch.cat((xy - grid_add, wh), dim=0),cls,)

                        elif judge[0] > 0:
                            grid_add[0] = (gridij[0] + 1).clamp(0, pi_size[0])
                            self.set_tg(gti,batch_index,a_id,grid_add,torch.cat((xy - grid_add, wh), dim=0),cls,)

                            grid_add[0] = gridij[0]
                            if judge[1] < 0:
                                grid_add[1] = (gridij[1] - 1).clamp(0)
                            elif judge[1] > 0:
                                grid_add[1] = (gridij[1] + 1).clamp(0, pi_size[0])
                            self.set_tg(gti,batch_index,a_id,grid_add,torch.cat((xy - grid_add, wh), dim=0),cls,)
            else:
                gti = torch.zeros_like(pi)

            gt_list.append(gti.to(self.device))

        return gt_list

    def __call__(self, pre, targets):
        device = targets.device
        lcls, lbox, lobj = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        gts = self._build_target(pre, targets)

        for i, pre_i in enumerate(pre):  # loop layers
            gt = gts[i]
            batch_size = pre_i.shape[0]
            tobj = torch.zeros_like(pre_i[..., 0], device=device)  # target obj
            obj_ids, anchs = self.generate_gtojb_anchors_indices(gt, self.anchors[i])

            pxy = pre_i[obj_ids][:, :2].sigmoid() * 2 - 0.5
            pwh = (pre_i[obj_ids][:, 2:4].sigmoid() * 2) ** 2 * anchs[obj_ids]
            pbox = torch.cat((pxy, pwh), 1)
            pbox = pbox.permute(1, 0)
            gbox = gt[obj_ids][:, :4].permute(1, 0)
            ciou = bbox_iou(pbox, gbox.T, x1y1x2y2=False, CIoU=True)
            # ciou2=bbox_iou(pbox,gbox , x1y1x2y2=False, CIoU=True)
            lbox = lbox + (1 - ciou).mean()

            lcls = lcls + self.BCE_cls(pre_i[obj_ids][:, 5:], gt[obj_ids][:, 5:])

            score_iou = ciou.detach().clamp(0).type(tobj.dtype)
            # 获取 obj_ids  gridj gridi

            tobj[obj_ids] = score_iou

            lobj = lobj + self.BCE_obj(pre_i[..., 4], tobj)

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        # lbox *= 0.05
        # lobj *= 1
        # lcls *= 0.01875
        return (lbox + lobj + lcls) * 4, torch.cat((lbox, lobj, lcls)).detach()
