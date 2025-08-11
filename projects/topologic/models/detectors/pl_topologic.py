import pytorch_lightning as pl
from mmcv import Config
from mmdet.models import build_detector
from pytorch_lightning.utilities.types import STEP_OUTPUT

from projects.topologic.models.dense_heads import TopoLogicHead
from projects.topologic import CustomDeformableDETRHead
from projects.topologic.models.modules.bevformer_constructer import BEVFormerConstructer
from projects.bevformer import BEVFormerEncoder
from projects.topologic.utils.builder import build_bev_constructor
from mmdet.models.builder import build_head
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
import torch
import random


def build(cfg):
    return cfg.pop("type")

class TopoLogicPL(pl.LightningModule):
    def __init__(self, cfg_path):
        super().__init__()
        cfg = Config.fromfile(cfg_path)
        # MMCV Config 객체를 일반 딕셔너리로 변환하여 저장합니다.
        cfg_dict = dict(cfg)
        self.save_hyperparameters(cfg_dict)

        model_cfg = self.hparams.model

        # --- Backbone and Neck ---
        img_backbone_cfg = model_cfg.img_backbone.copy()
        img_backbone_cfg.pop('type', None)
        self.img_backbone = ResNet(**img_backbone_cfg)

        if model_cfg.get('img_neck'):
            self.with_img_neck = True
            img_neck_cfg = model_cfg.img_neck.copy()
            img_neck_cfg.pop('type', None)
            self.img_neck = FPN(**img_neck_cfg)
        else:
            self.with_img_neck = False

        # --- BEV Constructor ---
        bev_constructor_cfg = model_cfg.bev_constructor.copy()
        bev_constructor_cfg.pop('type')
        self.bev_constructor = BEVFormerConstructer(**bev_constructor_cfg)

        # --- Bbox Head ---
        bbox_head_cfg = model_cfg.bbox_head.copy()
        bbox_head_cfg.pop('type')
        bbox_head_cfg.update(train_cfg=model_cfg.train_cfg.bbox)
        self.bbox_head = CustomDeformableDETRHead(**bbox_head_cfg)

        # --- Lane Head ---
        lane_head_cfg = model_cfg.lane_head.copy()
        lane_head_cfg.pop('type')
        lane_head_cfg.update(train_cfg=model_cfg.train_cfg.lane)
        self.pts_bbox_head = TopoLogicHead(**lane_head_cfg)


    def extract_feat(self, img):
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                # N = Number of Camera
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B,-1,C,H,W))

        return img_feats_reshaped

    def forward(self, batch):
        img = batch['img'][0]
        img_metas = batch['img_metas'][0]

        len_queue = img.size(1)
        img_metas = [each[len_queue - 1] for each in img_metas]

        img = img[:, -1, ...]

        img_feats = self.extract_feat(img)

        front_view_img_feats = [lvl[:, 0] for lvl in img_feats]
        batch_input_shape = tuple(img[0, 0].size()[-2:])
        bbox_img_metas = []
        for img_meta in img_metas:
            bbox_img_metas.append(
                dict(
                    batch_input_shape=batch_input_shape,
                    img_shape=img_meta['img_shape'][0],
                    scale_factor=img_meta['scale_factor'][0],
                    crop_shape=img_meta['crop_shape'][0]))
            img_meta['batch_input_shape'] = batch_input_shape

        bbox_outs = self.bbox_head(front_view_img_feats, bbox_img_metas)


        te_feats = bbox_outs['history_states']
        te_cls_scores = bbox_outs['all_cls_scores']

        prev_bev = None
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        outs = self.pts_bbox_head(img_feats, bev_feats, img_metas, te_feats, te_cls_scores)


        return {**bbox_outs, **outs}

    def training_step(self, batch, batch_idx):

        img_metas = batch['img_metas'][0]
        gt_bboxes = batch['gt_bboxes'][0]
        gt_labels = batch['gt_labels'][0]
        gt_lanes_3d = batch['gt_lanes_3d'][0]
        gt_lane_labels_3d = batch['gt_lane_labels_3d'][0]
        gt_lane_adj = batch['gt_lane_adj'][0]
        gt_lane_lcte_adj = batch['gt_lane_lcte_adj'][0]


        pred_dict = self.forward(batch)
        te_losses = {}
        losses = dict()
        bbox_losses, te_assign_result = self.bbox_head.loss(preds_dict, gt_bboxes, gt_labels, bbox_img_metas,
                                                            gt_bboxes_ignore=None)
        loss_inputs = [pred_dict, gt_lanes_3d, gt_lane_labels_3d, gt_lane_adj, gt_lane_lcte_adj, te_assign_result]
        lane_losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        for loss in bbox_losses:
            te_losses['bbox_head.' + loss] = bbox_losses[loss]

        num_gt_bboxes = sum([len(gt) for gt in gt_labels])
        if num_gt_bboxes == 0:
            for loss in te_losses:
                te_losses[loss] *= 0

        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]
        losses.update(te_losses)

        return losses

if __name__ == '__main__':
    random.seed(0)
    cfg_path = '/home/ircvlab-504/TopoLogic_Lightning/projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py'
    model = TopoLogicPL(cfg_path)
    print("Model initialized successfully!")

    # check_data.py의 로직을 여기에 통합하여 테스트
    from projects.topologic.datasets.pl_dataset import TopoLogicDataModule

    data_module = TopoLogicDataModule(
        data_root=model.hparams.data_root,
        ann_file=model.hparams.data.train.ann_file,
        batch_size=1,
        num_workers=0,
        queue_length=1,
        filter_empty_te=False,
        split='train',
        filter_map_change=True,
        pipeline=model.hparams.train_pipeline,
        modality=model.hparams.input_modality,
        classes=model.hparams.class_names,
        test_mode=False
    )
    data_module.setup('fit')
    data_batch = next(iter(data_module.train_dataloader()))

    model.eval()
    with torch.no_grad():
        # 'img_metas'가 없을 경우를 대비하여 빈 리스트를 전달
        if 'img_metas' not in data_batch:
            data_batch['img_metas'] = [None]

        output = model(data_batch)

    print("\nOutput ")
    for k,v in output.items():
        print(k)
    #print(output.items())

