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
import copy


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

        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.test_step_outputs = []

    def extract_feat(self, img, img_metas=None):
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

        gt_bboxes = batch['gt_bboxes'][0]
        gt_labels = batch['gt_labels'][0]
        gt_lanes_3d = batch['gt_lanes_3d'][0]
        gt_lane_labels_3d = batch['gt_lane_labels_3d'][0]
        gt_lane_adj = batch['gt_lane_adj'][0]
        gt_lane_lcte_adj = batch['gt_lane_lcte_adj'][0]

        len_queue = img.size(1)
        img_metas = batch['img_metas'][0]
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
        te_losses = {}

        bbox_losses, te_assign_result = self.bbox_head.loss(bbox_outs, gt_bboxes, gt_labels, bbox_img_metas,
                                                            gt_bboxes_ignore=None)
        te_feats = bbox_outs['history_states']
        te_cls_scores = bbox_outs['all_cls_scores']

        for loss in bbox_losses:
            te_losses['bbox_head.' + loss] = bbox_losses[loss]

        num_gt_bboxes = sum([len(gt) for gt in gt_labels])
        if num_gt_bboxes == 0:
            for loss in te_losses:
                te_losses[loss] *= 0

        losses = dict()
        prev_bev = None
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        outs = self.pts_bbox_head(img_feats, bev_feats, img_metas, te_feats, te_cls_scores)
        loss_inputs = [outs, gt_lanes_3d, gt_lane_labels_3d, gt_lane_adj, gt_lane_lcte_adj, te_assign_result]
        lane_losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]
        losses.update(te_losses)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.forward(batch)
        total_loss = sum(losses.values())
        for k, v in losses.items():
            self.log(f'train/{k}', v.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/total_loss', total_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        losses = self.forward(batch)
        total_loss = sum(losses.values())
        for k, v in losses.items():
            self.log(f'val/{k}', v.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/total_loss', total_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def on_test_start(self):
        """테스트 시작 시 상태를 초기화합니다."""
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def test_step(self, batch, batch_idx):
        img_metas = batch['img_metas'][0]
        img = batch['img'][0]
        """model의 테스트 과정 정의, simple_test 함수를 호출 하여 실제 추론을 수행."""
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information

        self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'],rescale=True)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        self.test_step_outputs.append(results_list)
        return results_list


    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)

        front_view_img_feats = [lvl[:, 0] for lvl in x]
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
        bbox_results = self.bbox_head.get_bboxes(bbox_outs, bbox_img_metas, rescale=rescale)
        te_feats = bbox_outs['history_states']
        te_cls_scores = bbox_outs['all_cls_scores']
        bev_feats = self.bev_constructor(x, img_metas, prev_bev)

        outs = self.pts_bbox_head(x, bev_feats, img_metas, te_feats, te_cls_scores)
        lane_results, lclc_results, lcte_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale)

        return bev_feats, bbox_results, lane_results, lclc_results, lcte_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_results, lane_results, lclc_results, lcte_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale)
        for result_dict, bbox, lane, lclc, lcte in zip(results_list, bbox_results, lane_results, lclc_results, lcte_results):
            result_dict['bbox_results'] = bbox
            result_dict['lane_results'] = lane
            result_dict['lclc_results'] = lclc
            result_dict['lcte_results'] = lcte

        return new_prev_bev, results_list


    def on_test_epoch_end(self):
        outputs = [item for sublist in self.test_step_outputs for item in sublist]
        dataset = self.trainer.datamodule.test_dataloader().dataset
        eval_results = dataset.evaluate(outputs, logger=self.logger, interval=24,
                                        pipeline=self.hparams.test_pipeline, show=True, out_dir='/home/ircvlab-504/TopoLogic_Lightning/vis')

        print(eval_results)


        self.log_dict(eval_results)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer_cfg = self.hparams.optimizer.copy()
        optimizer_type = optimizer_cfg.pop('type')

        paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        params = self.parameters()
        if paramwise_cfg:
            base_lr = optimizer_cfg['lr']
            params = []
            for name, param in self.named_parameters():
                param_group = {'params': [param]}
                if not param.requires_grad:
                    params.append(param_group)
                    continue

                for key, cfg in paramwise_cfg['custom_keys'].items():
                    if key in name:
                        param_group['lr'] = base_lr * cfg['lr_mult']
                        break
                params.append(param_group)

        optimizer = getattr(torch.optim, optimizer_type)(params, **optimizer_cfg)

        lr_scheduler_cfg = self.hparams.lr_config.copy()
        policy = lr_scheduler_cfg.pop('policy')

        if 'min_lr_ratio' in lr_scheduler_cfg:
            lr_scheduler_cfg['eta_min'] = optimizer_cfg['lr'] * lr_scheduler_cfg.pop('min_lr_ratio')

        lr_scheduler_cfg.pop('warmup', None)
        lr_scheduler_cfg.pop('warmup_iters', None)
        lr_scheduler_cfg.pop('warmup_ratio', None)

        lr_scheduler_cfg['T_max'] = self.hparams.total_epochs

        lr_scheduler = getattr(torch.optim.lr_scheduler, policy)(optimizer, **lr_scheduler_cfg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

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

