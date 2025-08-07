import pytorch_lightning as pl
from mmcv import Config
from mmdet.models import build_detector

from projects.topologic.models.modules.bevformer_constructer import BEVFormerConstructer
from projects.bevformer import BEVFormerEncoder
from projects.topologic.utils.builder import build_bev_constructor
from mmdet.models.builder import build_head
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
import torch
import random

class TopoLogicPL(pl.LightningModule):
    def __init__(self, cfg_path):
        super().__init__()
        cfg = Config.fromfile(cfg_path)
        # MMCV Config 객체를 일반 딕셔너리로 변환하여 저장합니다.
        cfg_dict = dict(cfg)
        self.save_hyperparameters(cfg_dict)

        model_cfg = self.hparams.model
        # 'type' 키는 mmcv 설정에서 클래스를 지정하는 데 사용되지만,
        # 클래스 생성자에는 전달되지 않아야 하므로 제거합니다.
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

        bev_constructor_cfg = model_cfg.bev_constructor.copy()
        bev_constructor_cfg.pop('type')

        self.bev_constructor = BEVFormerConstructer(**bev_constructor_cfg)


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
        img = batch['img']
        img_metas = batch.get('img_metas')[0]
        len_queue = 1
        img_metas = [each[len_queue - 1] for each in img_metas]

        img = img[0][:, -1, ...]


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

        prev_bev = None
        bev_embed = self.bev_constructor(img_feats, img_metas, prev_bev)


        return {"bev_embed" : bev_embed}


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

    print("\nOutput BEV embedding shape:")
    print(output['bev_embed'].shape)
