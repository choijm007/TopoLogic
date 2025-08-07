import pytorch_lightning as pl
from mmcv import Config
from mmdet.models import build_detector
from projects.topologic.utils.builder import build_bev_constructor
from mmdet.models.builder import build_head
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN


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
        img_feats = self.extract_feat(img)

        return {"img_feats" : img_feats}