_base_ = []
custom_imports = dict(imports=['projects.bevformer', 'projects.topologic'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]
# 6개인 이유 Xmin Xmax, ... 총 6개
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['centerline']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_cams = 7
pts_dim = 3

dataset_type = 'OpenLaneV2_subset_A_Dataset'
data_root = './data/OpenLane-V2/'

para_method = 'fix_pts_interp'
method_para = dict(n_points=11)
code_size = pts_dim * method_para['n_points']

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
),

_num_levels_ = 4
bev_h_ = 100
bev_w_ = 200

model = dict(
    type='TopoLogic',
    img_backbone=dict(
        # 백본 네트워크의 종류로 ResNet을 사용합니다.
        type='ResNet',
        # ResNet의 깊이를 50으로 설정합니다. 즉, ResNet-50 모델을 사용합니다.
        depth=50,
        # ResNet 내부의 스테이지(stage) 개수를 4개로 설정합니다.
        num_stages=4,
        # 4개의 스테이지 중 1, 2, 3번(0부터 시작) 스테이지의 출력을 다음 레이어(FPN)로 전달하도록 설정합니다.
        out_indices=(1, 2, 3),
        # 첫 번째 스테이지의 가중치를 학습 중에 업데이트되지 않도록 고정(freeze)합니다.
        frozen_stages=1,
        # 정규화(Normalization) 레이어에 대한 설정입니다. 배치 정규화(Batch Normalization)를 사용하며,
        # 이 레이어의 파라미터들은 학습되지 않도록 설정합니다.
        norm_cfg=dict(type='BN', requires_grad=False),
        # 정규화 레이어를 항상 평가(evaluation) 모드로 동작하게 합니다.
        norm_eval=True,
        # ResNet의 구조 스타일을 PyTorch 방식으로 설정합니다.
        style='pytorch',
        # 모델의 가중치 초기화 설정입니다. torchvision에서 제공하는 사전 학습된 ResNet-50 모델의 가중치를 불러와서 초기화합니다.
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        # 이미지 넥(neck)의 타입으로 Feature Pyramid Network(FPN)를 사용합니다.
        # FPN은 백본 네트워크에서 추출된 다양한 크기의 특징 맵을 결합하여
        # 객체 검출 성능을 향상시키는 역할을 합니다.
        type='FPN',

        # FPN에 입력되는 특징 맵들의 채널 수를 리스트로 지정합니다.
        # 이 값들은 img_backbone(ResNet-50)의 out_indices=(1, 2, 3)에 해당하는
        # 각 스테이지 출력의 채널 수와 일치해야 합니다.
        in_channels=[512, 1024, 2048],

        # FPN을 통과한 후 출력되는 특징 맵들의 채널 수를 지정합니다.
        # 여기서는 _dim_ 변수(값: 256)를 사용하여 모든 출력 레벨의 채널 수를 통일합니다.
        out_channels=_dim_,

        # FPN이 피라미드를 구축하기 시작할 백본 출력의 인덱스를 지정합니다.
        # 0은 in_channels 리스트의 첫 번째(512 채널) 특징 맵부터 사용한다는 의미입니다.
        start_level=0,

        # 추가적인 컨볼루션 레이어를 어디에 추가할지 설정합니다.
        # 'on_output'은 FPN의 가장 마지막 레벨(가장 해상도가 낮은) 특징 맵을 입력으로 받아
        # 새로운 컨볼루션 레이어를 통과시켜 한 단계 더 높은 레벨의 특징 맵을 생성합니다.
        # 이 설정 때문에 입력은 3개지만 출력(num_outs)은 4개가 됩니다.
        add_extra_convs='on_output',

        # FPN에서 생성할 최종 출력 특징 맵의 개수를 지정합니다.
        # 여기서는 _num_levels_ 변수(값: 4)를 사용합니다.
        num_outs=_num_levels_,

        # add_extra_convs로 추가되는 컨볼루션 레이어 이전에
        # ReLU 활성화 함수를 적용할지 여부를 결정합니다.
        relu_before_extra_convs=True),


    bev_constructor=dict(
        type='BEVFormerConstructer',
        num_feature_levels=_num_levels_, # FCN에서 받아온 multi - scale feature level
        num_cams=num_cams, # 사용되는 카메라 수 여기서는 7개
        embed_dims=_dim_, # 256 임베딩 차원의 크기를 지정
        rotate_prev_bev=True, # 이전 프레임의 BEV 특징을 현재 프레임의 좌표계로 회전시켜 사용하도록 설정
        use_shift=True, # BEV 특징을 이동시키는 기법을 사용하여 성능을 향상 -> Deformable Attention때문에 더 다양한 것을 볼 수 있다. 그리고 위치정보는 어차피 positional embeding을 하기 때문에 위치정저는 보존
        use_can_bus=True, # can bus data를 통해 차량의 움직임을 보정
        pc_range=point_cloud_range, # 포인트 클라우드 데이터의 범위를 지정한다.
        bev_h=bev_h_, # BEV feature Height
        bev_w=bev_w_, # BEV feature Width
        rotate_center=[bev_h_//2, bev_w_//2], # BEV 특징을 회전시킬 때의 중심점을 지정합니다.
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4, # 3D 공간을 기둥(Pillar) 형태로 나눌떄, 각 기둥 내에서 샘플링할 점의 수(Z 값)
            return_intermediate=False, # num_layer 모두 거친 최종 출력만 반환
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='SpatialCrossAttention',
                        embed_dims=_dim_,
                        num_cams=num_cams,
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8, # number of offset, 샘플링 할 점의 갯수
                            num_levels=_num_levels_)
                    )
                ],
                ffn_cfgs=_ffn_cfg_,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_, # 이 친구가 128인 이유는 row에서 128, col에서 128 이렇게 2개를 연결해서 더해주는 느낌
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
    ),



    bbox_head=dict(
        type='CustomDeformableDETRHead',
        num_query=100,
        num_classes=13,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=_dim_),
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_pos_dim_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0),
        test_cfg=dict(max_per_img=100)),





    lane_head=dict(
        type='TopoLogicHead',
        num_classes=1,
        in_channels=_dim_,
        num_query=200,

        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        pts_dim=pts_dim,
        sync_cls_avg_factor=False,
        code_size=code_size,
        code_weights= [1.0 for i in range(code_size)],
        transformer=dict(
            type='TopoLogicTransformerDecoderOnly',
            embed_dims=_dim_,
            pts_dim=pts_dim,
            decoder=dict(
                type='TopoLogicSGNNDecoder',
                pc_range=point_cloud_range,
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='SGNNDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN_SGNN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_te_classes=13,
                        edge_weight=0.6),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        lclc_head=dict(
            type='SingleLayerRelationshipHead',
            in_channels_o1=_dim_,
            in_channels_o2=_dim_,
            shared_param=False,
            loss_rel=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=5)),
        lcte_head=dict(
            type='SingleLayerRelationshipHead',
            in_channels_o1=_dim_,
            in_channels_o2=_dim_,
            shared_param=False,
            loss_rel=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=5)),
        bbox_coder=dict(type='LanePseudoCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.025)),
    # model training and testing settings
    train_cfg=dict(
        bbox=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))),
        lane=dict(
            assigner=dict(
                type='LaneHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.025),
                pc_range=point_cloud_range))))

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3DLane',
         with_lane_3d=True, with_lane_label_3d=True, with_lane_adj=True,
         with_bbox=True, with_label=True, with_lane_lcte_adj=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='GridMaskMultiViewImage'),
    dict(type='LaneParameterize3D', method=para_method, method_para=method_para),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=[
        'img', 'gt_lanes_3d', 'gt_lane_labels_3d', 'gt_lane_adj',
        'gt_bboxes', 'gt_labels', 'gt_lane_lcte_adj'])
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        split='train',
        filter_map_change=True,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        test_mode=True)
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


checkpoint_config = dict(interval=1, max_keep_ckpts=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
