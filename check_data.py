import torch
from projects.topologic.datasets.pl_dataset import TopoLogicDataModule
from mmcv import Config

def check_data():
    # 설정 파일 로드
    cfg = Config.fromfile('/home/ircvlab-504/TopoLogic_Lightning/projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py')

    # TopoLogicDataModule 인스턴스 생성
    data_module = TopoLogicDataModule(
        data_root=cfg.data_root,
        ann_file=cfg.data.train.ann_file,
        batch_size=1,
        num_workers=0,
        queue_length=1,
        filter_empty_te=False,
        split='train',
        filter_map_change=True,
        pipeline=cfg.train_pipeline,
        modality=cfg.input_modality,
        classes=cfg.class_names,
        test_mode=False
    )

    # 데이터셋 설정
    data_module.setup('fit')

    # 학습 데이터로더에서 데이터 하나 가져오기
    train_loader = data_module.train_dataloader()
    data_iter = iter(train_loader)
    data_batch = next(data_iter)

    # 데이터 내용 출력
    print("--- Data Batch ---")
    # for key, value in data_batch.items():
    #     if hasattr(value, 'data'):
    #         data_batch[key] = value.data
    #
    # for key, value in data_batch.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: tensor with shape {value.shape}")
    #     elif isinstance(value, list):
    #         print(f"{key}: list with {len(value)} items")
    #         if len(value) > 0:
    #             if isinstance(value[0], torch.Tensor):
    #                 print(f"  - First item shape: {value[0].shape}")
    #             else:
    #                 print(f"  - First item type: {type(value[0])}")
    #     else:
    #         print(f"{key}: {type(value)}")

    for i, data_batch in enumerate(train_loader):
        print(data_batch)

    print("------------------")

if __name__ == '__main__':
    check_data()
