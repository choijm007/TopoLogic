import torch
from mmcv import Config
from projects.topologic.datasets.pl_dataset import TopoLogicDataModule
from projects.topologic.models.detectors.pl_topologic import TopoLogicPL


def check_data_and_feature_extraction():
    """
    이 함수는 다음 두 가지를 검증합니다:
    1. TopoLogicDataModule이 데이터를 올바르게 로드하는지 확인합니다.
    2. 로드된 데이터를 TopoLogicPL 모델에 전달하여,
       extract_feat 메서드가 성공적으로 이미지 특징을 추출하는지 확인합니다.
    """
    # 1. 설정 파일 로드
    cfg_path = '/home/ircvlab-504/TopoLogic_Lightning/projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py'
    cfg = Config.fromfile(cfg_path)

    # 2. TopoLogicDataModule 인스턴스 생성
    # 학습 데이터셋 설정을 사용합니다.
    data_module = TopoLogicDataModule(
        data_root=cfg.data_root,
        ann_file=cfg.data.train.ann_file,
        batch_size=1,  # 테스트를 위해 배치 크기는 1로 설정
        num_workers=0,
        queue_length=cfg.data.train.queue_length if 'queue_length' in cfg.data.train else 1,
        filter_empty_te=False,
        split='train',
        filter_map_change=True,
        # train_pipeline을 전달해야 CustomFormatBundle3DLane 등을 통해
        # 'img' 키를 가진 텐서가 생성됩니다.
        pipeline=cfg.train_pipeline,
        modality=cfg.input_modality,
        classes=cfg.class_names,
        test_mode=False
    )

    # 데이터셋 셋업
    data_module.setup('fit')

    # 학습 데이터로더에서 데이터 배치 하나 가져오기
    train_loader = data_module.train_dataloader()
    data_batch = next(iter(train_loader))

    print("--- 1. Data Batch Loaded Successfully ---")
    print("Batch keys:", data_batch.keys())
    # 'img' 키가 존재하는지, 텐서의 모양은 어떤지 확인
    if 'img' in data_batch:
        # img는 이제 텐서가 아닌 리스트이므로, 리스트의 첫 번째 요소에 접근해야 합니다.
        img_tensor = data_batch['img'][0]
        img_tensor = img_tensor[:,-1,...]
        print(f"Image tensor shape: {img_tensor.shape}")
        print("----------------------------------------\n")
    else:
        print("'img' key not found in the batch. Check your data pipeline.")
        return

    # 3. TopoLogicPL 모델 인스턴스 생성
    model = TopoLogicPL(cfg_path)
    model.eval()  # 테스트를 위해 평가 모드로 설정

    print("--- 2. Running Feature Extraction ---")

    # 4. 특징 추출 실행 (no_grad 컨텍스트에서 실행하여 메모리 절약)
    with torch.no_grad():
        # 모델의 extract_feat 메서드를 직접 호출하여 테스트
        img_feats_reshaped = model.extract_feat(img_tensor)

    print("Feature extraction successful!")
    print("\nOutput feature shapes:")
    for i, feat in enumerate(img_feats_reshaped):
        print(f"Level {i}: {feat.shape}")
    print("----------------------------------")


if __name__ == '__main__':
    check_data_and_feature_extraction()