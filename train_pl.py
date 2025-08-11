import pytorch_lightning as pl
from mmcv import Config
from projects.topologic.models.detectors.pl_topologic import TopoLogicPL
from projects.topologic.datasets.pl_dataset import TopoLogicDataModule
def main():
    # 1. 설정 파일 경로 지정
    cfg_path = '/home/ircvlab-504/TopoLogic_Lightning/projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py'
    cfg = Config.fromfile(cfg_path)

    # 2. 데이터 모듈 초기화
    # 설정 파일에서 val_pipeline을 명시적으로 전달합니다.
    data_module = TopoLogicDataModule(
        data_root=cfg.data_root,
        ann_file=cfg.data.train.ann_file,
        batch_size=cfg.data.samples_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        queue_length=cfg.data.train.get('queue_length', 1),
        train_pipeline=cfg.train_pipeline,
        test_pipeline=cfg.test_pipeline,
        modality=cfg.input_modality,
        classes=cfg.class_names,
    )

    # 3. 모델 초기화
    model = TopoLogicPL(cfg_path)

    # 4. Trainer 설정
    # TensorBoard 로거 설정
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="topologic_pl")

    # 체크포인트 콜백 설정
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train/total_loss',
        dirpath='pl_checkpoints',
        filename='topologic-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=cfg.total_epochs,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        # 디버깅을 위해 빠른 실행 모드 (1개의 배치만 학습 및 검증)
        # fast_dev_run=True,
    )

    # 5. 학습 시작
    print("--- Starting Training ---")
    trainer.fit(model, datamodule=data_module)
    print("--- Training Finished ---")

if __name__ == '__main__':
    main()
