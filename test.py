import argparse
import pytorch_lightning as pl
import torch
from mmcv import Config
from projects.topologic.models.detectors.pl_topologic import TopoLogicPL
from projects.topologic.datasets.pl_dataset import TopoLogicDataModule
import random
import numpy as np


def main(cfg_path, checkpoint_path):
    cfg = Config.fromfile(cfg_path)

    # 1. Instantiate DataModule
    data_module = TopoLogicDataModule(
        data_root=cfg.data_root,
        data=cfg.data,
        batch_size=cfg.data.samples_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        queue_length=1,
        train_pipeline=cfg.train_pipeline,  # Not used in test, but required by init
        test_pipeline=cfg.test_pipeline,
        modality=cfg.input_modality,
        classes=cfg.class_names,
    )

    # 2. Instantiate Model and load weights manually
    model = TopoLogicPL(cfg_path)
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    tb_logger = pl.loggers.TensorBoardLogger("tb_logs", name="topologic_pt_test")
    csv_logger = pl.loggers.CSVLogger("tb_logs", name="topologic_pl_test")
    # 3. Setup Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=[tb_logger,csv_logger]
    )

    # 4. Start Testing
    print("--- Starting Testing ---")
    # Pass the model with loaded weights, not the ckpt_path
    trainer.test(model, datamodule=data_module)
    print("--- Testing Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a TopoLogicPL model.')
    parser.add_argument('--config', type=str, default='projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py',
                        help='Path to the model config file.')
    parser.add_argument('--checkpoint', type=str, default='pl_latest.pth',
                        help='Path to the checkpoint file.')
    args = parser.parse_args()

    # Set seed for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main(args.config, args.checkpoint)
