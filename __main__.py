import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger


import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        return COCOEvaluator(dataset_name, output_dir=output_folder)

if __name__ == '__main__':

    ### Prepare data
    register_coco_instances("train_set", {}, "/src/train_df.json", "/src/Dataset")
    register_coco_instances("val_set", {}, "/src/val_df.json", "/src/Dataset")

    ### Setup and train
    cfg = get_cfg()

    #setting backbone
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))

    #defining datasets
    cfg.DATASETS.TRAIN = ('train_set',)
    cfg.DATASETS.TEST = ('val_set',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2

    #loading weights (transfer learning)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")

    #learning rate monitor
    cfg.SOLVER.BASE_LR = 1e-3
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = []
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    #new top layer for new application
    cfg.MODEL.RETINANET.NUM_CLASSES = 7

    #output for final weights, tensorboard metrics, etc
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = '/src/models/retinanet_R101.yaml'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #Training or resuming interrupted train
    trainer = Trainer(cfg) 
    trainer.resume_or_load()

    trainer.train()

    #saving checkpoints
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)



