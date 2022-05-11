import torch, torchvision
import detectron2
import time
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import pandas as pd
import os, json, cv2, random, glob
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode


if __name__ == '__main__':

    # Prepare data
    register_coco_instances("train", {}, "/src/train_df.json", "/src/Dataset")
    register_coco_instances("val", {}, "/src/val_df.json", "/src/Dataset")

    # Setup and train
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    #cfg.DATASETS.TRAIN = ('taco_train',)
    #cfg.DATASETS.TEST = ('taco_val',)
    cfg.OUTPUT_DIR = '/src/models/retinanet_R101.yaml'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.RETINANET.NUM_CLASSES = 7

    #cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.25
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)
    
    if os.path.exists(os.path.join( cfg.OUTPUT_DIR, 'samples' )):
        filenames = glob.glob(os.path.join( cfg.OUTPUT_DIR, 'samples', '*' ))
        for path in filenames:
            os.remove(path)
    else:
        os.mkdir( os.path.join( cfg.OUTPUT_DIR, 'samples' ) )

    # Make inferences and draw images
    predictor = DefaultPredictor(cfg)
    val_metadata = MetadataCatalog.get('val')
    val_data = DatasetCatalog.get('val')

    #create dataset
    list = []

    #predict
    start_time = time.time()
    samples = 0
    for i, d in enumerate(val_data):
        img = cv2.imread(d['file_name'])
        outputs = predictor(img)
        print(outputs)
        print(outputs['instances'].pred_boxes.tensor)
        tensor = outputs['instances'].pred_boxes.tensor.tolist()
        list.append((d['file_name'], tensor))
        samples=samples+1

    print(f"--- {(time.time() - start_time)} seconds for predicting {samples} samples ---")


    with open('/src/predicts.json','w') as f:
        json.dump(list, f)

    for i, d in enumerate(val_data):
        print('Processing image', d['file_name'])
        img = cv2.imread(d['file_name'])
        outputs = predictor(img)
        print(outputs)
        print(outputs['instances'].pred_boxes.tensor)
        tensor = outputs['instances'].pred_boxes.tensor.tolist()
        

        v = Visualizer( img[:, :, ::-1],
                        metadata=val_metadata, 
                        #scale=0.5, 
                        #instance_mode=ColorMode.IMAGE_BW
                        )
        out = v.draw_instance_predictions(outputs['instances'].to('cpu') )
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, 'samples', '%d.jpg'%i),
                    out.get_image()[:, :, ::-1])