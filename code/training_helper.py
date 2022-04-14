'''
Helper File for Training Mask R-CNN Model
'''

## regular module imports
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import skimage.io
import sys

## detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import BoxMode

## ampis
ampis_root = Path('../../')
sys.path.append(str(ampis_root))

from ampis import data_utils, visualize, analyze
from ampis.applications import powder
from ampis.structures import InstanceSet
from ampis.visualize import display_iset
import seaborn as sns



'''
Parameters Required:
------------------------------
Object Name
json_path_train
json_path_val
InputMaskType
imagesPerBatch
CheckpointPeriod
Model Device
Model ROI Num Classes
Detections Per Image
Max Iterations
'''

def trainModel(objectName, json_path_train, json_path_val, maxIterations=5000, checkpointIteration=5001, experimentNum='0', 
    modelDevice='cuda', detectionsPerImage=250, visualizeTrainVal=False, imagesPerBatch=1,numClasses=1, inputMaskType='polygon'):
    '''
    Parameters:
    objectName: the name of the object that is being detected. Largely used for fileSystem organization
        Type: String
        Default Value: must be passed in by user
    json_path_train: path to training annotations
        Type: pathLib Path()
        Default Value: must be passed in by user
    json_path_val: path to validation annotations
        Type: pathLib Path()
        Default Value: must be passed in by user
    maxIterations: total number of training iterations 
        Type: Integer
        Default Value: 5,000
        Note on changing value: 
            Increasing this number will improve accuracy. However, this will also increase training time
            Additionally, increasing this number too much may cause the model to overfit to the training data
    checkpointIteration: Number of Iterations before a checkpoint is stored
        Type: Integer
        Default Value: 5,001
        Note on changing value: 
            Currently Set to not store a checkpoint. Reduce this value to store 1 (or more) checkpoints. Will save everytime
            num iterations trained is divisble by this value
    experimentNum: Documentation Purposes. Defines a couple director and file names
        Type: String
        Default Value: '0'
        Note on changing value: 
            This is only used for documentation purposes on where to store files. If this value is not changed, it will overwrite any data 
            in the same location with shared names. It is recommended to make this parameter distinct each time a model is trained unless 
            intended to replace previous work
    modelDevice:
        Type: String
        Default Value: 'cuda'
        Alternative Value: 'cpu'
        Note on changing value: 
            Setting this value to 'cuda' performs these computations on a GPU. 'cpu' will force these computations to 
    detectionsPerImage: Total Allows Detections per image
        Type: Integer
        Default Value: 250
        Note on changing value:
            Limits how many total detections can be in an image. Should be the upper threshold of possible instances in a single image
    visualizeTrainVal: Allows or prevents visualization of the training and validation set
        Type: Boolean
        Default Value: False
        Alternative Value: True
        Note on changing value:
            Should only be set to True if a graphical interface is available, and you are unfamiliar with the dataset
    imagesPerBatch:
        Type: Integer
        Default Value: 1
        Note on changing value: 
            Increasing this value too much may cause the memory required to exceed the what is computationally accessible, causing it to 
            either crash, or begin overwriting data 
    numClasses: Number of classes to detect with this model
        Type: Integer
        Default Value: 1
        Note on changing value:
            This code is set up for single class detection, similar to mask type, changing this may work, but has not been tested and is not gaurunteed
    inputMaskType:
        Type: String
        Default Value: 'polygon'
        Note on changing value: 
            While other mask types exist, support code is specifically set up for polygon format. Changing this may cause problems
    
     '''
    assert json_path_train.is_file(), 'training file not found!'
    assert json_path_val.is_file(), 'validation file not found!'
    DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times

    # store names of datasets that will be registered for easier access later
    dataset_train = f'{objectName}_Train'
    dataset_valid = f'{objectName}_Val'

    # register the training dataset
    DatasetCatalog.register(dataset_train, lambda f = json_path_train: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                                            im_root=f,  # path to the training data json file
                                                                                            dataset_class='Train'))  # indicates this is training data
    # register the validation dataset
    DatasetCatalog.register(dataset_valid, lambda f = json_path_val: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                                            im_root=f,  # path to validation data json file
                                                                                            dataset_class='Validation'))  # indicates this is validation data
    '''---------------------------'''
    #SWAPPING
    #print(f'Registered Datasets: {list(DatasetCatalog._REGISTERED.keys())}')
    print(f'Registered Datasets: {DatasetCatalog.list()}')
    '''---------------------------'''
    ## There is also a metadata catalog, which stores the class names.
    for d in [dataset_train, dataset_valid]:
        MetadataCatalog.get(d).set(**{'thing_classes': [EXPERIMENT_NAME]})

    if visualizeTrainVal:
        for i in np.random.choice(DatasetCatalog.get(dataset_train), 3, replace=False):
            visualize.display_ddicts(i, None, dataset_train, suppress_labels=True)
        for i in DatasetCatalog.get(dataset_valid):
            visualize.display_ddicts(i, None, dataset_valid, suppress_labels=True)
    cfg = get_cfg() # initialize cfg object
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))  # load default parameters for Mask R-CNN
    cfg.INPUT.MASK_FORMAT = 'polygon'  # masks generated in VGG image annotator are polygons
    cfg.DATASETS.TRAIN = (dataset_train,)  # dataset used for training model
    cfg.DATASETS.VALIDATION = (dataset_valid,)
    cfg.DATASETS.TEST = (dataset_train, dataset_valid)  # we will look at the predictions on both sets after training
    cfg.SOLVER.IMS_PER_BATCH = 1 # number of images per batch (across all machines)
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpointIteration  # number of iterations after which to save model checkpoints
    cfg.MODEL.DEVICE = modelDevice  # 'cpu' to force model to run on cpu, 'cuda' if you have a compatible gpu
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Since we are training separate models for particles and satellites there is only one class output
    cfg.TEST.DETECTIONS_PER_IMAGE = 400 if EXPERIMENT_NAME == 'particle' else 150  # maximum number of instances that can be detected in an image (this is fixed in mask r-cnn)
    cfg.SOLVER.MAX_ITER = maxIterations  # maximum number of iterations to run during training
                                # Increasing this may improve the training results, but will take longer to run (especially without a gpu!)












'''---------------------------'''
EXPERIMENT_NAME = 'satellite' # can be 'particles' or 'satellite'
json_path_train = Path('..', 'data','training', f'{EXPERIMENT_NAME}_training.json')  # path to training data
json_path_val = Path('..', 'data','training', f'{EXPERIMENT_NAME}_validation.json')  # path to training data

assert json_path_train.is_file(), 'training file not found!'
assert json_path_val.is_file(), 'validation file not found!'


'''---------------------------'''
DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times

# store names of datasets that will be registered for easier access later
dataset_train = f'{EXPERIMENT_NAME}_Train'
dataset_valid = f'{EXPERIMENT_NAME}_Val'

# register the training dataset
DatasetCatalog.register(dataset_train, lambda f = json_path_train: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                                                     im_root=f,  # path to the training data json file
                                                                                                     dataset_class='Train'))  # indicates this is training data
# register the validation dataset
DatasetCatalog.register(dataset_valid, lambda f = json_path_val: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                                                im_root=f,  # path to validation data json file
                                                                                                dataset_class='Validation'))  # indicates this is validation data
'''---------------------------'''
#SWAPPING
#print(f'Registered Datasets: {list(DatasetCatalog._REGISTERED.keys())}')
print(f'Registered Datasets: {DatasetCatalog.list()}')
'''---------------------------'''

## There is also a metadata catalog, which stores the class names.
for d in [dataset_train, dataset_valid]:
    MetadataCatalog.get(d).set(**{'thing_classes': [EXPERIMENT_NAME]})

'''---------------------------'''
for i in np.random.choice(DatasetCatalog.get(dataset_train), 3, replace=False):
    visualize.display_ddicts(i, None, dataset_train, suppress_labels=True)

'''---------------------------'''

for i in DatasetCatalog.get(dataset_valid):
    visualize.display_ddicts(i, None, dataset_valid, suppress_labels=True)
'''---------------------------'''
cfg = get_cfg() # initialize cfg object
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))  # load default parameters for Mask R-CNN
cfg.INPUT.MASK_FORMAT = 'polygon'  # masks generated in VGG image annotator are polygons
cfg.DATASETS.TRAIN = (dataset_train,)  # dataset used for training model
cfg.DATASETS.VALIDATION = (dataset_valid,)
cfg.DATASETS.TEST = (dataset_train, dataset_valid)  # we will look at the predictions on both sets after training
cfg.SOLVER.IMS_PER_BATCH = 1 # number of images per batch (across all machines)
cfg.SOLVER.CHECKPOINT_PERIOD = 15000  # number of iterations after which to save model checkpoints
cfg.MODEL.DEVICE='cuda'  # 'cpu' to force model to run on cpu, 'cuda' if you have a compatible gpu
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Since we are training separate models for particles and satellites there is only one class output
cfg.TEST.DETECTIONS_PER_IMAGE = 400 if EXPERIMENT_NAME == 'particle' else 150  # maximum number of instances that can be detected in an image (this is fixed in mask r-cnn)
cfg.SOLVER.MAX_ITER = 15000  # maximum number of iterations to run during training
                            # Increasing this may improve the training results, but will take longer to run (especially without a gpu!)

# model weights will be downloaded if they are not present
weights_path = Path('..','models','model_final_f10217.pkl')
if weights_path.is_file():
    print('Using locally stored weights: {}'.format(weights_path))
else:
    weights_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    print('Weights not found, weights will be downloaded from source: {}'.format(weights_path))
cfg.MODEL.WEIGHTs = str(weights_path)
cfg.OUTPUT_DIR = str(Path(f'../models/{EXPERIMENT_NAME}_output_1'))
# make the output directory
os.makedirs(Path(cfg.OUTPUT_DIR), exist_ok=True)

'''---------------------------'''
trainer = DefaultTrainer(cfg)  # create trainer object from cfg
trainer.resume_or_load(resume=False)  # start training from iteration 0
trainer.train()  # train the model!
