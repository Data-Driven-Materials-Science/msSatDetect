'''
Stage 1 of Pipeline:
    Model training and calculating Accuracy of model
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
print("Imports loaded correctly")

'''---------------------------'''
EXPERIMENT_NAME = 'satellite' # can be 'particles' or 'satellite'
json_path_correct = Path('..', 'data','newData', 'sat_char_extract.json')  # path to training data

assert json_path_correct.is_file(), 'training file not found!'


'''---------------------------'''
DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times

# store names of datasets that will be registered for easier access later
dataset_train = f'{EXPERIMENT_NAME}_Train'

# register the training dataset
DatasetCatalog.register(dataset_train, lambda f = json_path_correct: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                                                     im_root=f,  # path to the training data json file
                                                                                                     dataset_class='Train'))  # indicates this is training data
# register the validation dataset

'''---------------------------'''
#SWAPPING
#print(f'Registered Datasets: {list(DatasetCatalog._REGISTERED.keys())}')
print(f'Registered Datasets: {DatasetCatalog.list()}')
'''---------------------------'''
'''
## There is also a metadata catalog, which stores the class names.
for d in [dataset_train, dataset_valid]:
    MetadataCatalog.get(d).set(**{'thing_classes': [EXPERIMENT_NAME]})'''

'''---------------------------'''
for i in DatasetCatalog.get(dataset_train):
    visualize.display_ddicts(i, '../data/newData/output/touchedUp/', dataset_train, suppress_labels=True)

'''---------------------------'''
