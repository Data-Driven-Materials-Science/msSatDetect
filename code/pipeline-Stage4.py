## regular module imports
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from IPython.display import display
import pickle
import skimage.io
import pandas as pd
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

#satellites_gt_path_train = Path('..', 'data', 'training', 'satellite_training.json')
satellites_gt_path_train = Path('..', 'data', 'newData', 'sat_char_extract.json')
satellites_gt_dd_train = data_utils.get_ddicts('via2', satellites_gt_path_train, dataset_class='train')
iset_satellites_gt = [InstanceSet().read_from_ddict(x, inplace=False) for x in satellites_gt_dd_train]
print('Num Satellite Images: ' + str(len(iset_satellites_gt)))
k = ['area', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity']
for iset in iset_satellites_gt:
    if iset.rprops is None:
        iset.compute_rprops(keys=k)

#The following loop prints each image name with the shape of the dataframe corresponding to the number of Satellites
'''
for i in range(len(iset_satellites_gt)):
    name = str(iset_satellites_gt[i].filepath).split('/')[-1]
    print(name)
    print(iset_satellites_gt[i].rprops.shape)'''

rowNames = ['numSats', 'avgArea', 'minArea', 'maxArea', 'stdDevArea','avgPerimeter', 'minPerimeter', 'maxPerimeter', 'stdDevPerimeter',
            'avgEccentricity', 'minEccentricity', 'maxEccentricity', 'stdDevEccentricity','avgMinor', 'minMinor', 'maxMinor', 'stdDevMinor',
            'avgMajor', 'minMajor', 'maxMajor', 'stdDevMajor','avgDiamter', 'minDiameter', 'maxDiameter', 'stdDevDiamater', ]
colNames = []
results = []
for imageNum in range(len(iset_satellites_gt)):
    tempName = (str(iset_satellites_gt[imageNum].filepath).split('/')[-1]).split('.')[0]
    colNames.append(str(tempName))
    temp = [float(len(iset_satellites_gt[imageNum].rprops.area)),
            float(sum(iset_satellites_gt[imageNum].rprops.area) / len(iset_satellites_gt[imageNum].rprops.area)),
            float(min(iset_satellites_gt[imageNum].rprops.area)),
            float(max(iset_satellites_gt[imageNum].rprops.area)),
            float(np.std(iset_satellites_gt[imageNum].rprops.area)),
            float(sum(iset_satellites_gt[imageNum].rprops.perimeter) / len(iset_satellites_gt[imageNum].rprops.perimeter)),
            float(min(iset_satellites_gt[imageNum].rprops.perimeter)),
            float(max(iset_satellites_gt[imageNum].rprops.perimeter)),
            float(np.std(iset_satellites_gt[imageNum].rprops.perimeter)),
            float(sum(iset_satellites_gt[imageNum].rprops.eccentricity) / len(iset_satellites_gt[imageNum].rprops.eccentricity)),
            float(min(iset_satellites_gt[imageNum].rprops.eccentricity)),
            float(max(iset_satellites_gt[imageNum].rprops.eccentricity)),
            float(np.std(iset_satellites_gt[imageNum].rprops.eccentricity)),
            float(sum(iset_satellites_gt[imageNum].rprops.minor_axis_length) / len(iset_satellites_gt[imageNum].rprops.minor_axis_length)),
            float(min(iset_satellites_gt[imageNum].rprops.minor_axis_length)),
            float(max(iset_satellites_gt[imageNum].rprops.minor_axis_length)),
            float(np.std(iset_satellites_gt[imageNum].rprops.minor_axis_length)),
            float(sum(iset_satellites_gt[imageNum].rprops.major_axis_length) / len(iset_satellites_gt[imageNum].rprops.major_axis_length)),
            float(min(iset_satellites_gt[imageNum].rprops.major_axis_length)),
            float(max(iset_satellites_gt[imageNum].rprops.major_axis_length)),
            float(np.std(iset_satellites_gt[imageNum].rprops.major_axis_length)),
            float(sum(iset_satellites_gt[imageNum].rprops.equivalent_diameter) / len(iset_satellites_gt[imageNum].rprops.equivalent_diameter)),
            float(min(iset_satellites_gt[imageNum].rprops.equivalent_diameter)),
            float(max(iset_satellites_gt[imageNum].rprops.equivalent_diameter)),
            float(np.std(iset_satellites_gt[imageNum].rprops.equivalent_diameter))]
    results.append(temp)
df = dfObj = pd.DataFrame(results, columns = rowNames, index=colNames)
df = df.T
df.to_csv('../data/newData/output/metrics.csv')
print(df)


#Prints out Average, Min, Max, and Std Deviation of Area, Equivalent Diemeter, Major Axis, Minor Axis, Perimeter and Eccentricity
'''print('Image One Metrics')
imageNum = 2
print('-' * 30)
print("AREA: ")
print('Average: ' + str(sum(iset_satellites_gt[imageNum].rprops.area) / len(iset_satellites_gt[imageNum].rprops.area)))
print('Min: ' + str(min(iset_satellites_gt[imageNum].rprops.area)))
print('Max: ' + str(max(iset_satellites_gt[imageNum].rprops.area)))
print('Std Deviation: ' + str(np.std(iset_satellites_gt[imageNum].rprops.area)))
print('-' * 30)
print("PERIMETER: ")
print('Average: ' + str(sum(iset_satellites_gt[imageNum].rprops.perimeter) / len(iset_satellites_gt[imageNum].rprops.perimeter)))
print('Min: ' + str(min(iset_satellites_gt[imageNum].rprops.perimeter)))
print('Max: ' + str(max(iset_satellites_gt[imageNum].rprops.perimeter)))
print('Std Deviation: ' + str(np.std(iset_satellites_gt[imageNum].rprops.perimeter)))
print('-' * 30)
print("ECCENTRICITY: ")
print('Average: ' + str(sum(iset_satellites_gt[imageNum].rprops.eccentricity) / len(iset_satellites_gt[imageNum].rprops.eccentricity)))
print('Min: ' + str(min(iset_satellites_gt[imageNum].rprops.eccentricity)))
print('Max: ' + str(max(iset_satellites_gt[imageNum].rprops.eccentricity)))
print('Std Deviation: ' + str(np.std(iset_satellites_gt[imageNum].rprops.eccentricity)))
print('-' * 30)
print("MINOR AXIS LENGTH: ")
print('Average: ' + str(sum(iset_satellites_gt[imageNum].rprops.minor_axis_length) / len(iset_satellites_gt[imageNum].rprops.minor_axis_length)))
print('Min: ' + str(min(iset_satellites_gt[imageNum].rprops.minor_axis_length)))
print('Max: ' + str(max(iset_satellites_gt[imageNum].rprops.minor_axis_length)))
print('Std Deviation: ' + str(np.std(iset_satellites_gt[imageNum].rprops.minor_axis_length)))
print('-' * 30)
print("MAJOR AXIS LENGTH: ")
print('Average: ' + str(sum(iset_satellites_gt[imageNum].rprops.major_axis_length) / len(iset_satellites_gt[imageNum].rprops.major_axis_length)))
print('Min: ' + str(min(iset_satellites_gt[imageNum].rprops.major_axis_length)))
print('Max: ' + str(max(iset_satellites_gt[imageNum].rprops.major_axis_length)))
print('Std Deviation: ' + str(np.std(iset_satellites_gt[imageNum].rprops.major_axis_length)))
print('-' * 30)
print("EQUIVALENT DIAMETER: ")
print('Average: ' + str(sum(iset_satellites_gt[imageNum].rprops.equivalent_diameter) / len(iset_satellites_gt[imageNum].rprops.equivalent_diameter)))
print('Min: ' + str(min(iset_satellites_gt[imageNum].rprops.equivalent_diameter)))
print('Max: ' + str(max(iset_satellites_gt[imageNum].rprops.equivalent_diameter)))
print('Std Deviation: ' + str(np.std(iset_satellites_gt[imageNum].rprops.equivalent_diameter)))
print('-' * 30)
'''