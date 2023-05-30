import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
import pycocotools.mask as RLE
import seaborn as sns
import skimage
import skimage.io

ampis_root = str(Path('..','..'))
import sys
if ampis_root not in sys.path:
    sys.path.append(ampis_root)

from ampis import analyze, data_utils
from ampis.applications import powder
from ampis.structures import InstanceSet
from ampis.visualize import display_iset

'''------------------------------------'''


print('This is starting')
## load ground truth labels

via_path = Path('data','via_2.0.8')

particles_gt_path_train = via_path / 'via_powder_particle_masks_training.json'
particles_gt_path_valid = via_path / 'via_powder_particle_masks_validation.json'

satellites_gt_path_train = via_path / 'via_powder_satellite_masks_training.json'
satellites_gt_path_valid = via_path / 'via_powder_satellite_masks_validation.json'

for path in [particles_gt_path_train, particles_gt_path_valid, satellites_gt_path_train, satellites_gt_path_valid]:
    assert path.is_file(), f'File not found : {path}'

particles_gt_dd_train = data_utils.get_ddicts('via2', particles_gt_path_train, dataset_class='train')
particles_gt_dd_valid = data_utils.get_ddicts('via2', particles_gt_path_valid, dataset_class='validation')

satellites_gt_dd_train = data_utils.get_ddicts('via2', satellites_gt_path_train, dataset_class='train')
satellites_gt_dd_valid = data_utils.get_ddicts('via2', satellites_gt_path_valid, dataset_class='validation')

'''------------------------------------'''
## load predicted labels

particles_path = Path('data','sample_particle_outputs.pickle')
assert particles_path.is_file()

satellites_path = Path('data','sample_satellite_outputs.pickle')
assert satellites_path.is_file()

with open(particles_path, 'rb') as f:
    particle_pred = pickle.load(f)

with open(satellites_path, 'rb') as f:
    satellites_pred = pickle.load(f)

'''------------------------------------'''
# Ground truth instance sets

iset_particles_gt = [InstanceSet().read_from_ddict(x,   # data 
                                                   inplace=False  # returns the set so it can be added to the list
                                                  ) for x in particles_gt_dd_train]

# instead of creating a separate list, we add the validation results to the training ones to make it easier later
iset_particles_gt.extend([InstanceSet().read_from_ddict(x, inplace=False) for x in particles_gt_dd_valid])

iset_satellites_gt = [InstanceSet().read_from_ddict(x, inplace=False) for x in satellites_gt_dd_train]
iset_satellites_gt.extend([InstanceSet().read_from_ddict(x, inplace=False) for x in satellites_gt_dd_valid])

# Predicted instance sets
iset_particles_pred = [InstanceSet().read_from_model_out(x, inplace=False) for x in particle_pred]
iset_satellites_pred = [InstanceSet().read_from_model_out(x, inplace=False) for x in satellites_pred]

#iset_particles_gt
'''------------------------------------'''
iset_particles_gt, iset_particles_pred = analyze.align_instance_sets(iset_particles_gt, iset_particles_pred)
iset_satellites_gt, iset_satellites_pred = analyze.align_instance_sets(iset_satellites_gt, iset_satellites_pred)

for gt, pred in zip(iset_particles_gt, iset_particles_pred):
    pred.HFW = gt.HFW
    pred.HFW_units = gt.HFW_units
    print(f'gt filename: {Path(gt.filepath).name}\t pred filename: {Path(pred.filepath).name}')

'''------------------------------------'''

dss_particles = [analyze.det_seg_scores(gt, pred, size=gt.instances.image_size) 
                 for gt, pred in zip(iset_particles_gt, iset_particles_pred)]
dss_satellites = [analyze.det_seg_scores(gt, pred, size=gt.instances.image_size)
                 for gt, pred in zip(iset_satellites_gt, iset_satellites_pred)]

'''------------------------------------'''

labels = []
counts = {'train': 0, 'validation': 0}

# the filenames are not helpful, so we will map them to labels ie ('Train 1', 'Train 2', 'Validation 1', etc)
for iset in iset_particles_gt:
    counts[iset.dataset_class] += 1
    labels.append('{} {}'.format(iset.dataset_class, counts[iset.dataset_class]))

# x values are arbitrary, we just want 2 values, 1 for precision, 2 for recall
x=[*([1] * len(labels)), *([2] * len(labels))]
# y values are the bar heights
scores = [*[x['det_precision'] for x in dss_particles],
     *[x['det_recall'] for x in dss_particles]]

# since we are plotting precision and recall on the same plot we need 2 sets of labels
labels = labels * 2
print('x: ', x)
print('y: ', [np.round(x, decimals=2) for x in scores])
print('labels: ', labels)

'''------------------------------------'''

fig, ax = plt.subplots(figsize=(6,3), dpi=150)
sns.barplot(x=x, y=scores, hue=labels, ax=ax)
ax.legend(bbox_to_anchor=(1,1))
ax.set_ylabel('detection score')
ax.set_xticklabels(['precision','recall'])

'''------------------------------------'''

gt = iset_particles_gt[-1]
pred = iset_particles_pred[-1]
iset_det, colormap = analyze.det_perf_iset(gt, pred)
img = skimage.io.imread(iset.filepath)
display_iset(img, iset=iset_det)

'''------------------------------------'''

iset_seg, (colors, color_labels) = analyze.seg_perf_iset(gt, pred,)
display_iset(img, iset=iset_seg, apply_correction=True)

'''------------------------------------'''

# select instances for the same image
gt_s, pred_s = [(x, y) for x, y in zip(iset_satellites_gt, iset_satellites_pred) if str(x.filepath) == pred.filepath][0]
iset_det_s, colormap_s = analyze.det_perf_iset(gt_s, pred_s)
display_iset(img, iset=iset_det_s)

'''------------------------------------'''

for iset in [*iset_particles_gt, *iset_particles_pred]:
    if iset.rprops is None:  # avoid re-computing regionprops if cell has already been run
        iset.compute_rprops()  # since rprops requires the masks to be uncompressed, this takes a bit longer to run
iset_particles_pred[-1].rprops.head()

'''------------------------------------'''

print('ground truth PSD')
areas_gt = powder.psd(iset_particles_gt)
print('predicted PSD')
areas_pred = powder.psd(iset_particles_pred)

'''------------------------------------'''

iset_particles_gt_ss, iset_satellites_gt_ss = analyze.align_instance_sets(iset_particles_gt, iset_satellites_gt)
iset_particles_pred_ss, iset_satellites_pred_ss = analyze.align_instance_sets(iset_particles_pred, iset_satellites_pred)
psi_gt = []
psi_pred = []
for pg, pp, sg, sp in zip(iset_particles_gt_ss, iset_particles_pred_ss, iset_satellites_gt_ss, iset_satellites_pred_ss):
    files = [Path(x).name for x in [pg.filepath, pp.filepath, sg.filepath, sp.filepath]]
    assert all([x == files[0] for x in files])  # the files are in the same order and there are no excess files
    psi_gt.append(powder.PowderSatelliteImage(particles=pg, satellites=sg))
    psi_pred.append(powder.PowderSatelliteImage(particles=pp, satellites=sp))

'''------------------------------------'''

for gt, pred in zip(psi_gt, psi_pred):
    for psi in [gt, pred]:
        psi.compute_matches()

'''------------------------------------'''

gt = psi_gt[0]
pred = psi_pred[0]
gt_idx = np.random.choice(list(gt.matches['match_pairs'].keys()), 3)
pred_idx = np.random.choice(list(pred.matches['match_pairs'].keys()), 3)
fig, ax = plt.subplots(2,3)
for i, (g, p) in enumerate(zip(gt_idx, pred_idx)):
    gt.visualize_particle_with_satellites(g, ax[0, i])
    pred.visualize_particle_with_satellites(p, ax[1, i])
ax[0,0].set_title('ground truth')
ax[1,0].set_title('predicted')

'''------------------------------------'''

print('ground truth results')
results_gt = powder.satellite_measurements(psi_gt, print_summary=True, output_dict=True)
print('predicted results')
results_pred = powder.satellite_measurements(psi_pred, True, True)
'''------------'''
#results_gt
