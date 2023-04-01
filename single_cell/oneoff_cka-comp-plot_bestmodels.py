# Kai Sandbrink
# 2021-12-14
# This plot creates the CKA plots for specified models outside of the designated pipeline

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import format_axis

cka_mats_filepaths = [
    '/media/data/DeepDraw/revisions/analysis-data/exp301/analysis/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/comparison/horall/rsa_trreg/cka_matrix.npy',
    '/media/data/DeepDraw/revisions/analysis-data/exp307/analysis/spatiotemporal_4_8-8-32-64_7272/comparison/horall/rsa_trreg/cka_matrix.npy',
    '/media/data/DeepDraw/revisions/analysis-data/exp320/analysis/lstm_3_8-16-16_256/comparison/horall/rsa_trreg/cka_matrix.npy'
]

savepath = '/media/data/DeepDraw/revisions/analysis-data/oneoff'
savefile = 'cka_depth_comp_plot_bestmodels.'

cka_mats = [np.load(filepath) for filepath in cka_mats_filepaths]

modelnames = ['S', 'ST', 'LSTM']
colors = ['C0', 'C2', 'C4']

max_nlayers = 8

fig = plt.figure(figsize=(14,6), dpi=200)
ax = fig.add_subplot(111)

ax.set_xlabel("Network Depth")
ax.set_ylabel("CKA Score")

def adjust_by_depth(layer_cka_scores, nlayers):
    layer_cka_depth_scores = np.empty((max_nlayers + 1,))
    layer_cka_depth_scores[:] = np.nan

    for i in range(nlayers+1):
        layer_cka_depth_scores[int(i*max_nlayers/nlayers)] = layer_cka_scores[i]

    return layer_cka_depth_scores

for cka_mat, nlayer, color, modelname in zip(cka_mats, [8, 4, 4], colors, modelnames):
    cka_scores = cka_mat

    cka_depth_scores = np.apply_along_axis(adjust_by_depth, axis=1, arr=cka_scores, **{'nlayers': nlayer})
    print(cka_depth_scores, cka_depth_scores.shape)

    cka_means = np.nanmean(cka_depth_scores, axis=0)
    cka_err = np.nanstd(cka_depth_scores, axis=0)

    poss = np.arange(max_nlayers+1)

    mask = ~np.isnan(cka_means)

    ax.errorbar(poss[mask], cka_means[mask], cka_err[mask], marker="D", capsize=3.0, label=modelname, color=color)

ax.legend()
format_axis(ax)

ticks = [i/8*max_nlayers for i in range(9)]
ax.set_xticks(ticks)
ax.set_xticklabels(['%d/8' %i for i in range(9)])

fig.savefig(os.path.join(savepath, savefile + 'png'))
fig.savefig(os.path.join(savepath, savefile + 'pdf'))
fig.savefig(os.path.join(savepath, savefile + 'svg'))