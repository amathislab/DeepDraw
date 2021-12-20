# Kai Sandbrink
# 2021-12-14

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import format_axis

cka_df_filepaths = [
    '/media/data/DeepDraw/revisions/analysis-data/exp308/analysis/regression_task/cka_df.csv',
    '/media/data/DeepDraw/revisions/analysis-data/exp309/analysis/regression_task/cka_df.csv',
    '/media/data/DeepDraw/revisions/analysis-data/exp310/analysis/regression_task/cka_df.csv'
]

savepath = '/media/data/DeepDraw/revisions/analysis-data/oneoff'

cka_dfs = [pd.read_csv(filepath, header=[0,1], index_col=0) for filepath in cka_df_filepaths]

#print(cka_dfs[0])

modelnames = ['S', 'ST', 'LSTM']
colors = ['C0', 'C2', 'C4']
#offsets = [-0.2, 0, 0.2]

max_nlayers = 8*6*5*3

fig = plt.figure(figsize=(14,6), dpi=200)
ax = fig.add_subplot(111)
#ax.set_xlim((-1, 9))

ax.set_xlabel("Network Depth")
ax.set_ylabel("CKA Score")
#ax.set_xticklabels([''] + [i/8 for i in range(9)])

def adjust_by_depth(layer_cka_scores, nlayers):
    #print(layer_cka_scores)
    #print(nlayers)
    #print(nlayers)
    layer_cka_depth_scores = np.empty((max_nlayers + 1,))
    layer_cka_depth_scores[:] = np.nan

    for i in range(nlayers+1):
        layer_cka_depth_scores[int(i*max_nlayers/nlayers)] = layer_cka_scores[i]
        #print("Position %d / %d " %(i*max_nlayers/nlayers, max_nlayers))

    return pd.Series(layer_cka_depth_scores)

#for cka_df, color, offset in zip(cka_dfs, colors, offsets):
for cka_df, color, modelname in zip(cka_dfs, colors, modelnames):
    nlayers = cka_df['nlayers','nlayers']
    cka_scores = cka_df['cka_scores']

    cka_depth_scores = cka_df.apply(lambda x : adjust_by_depth(x['cka_scores'].values, int(x['nlayers', 'nlayers'])), axis=1)
    print(cka_depth_scores)

    cka_means = cka_depth_scores.apply(np.nanmean)
    cka_err = cka_depth_scores.apply(np.nanstd)

    poss = np.arange(max_nlayers+1)

    mask = ~np.isnan(cka_means)
    #print(mask)
    #print(type(mask))

    ax.errorbar(poss[mask], cka_means[mask], cka_err[mask], marker="D", capsize=3.0, label=modelname, color=color)

#ax.legend(['S','ST','LSTM'])
ax.legend()
format_axis(ax)
#ax.set_xticklabels([0] + [i/max_nlayers for i in range(max_nlayers+1)])

ticks = [i/8*max_nlayers for i in range(9)]
ax.set_xticks(ticks)
ax.set_xticklabels(['%d/8' %i for i in range(9)])

fig.savefig(os.path.join(savepath, 'cka_depth_comp_plot.png'))
fig.savefig(os.path.join(savepath, 'cka_depth_comp_plot.pdf'))
fig.savefig(os.path.join(savepath, 'cka_depth_comp_plot.svg'))