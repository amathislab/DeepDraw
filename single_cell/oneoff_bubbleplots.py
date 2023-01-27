# Kai Sandbrink
# 2022-03-05
# This script plots the bubble plots showing neuron fractions

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#layers = np.arange(9)

### ART FILES
dataset_files = ['/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_4_8-16-16-32_32-32-64-64_7293_1/horall/unit_classification/unit_classification_dataset.pkl',
    '/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_4_8-16-16-32_32-32-64-64_7293_1r/horall/unit_classification/unit_classification_dataset.pkl',
    '/media/data/DeepDraw/revisions/analysis-data/exp403/analysis/spatiotemporal_4_8-8-32-64_7272/spatiotemporal_4_8-8-32-64_7272_1/horall/unit_classification/unit_classification_dataset.pkl',
    '/media/data/DeepDraw/revisions/analysis-data/exp403/analysis/spatiotemporal_4_8-8-32-64_7272/spatiotemporal_4_8-8-32-64_7272_1r/horall/unit_classification/unit_classification_dataset.pkl',
    '/media/data/DeepDraw/revisions/analysis-data/exp405/analysis/lstm_3_8-16-16_256/lstm_3_8-16-16_256_1/horall/unit_classification/unit_classification_dataset.pkl',
    '/media/data/DeepDraw/revisions/analysis-data/exp405/analysis/lstm_3_8-16-16_256/lstm_3_8-16-16_256_1r/horall/unit_classification/unit_classification_dataset.pkl'
    ]


names = ['S', 'S_r', 'ST', 'ST_r', 'LSTM', 'LSTM_r']
colors = ['C0', 'grey', 'green', 'grey', 'C4', 'grey']


### TDT FILES

# dataset_files = ['/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_1/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_1r/horall/unit_classification/unit_classification_dataset.pkl',
#     ]

# names = ['S', 'S_r']
# colors = ['C1', 'grey']

for dataset_file, name, color in zip(dataset_files, names, colors):

    print('currently on %s' %name)

    art = pd.read_pickle(dataset_file)
    art = art.divide(art['Total'], axis=0)

    nlayers =len(art)

    art['Layer'] = np.arange(len(art))
    art = pd.melt(art, "Layer", ["Dir.", "Vel.", "Dir. x Vel.", "Pos. Cart.", "Pos. Polar", "Acc.", "Labels"])

    #tdt = pd.read_csv("./unit_classification_dataset_S_TDT.csv")
    #tdt = tdt.divide(tdt['Total'], axis=0)
    #tdt['Layer'] = layers
    #tdt = pd.melt(tdt, "Layer", ["Dir.", "Vel.", "Dir. x Vel.", "Pos. Cart.", "Pos. Polar", "Acc.", "Labels"])

    plt.figure(figsize=[4, 3])
    sns.scatterplot(x='Layer', y='variable', size='value', data=art, sizes=(1, 300), color=color, clip_on=False)
    plt.xticks(art['Layer'], ['Sp.'] + ['L%d'%i for i in art['Layer'][1:]], fontsize=7)
    plt.yticks(fontsize=9)
    sns.despine(trim=True)
    #plt.tight_layout()
    plt.savefig("/media/data/DeepDraw/revisions/analysis-data/oneoff/bubbleplots/art_%s.svg" %name, transparent=True)
    plt.savefig("/media/data/DeepDraw/revisions/analysis-data/oneoff/bubbleplots/art_%s.pdf" %name, transparent=True)