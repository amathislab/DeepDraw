# Kai Sandbrink
# 2022-03-05
# This script plots the bubble plots showing neuron fractions

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#layers = np.arange(9)

### ART FILES
# dataset_files = ['/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_4_8-16-16-32_32-32-64-64_7293_1/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_4_8-16-16-32_32-32-64-64_7293_1r/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp403/analysis/spatiotemporal_4_8-8-32-64_7272/spatiotemporal_4_8-8-32-64_7272_1/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp403/analysis/spatiotemporal_4_8-8-32-64_7272/spatiotemporal_4_8-8-32-64_7272_1r/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp405/analysis/lstm_3_8-16-16_256/lstm_3_8-16-16_256_1/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp405/analysis/lstm_3_8-16-16_256/lstm_3_8-16-16_256_1r/horall/unit_classification/unit_classification_dataset.pkl'
#     ]

# names = ['S', 'S_r', 'ST', 'ST_r', 'LSTM', 'LSTM_r']
# #colors = ['C0', 'grey', 'green', 'grey', 'C4', 'grey']
# colors = ['midnightblue', 'grey', 'green', 'grey', 'C4', 'grey']
# file_prefix = '20230217_art'

### TDT FILES

# dataset_files = ['/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_1/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp402/analysis/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_1r/horall/unit_classification/unit_classification_dataset.pkl',
#     '/media/data/DeepDraw/revisions/analysis-data/exp403/analysis/spatiotemporal_r_4_8-8-32-64_7272/spatiotemporal_r_4_8-8-32-64_7272_1/horall/unit_classification/unit_classification_dataset.pkl',
#     ]

# names = ['S', 'S_r', 'ST']
# colors = ['darkturquoise', 'grey', 'red']
# file_prefix = '20230217a_tdt'


### TDT_POSVEL FILES

dataset_files = ['/media/data/DeepDraw/revisions/analysis-data/exp406/analysis/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_1/horall/unit_classification/unit_classification_dataset.pkl',
    '/media/data/DeepDraw/revisions/analysis-data/exp407/analysis/spatiotemporal_r_4_8-8-32-64_7272/spatiotemporal_r_4_8-8-32-64_7272_1/horall/unit_classification/unit_classification_dataset.pkl',
    '/media/data/DeepDraw/revisions/analysis-data/exp408/analysis/lstm_r_3_8-16-16_256/lstm_r_3_8-16-16_256_1/horall/unit_classification/unit_classification_dataset.pkl'
    ]

names = ['S', 'ST', 'LSTM']
colors = ['darkturquoise', 'yellowgreen', 'orchid']
file_prefix = '20230218a_tdt_pv'

for dataset_file, name, color in zip(dataset_files, names, colors):

    print('currently on %s' %name)

    art = pd.read_pickle(dataset_file)
    art = art.divide(art['Total'], axis=0)

    art_max = art.loc[:, art.columns != 'Total'].max().max()
    print(art_max)

    nlayers =len(art)

    art['Layer'] = np.arange(len(art))
    art = pd.melt(art, "Layer", ["Dir.", "Vel.", "Dir. x Vel.", "Pos. Cart.", "Pos. Polar", "Acc.", "Labels"])

    #tdt = pd.read_csv("./unit_classification_dataset_S_TDT.csv")
    #tdt = tdt.divide(tdt['Total'], axis=0)
    #tdt['Layer'] = layers
    #tdt = pd.melt(tdt, "Layer", ["Dir.", "Vel.", "Dir. x Vel.", "Pos. Cart.", "Pos. Polar", "Acc.", "Labels"])

    plt.figure(figsize=[4, 3])
    sns.scatterplot(x='Layer', y='variable', size=[300]*len(art['Layer']), marker='+', data=art, sizes=(300, 300), color='grey', clip_on=False, legend=False, linewidth=0.3)
    sns.scatterplot(x='Layer', y='variable', size='value', data=art, sizes=(1, 300*art_max), color=color, clip_on=False, legend=False)
    plt.xticks(art['Layer'], ['Sp.'] + ['L%d'%i for i in art['Layer'][1:]], fontsize=7)
    plt.yticks(fontsize=9)
    
    ax = plt.gca() 
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # increase tick width
    ax.tick_params(width=2.5)

    sns.despine(trim=True)
    plt.tight_layout()

    plt.savefig("/media/data/DeepDraw/revisions/analysis-data/oneoff/bubbleplots/%s_%s.svg" %(file_prefix, name), transparent=True, dpi=300)
    plt.savefig("/media/data/DeepDraw/revisions/analysis-data/oneoff/bubbleplots/%s_%s.pdf" %(file_prefix , name), transparent=True, dpi=300)