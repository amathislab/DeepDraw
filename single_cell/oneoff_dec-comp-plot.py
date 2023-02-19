# Kai Sandbrink
# 2022-01-31

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel
import os

# %% FUNCTION

def plotcomp_tr_reg_decoding(tcfdf, tcf, rmname, rc, nlayers):
    ''' Plot comparisons between for trained and control models for decoding
    
    Arguments
    ---------
    tcfdf : pd.DataFrame, index and columns matching output of compile_comparisons
    tcfs : list of strs, names of tuning features
    
    Returns
    -------
    
    '''

    x = np.arange(nlayers + 1)
    if nlayers == 4:
        x_to_plot = x*2
    else:
        x_to_plot = x
        
    trainednamer = lambda i: rmname + '_%d' %i
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    controlnamer = lambda i: rmname + '_%dr' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]  

    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    print(tcfdf)

    trainedvars1 = np.vstack([tcfdf.loc[(trainednames, i, tcf)] for i in np.arange(nlayers+1)]).T
    controlvars1 = np.vstack([tcfdf.loc[(controlnames, i, tcf)] for i in np.arange(nlayers+1)]).T

    masked_trained = np.where(trainedvars1 <= 1000, trainedvars1, np.nan)
    masked_controls = np.where(controlvars1 <= 1000, controlvars1, np.nan)
    print("Masked Trained:", masked_trained)
    print("Masked Trained:", masked_controls)

    trainedvars1_mean = np.nanmean(masked_trained, axis=0)
    errs_trainedvars1 = np.nanstd(masked_trained, axis=0)
    controlvars1_mean = np.nanmean(masked_controls, axis=0)
    errs_controlvars1 = np.nanstd(masked_controls, axis=0)

    suppressed_counts = np.sum(np.where(np.isnan(masked_trained), 1, 0), axis=0)
    print("Suppressed: Trained", suppressed_counts)

    suppressed_counts = np.sum(np.where(np.isnan(masked_controls), 1, 0), axis=0)
    print("Suppressed: Trained", suppressed_counts)

    for i, mname in enumerate(controlnames):
        #if np.sum(np.isnan(controlvars1[i]))==0:
        line_indtrainedvar1, = plt.plot(x_to_plot, masked_trained[i], color=rc, marker = 'D', alpha = 0.15)
        line_indcontrolsvar1, = plt.plot(x_to_plot, masked_controls[i], color=rc,  linestyle='dotted', marker = 'D', alpha = 0.15)
    
    line_meantrainedsvar1,_,_ = plt.errorbar(x_to_plot, trainedvars1_mean, yerr=errs_trainedvars1, color=rc, marker='D', capsize=3.0)
    line_meancontrolsvar1,_,_ = plt.errorbar(x_to_plot, controlvars1_mean, yerr=errs_controlvars1, linestyle='dotted', color=rc, marker='D', capsize=3.0)

    return line_meantrainedsvar1,line_indtrainedvar1,line_meancontrolsvar1, line_indcontrolsvar1

# %% INITALIZATIONS

decoding_kindiffs_files = [
    #'/media/data/DeepDraw/revisions/analysis-data/exp316/analysis/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/comparison/horall/decoding_kindiffs/spatial_temporal_4_8-16-16-32_32-32-64-64_7293_decoding_comparisons_tr_reg_df_a0.csv',
    #'/media/data/DeepDraw/revisions/analysis-data/exp318/analysis/spatiotemporal_4_8-8-32-64_7272/comparison/horall/decoding_kindiffs/spatiotemporal_4_8-8-32-64_7272_decoding_comparisons_tr_reg_df_a0.csv',
    #'/media/data/DeepDraw/revisions/analysis-data/exp319/analysis/lstm_3_8-16-16_256/comparison/horall/decoding_kindiffs/lstm_3_8-16-16_256_decoding_comparisons_tr_reg_df_a0.csv'
    '/media/data/DeepDraw/revisions/analysis-data/exp316/analysis/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/comparison/horall/decoding_kindiffs/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_decoding_comparisons_df_a0.csv',
    '/media/data/DeepDraw/revisions/analysis-data/exp318/analysis/spatiotemporal_r_4_8-8-32-64_7272/comparison/horall/decoding_kindiffs/spatiotemporal_r_4_8-8-32-64_7272_decoding_comparisons_df_a0.csv',
    '/media/data/DeepDraw/revisions/analysis-data/exp319/analysis/lstm_r_3_8-16-16_256/comparison/horall/decoding_kindiffs/lstm_r_3_8-16-16_256_decoding_comparisons_df_a0.csv',
]

dfs = [pd.read_csv(file, header=0, index_col=[0, 1, 2]) for file in decoding_kindiffs_files]

nlayers = 8
x = range(nlayers + 1)

modelnames_short = ['S', 'ST', 'LSTM']
modelnames_long = ['spatial_temporal_4_8-16-16-32_32-32-64-64_7293', 'spatiotemporal_4_8-8-32-64_7272', 'lstm_3_8-16-16_256']
rmodelnames_long = ['spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293', 'spatiotemporal_r_4_8-8-32-64_7272', 'lstm_r_3_8-16-16_256']
colors = ['C0', 'C2', 'C4']
rcolors = ['orange', 'red', 'yellow']

savepath = '/media/data/DeepDraw/revisions/analysis-data/oneoff'
savefile = 'dec_comp_plot.'

# %% INITIALIZE FIGURE

fig = plt.figure(figsize=(12,5.5), dpi=300)   
ax = fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_axisbelow(True)

handles_mean = []
handles_ind = []
handles_mean_r = handles_ind_r = []
#handles = []

for mname, rmname, color, rcolor, df, nl in zip(modelnames_long, rmodelnames_long, colors, rcolors, dfs, [8, 4, 4]):
    tcdf = df.loc[(slice(None), slice(None), 'ee_x'), 'PCC/dist']

    #print(df)
    #print(tcdf)

    '''
    print(mname)
    handle_mean, handle_ind = plotcomp_tr_reg_decoding(tcdf, 'ee_x', mname, color, nl)
    handles_mean.append(handle_mean)
    handles_ind.append(handle_ind)
    '''

    print(rmname)
    handle_mean, handle_ind, handle_mean_r, handle_ind_r = plotcomp_tr_reg_decoding(tcdf, 'ee_x', rmname, color, nl)
    handles_mean.append(handle_mean)
    handles_ind.append(handle_ind)
    handles_mean_r.append(handle_mean_r)
    handles_ind_r.append(handle_ind_r)

plt.ylabel('dist')
#plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
#           horizontalalignment = 'right')
plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,nlayers+1)])
#plt.ylim((-2,15))

#handles, _ = ax.get_legend_handles_labels()
#handles = np.array(handles)
#plt.legend(['%s trained' %tcf, '%s controls' %tcf])

#ax = plt.gca()
#format_axis(ax)
#handles, _ = ax.get_legend_handles_labels()
#handles = np.array(handles)

first_legend = plt.legend(handles=handles_mean, \
    labels=modelnames_short, loc="upper right")
ax.add_artist(first_legend)

#plt.legend(handles= [handles_ind[0], handles_mean[0]], labels=["Ind.", 'Mean'], loc='upper left')
plt.legend(handles= [handles_mean[0], handles_mean_r[0]], labels=["TDT Trained", 'Control'], loc='upper left')
#print("Handles: ", handles)
#plt.legend(handles[[0,1,10,11]], ['Ind. Recog.', 'Ind. Decod.', \
#            'Mean of Recog.', 'Mean of Decod.'])

plt.tight_layout()

fig.savefig(os.path.join(savepath, savefile + 'png'))
fig.savefig(os.path.join(savepath, savefile + 'pdf'))
fig.savefig(os.path.join(savepath, savefile + 'svg'))