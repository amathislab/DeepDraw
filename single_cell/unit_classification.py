# Kai Sandbrink
# 2021-12-14

import numpy as np
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def main(modelinfo, runinfo):

    r2threshold = 0.2 #threshold above which units are counted as tuned for a particular feature
    nlayers = modelinfo['nlayers']
    nlayers += 1
    t_stride = 2
    ntime=320
    metrics = ['RMSE', 'r2', 'PCC']
    nmetrics = len(metrics)
    fset = 'vel'
    mmod = 'std'
    tcoff = 32

    expf={
        'vel': runinfo.resultsfolder(modelinfo, 'vel'),
        'acc': runinfo.resultsfolder(modelinfo, 'acc'),
        'labels': runinfo.resultsfolder(modelinfo, 'labels')
    }
    # %% READ IN OF REGS AND THRESHOLD
    bes = []

    for ilayer in np.arange(0,nlayers):
        
        dvevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(expf['vel'], ilayer, 'vel', mmod, runinfo.planestring() ))        
        accevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(expf['acc'], ilayer, 'acc', mmod, runinfo.planestring()))
        labevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(expf['labels'], ilayer, 'labels', mmod, runinfo.planestring()))
        
        modevals = np.zeros(dvevals.shape[:len(dvevals.shape) - 2] + tuple([4]))
        modevals[...,0] = dvevals[...,1,1] #dir
        modevals[...,1] = dvevals[...,2,1] #vel
        modevals[...,2] = accevals[...,2,1] #acc
        modevals[...,3] = labevals[...,0,0] #labels
        
        modevals = np.where(modevals > r2threshold, True, False)
        
        bes.append(modevals)

    # %% CLASSIFICATION
    groupnames = ['Dir', 'Vel', 'Acc', 'Dir ∧ Vel', 'Dir ∧ Acc', 'Vel ∧ Acc', 'Dir ∧ Vel ∧ Acc', 'Labels', 'Total']

    ncats = len(groupnames) - 1
    cats = np.zeros((nlayers, ncats + 1))
    for ilayer, be in enumerate(bes):
        #print("Layer %d" %ilayer)
        #be = be.squeeze()
        #print(be.shape)
        cats[ilayer, ncats] = be[...,0].size
        for irow, _ in np.ndenumerate(be[...,0]):
            b = np.squeeze(be[irow])
            #print(b.shape)
            if(sum(b[:3]) == sum(b)):
                if sum(b[:3]) == 1: #We evaluate the labels tuning separately
                    ###ATTENTION! THIS WILL NOT WORK ACCURATELY IF CERTAIN LAYERS ARE TUNED FOR BOTH LABELS AND KIN FEATURES
                    if(b[0] == 1):
                        cats[ilayer, 0] +=1
                    elif(b[1] == 1):
                        cats[ilayer, 1] +=1
                    elif(b[2] == 1):
                        cats[ilayer, 2] +=1
                elif sum(b[:3]) == 2:
                    if(b[0] == 0):
                        cats[ilayer, 5] +=1
                    elif(b[1] == 0):
                        cats[ilayer, 4] +=1
                    elif(b[2] == 0):
                        cats[ilayer, 3] +=1 
                elif sum(b[:3]) == 3:
                    cats[ilayer, 6] += 1
            else:
                print("Warning! Label Tuning")
                if(b[2] == 1):
                    print("Warning! Also Acceleration tuned. Neuron double counted.")
                    
                cats[ilayer, 7] +=1

    #create modfractions
    modfractions = (cats[:,:8].swapaxes(0,1) / cats[:,8]).swapaxes(0,1)

    # %% OUTPUT TABLE
    print(groupnames)
    print(cats)

    # %% ONLY DO S MODEL
    #nmods = 1
    nmods = ncats

    '''
    def get_mtypecolorbar(mtype):
        """returns the needed colorbar for the given model type
        
        Arguments
        ---------
        mtype : str, one of ['Spatial-Temporal', 'Spatiotemporal']
        
        """
        if mtype=='Spatial-Temporal':
            cmap = matplotlib.cm.get_cmap('Blues_r') #spatial_temporal
        if mtype=='Spatiotemporal':
            cmap = matplotlib.cm.get_cmap('Greens_r') #spatiotemporal
        return cmap
    '''

    params = {
    'axes.labelsize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 14,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [8,8 ],
    'font.size': 18,
    'axes.titlepad': 20,
    'ytick.major.size': 4.5,
    'ytick.major.width': 1,
    'ytick.minor.size': 3.5
    }

    plt.rcParams.update(params)

    mtype = modfractions
    nmtypes = 1

    im = 0

    # %% PLOT


    space = 0.75
    width = 0.6
    lspace = space*nmtypes + 0.5

    fig = plt.figure(figsize=(10, 5.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)  #allows turning off spines... i.e. top and right is good :)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', length = 4.5, width=1)
    ax.get_yaxis().tick_left()

    #ax.tick_params(axis='y', length=0)
    # offset the spines
    #for spine in ax.spines.values():
    #  spine.set_position(('outward', 5))
    # put the grid behind
    ax.set_axisbelow(True)

    bars = []
    ct = 0.4

    #patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ( '\\', 'O', '-', '+','','x','o', '.')

    #for im, mtype in enumerate(modfractions):
    #    positions=[ilayer*int((nlayers[0]-1)/(nlayers[im]-1))*lspace+space*im for ilayer in range(nlayers[im])]
    positions=[ilayer*lspace+space*im for ilayer in range(nlayers)]

    #cmap = get_mtypecolorbar(modelinfo['shortbase'])
    cmap = matplotlib.cm.get_cmap(modelinfo['cmap'])
    for icat in range(ncats):
        #print(im, icat, nlayers[im])
        #print(mtype)
        #print(len(positions))
        bars.append(plt.bar(positions, mtype[:,icat] ,\
                bottom = mtype[:,:icat].sum(axis=1), width=width,
                color = cmap(icat*(1-ct)/(nmods-1)),
                align='edge', hatch=patterns[icat]
                ))
        
    #or ibar, bar in enumerate(bars):
    #   bar.set_hatch(patterns[ibar])

    plt.title('Unit Classification')
    #plt.xticks(np.arange(0,nlayers*lspace,lspace),
    #        ['Spindles'] + ['Layer %d' %i for i in np.arange(1,nlayers+1)],
    #        horizontalalignment='center', rotation=45)
    plt.ylabel('Fraction of Neurons')
    #plt.ylim(-0.1, 1.1)
    plt.ylim(0,1)

    #Creating the legend
    #modelpatches = [mpatches.Patch(color=get_mtypecolorbar(modelinfo['shortbase'])(1), label=mtype)]
    modelpatches = [mpatches.Patch(color=matplotlib.cm.get_cmap(modelinfo['cmap'])(1), label=mtype)]
    #modellegend = plt.legend(modelpatches, modelnames, loc=(0.78,0.842425))
    #ax.add_artist(modellegend)
    plt.legend(bars, groupnames[:ncats], loc=1)

    plt.tight_layout()

    ff = os.path.join(runinfo.analysisfolder(modelinfo), 'unit_classification')
    os.makedirs(ff, exist_ok=True)

    fig.savefig(os.path.join(ff, 'unit_classification.pdf'))
    fig.savefig(os.path.join(ff, 'unit_classification.png'))
    fig.savefig(os.path.join(ff, 'unit_classification.svg'))