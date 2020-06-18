"""
Script to modify the true UCI Machine Learning Repository's character trajectories dataset to
generate the base dataset for the Proprioceptive Character Recognition task.

The original dataset is available at : https://archive.ics.uci.edu/ml/datasets/Character+Trajectories

"""

import os
import numpy as np
import scipy.io as sio
import h5py
import copy

PATH_TO_DATA = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/data'

def trim_trajectory(traj):
    """Trim trajectory to only contain data where pen velocity is non-zero."""
    norms = np.linalg.norm(traj, axis=0)
    bool_id = np.array(norms != 0, dtype=bool)
    return traj[:, bool_id]

def upscale(traj):
    """Scale the trajectories to fit in a 10x10 cm2 box."""
    xmax = np.max(traj[0,:]); ymax = np.max(traj[1,:])
    xmin = np.min(traj[0,:]); ymin = np.min(traj[1,:])
    new_traj = np.zeros_like(traj)
    new_traj[0,:] = ((traj[0,:]-xmin)/(xmax-xmin) - 0.5)*10
    new_traj[1,:] = ((traj[1,:]-ymin)/(ymax-ymin) - 0.5)*10
    return new_traj

def pad_nan(traj, max_length):
    """Pad nans to character trajectories that take less time."""
    new_traj = np.zeros((2, max_length))
    new_traj[:, :traj.shape[1]] = traj
    new_traj[:, traj.shape[1]:] = np.nan
    return new_traj

def main():
    # Load original UCI ML dataset
    PATH_TO_DATA = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/data'
    data_path = os.path.join(PATH_TO_DATA, 'uci_dataset.mat')
    original_dataset = sio.loadmat(data_path)

    mytrajectories = copy.copy(original_dataset['mixout'][0]) # Contains velocities and pen-tip forces
    mylabels = original_dataset['consts'][0][0][4][0] 

    # Successively discard pen-tip forces, trim trajectory, retrieve XY positions and scale
    mytrajectories = [trim_trajectory(traj[:2]) for traj in mytrajectories]
    mytrajectories = [np.cumsum(traj, axis=1) for traj in mytrajectories]
    mytrajectories = [upscale(traj) for traj in mytrajectories]

    # Pad nans to character trajectories that take less time to write
    times = np.array([traj.shape[1] for traj in mytrajectories])
    max_length = max(times)
    mytrajectories = [pad_nan(traj, max_length) for traj in mytrajectories]

    # Save as .hdf5
    new_dataset = h5py.File(os.path.join(PATH_TO_DATA, 'pcr_trajectories.hdf5'), 'w')
    new_dataset.create_dataset('trajectories', data=mytrajectories)
    new_dataset.create_dataset('labels', data=mylabels)
    new_dataset.close()

    return

if __name__=='__main__':
    main()
