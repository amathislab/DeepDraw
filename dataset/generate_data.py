'''Scipt to generate the dataset for `Proprioceptive Character Recognition` task.'''

import os
import random
from collections import namedtuple
import argparse
import pickle
import copy

os.system('sudo pip install h5py')
import h5py

import numpy as np
from scipy.interpolate import interp1d
import opensim as osim
from pcr_data_utils import make_joint_config, make_muscle_config, compute_jerk

PATH_TO_DATA = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/data'
PATH_TO_OSIM_MODEL = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/data/dynamic_arm_model/'


def resize(trajectory, size):
    '''Resize the pen-tip trajectory, keeping the velocity profile constant.'''
    true_velocity = np.hstack((np.array([0, 0])[:, None], np.diff(trajectory, axis=1)))
    true_timestamps = np.arange(trajectory.shape[1])
    n_timestamps_new = int(true_timestamps.size*size)
    new_timestamps = np.linspace(0, true_timestamps[-1], n_timestamps_new)

    vel_func = interp1d(true_timestamps, true_velocity)
    new_velocity = vel_func(new_timestamps)
    new_traj = np.cumsum(new_velocity, axis=1) + trajectory[:, 0][:, None]
    return new_traj


def apply_rotations(trajectory, rot, shear_x, shear_y):
    aff = np.array([[1, np.tan(shear_x)], [np.tan(shear_y), 1]])
    aff = aff.dot(np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]))
    return aff.dot(trajectory)


def speedify(trajectory, speed):
    true_timestamps = np.arange(trajectory.shape[1])
    n_timestamps_new = int(true_timestamps.size/speed)
    new_timestamps = np.linspace(0, true_timestamps[-1], n_timestamps_new)

    func = interp1d(true_timestamps, trajectory)
    return func(new_timestamps)


def sample_latent_vars():
    '''Sample all latent transformations to be applied to the given trajectory.

    Returns
    -------
    latents: tuple, (size, rot, shear_x, shear_y, speed, noise)

    '''
    size_set = [0.7, 1., 1.3]
    rot_set = [-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6]
    speed_set = [0.8, 1., 1.2, 1.4]
    noise_set = [0, 0.1, 0.3]

    latents = (
        random.choice(size_set),
        *np.random.choice(rot_set, size=3),
        random.choice(speed_set),
        random.choice(noise_set))

    return latents


def main(args):
    '''Generate label specific joint angle and muscle length trajectories.

    '''
    # Load character trajectories and starting point data
    with h5py.File(os.path.join(PATH_TO_DATA, 'pcr_trajectories.hdf5'), 'r') as myfile:
        trajectories = myfile['trajectories'][()]
        labels = myfile['labels'][()]

    with h5py.File(os.path.join(PATH_TO_DATA, 'pcr_startingpoints.hdf5'), 'r') as myfile:
        startpts = myfile[args.plane][()]

    # Aligning pen-tip trajectories to {W}
    if args.plane == 'horizontal':
        plane_to_world = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    elif args.plane == 'vertical':
        plane_to_world = np.array([[0, 0, 0], [-1, 0, 0], [0, 1, 0]])

    char_idx = labels == args.label
    char_trajectories = trajectories[char_idx]

    char_data = []
    model = osim.Model(os.path.join(PATH_TO_OSIM_MODEL, 'MoBL_ARMS_module5_scaleIK.osim'))

    for traj in char_trajectories:
        traj = traj[:, np.all(~np.isnan(traj), axis=0)]

        tracing_error = 1e16
        joint_jerk = 1e16
        while tracing_error > 1e-2 and joint_jerk > 1:
            latent_vars = Latents(*sample_latent_vars())
            mytraj = resize(traj, latent_vars.size)
            mytraj = apply_rotations(mytraj, latent_vars.rot, latent_vars.shear_x, latent_vars.shear_y)
            mytraj = speedify(mytraj, latent_vars.speed)
            mytraj = np.insert(mytraj, 2, 0, axis=0)

            endeffector_coordinates = plane_to_world.dot(mytraj)
            startingpoint = random.choice(startpts)
            endeffector_coordinates += startingpoint[:, None]

            joint_coordinates, tracing_error = make_joint_config(endeffector_coordinates)
            joint_jerk = compute_jerk(joint_coordinates)

            muscle_coordinates = make_muscle_config(model, joint_coordinates)
            muscle_jerk = compute_jerk(muscle_coordinates)

            # Save the datapoint
            datapoint = {
                'label': args.label,
                'plane': args.plane,
                'startpt': startingpoint,
                'endeffector_coords': endeffector_coordinates,
                'joint_coords': joint_coordinates,
                'muscle_coords': muscle_coordinates,
                'muscle_jerk': muscle_jerk,
                'latents': latent_vars}

            char_data.append(copy.copy(datapoint))

    pickle.dump(char_data, open(os.path.join(PATH_TO_DATA, 'unprocessed_data/{}.p'.format(args.name)), 'wb'), protocol=-1)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Proprioceptive Character Recognition dataset')
    parser.add_argument('--label', type=int, help='Character label')
    parser.add_argument('--plane', type=str, help='Plane of writing {horizontal, vertical}')
    parser.add_argument('name', type=int, help='Job id for generating dataset')
    Latents = namedtuple('Latents', ('size', 'rot', 'shear_x', 'shear_y', 'speed', 'noise'))
    main(parser.parse_args())
