"""Scipt to generate candidate starting points in both vertical and horizontal planes for writing the
characters.

"""
import os
import itertools
import h5py
import numpy as np
from joblib import delayed, Parallel

from pcr_data_utils import Arm


PATH_TO_DATA = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/data'

def isreachable_point(target_xyz):
    myarm = Arm()
    shoulder_to_world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    target_xyz = (shoulder_to_world.T).dot(np.array(target_xyz)[:, None]).T # Same point in {S}
    angles = myarm.inv_kin(target_xyz)
    return np.linalg.norm(target_xyz - myarm.get_xyz(q=angles)[0])


def coarse_startpt_search(workspace, size, grid, grid_resolution, dim):
    '''Run a coarse grained search over the workspace to find candidate starting points, that is,
    the set of all reachable points for which [NE, NW, SE, SW] corners of a (size x size) square
    are also reachable.

    Arguments
    ---------
    workspace: np.ndarray, [N, 3] array of all reachable points of the Arm
    size: float, size of the bounding box for all characters [usually, 20x20 here]
    grid: np.array, levels of the grid used in generating the workspace
    grid_resolution: int, resolution used for generating the workspace
    dim: int, 2 if horizontal trajectory [z-plane], 0 if vertical [x-plane]

    Returns
    -------
    candidate_startpts: np.ndarray, candidate starting points after coarse search

    '''
    candidates = []
    for plane in grid:
        plane_mask = np.array((workspace[:, dim] == plane))
        candidates_inplane = workspace[plane_mask]
        candidates.append(search_inplane(candidates_inplane, size, grid_resolution, dim))
    return np.concatenate(candidates)


def search_inplane(candidates, size, grid_resolution, dim):
    temp_mask = np.array([True, True, False]) if dim == 2 else np.array([False, True, True])
    planar_workspace = candidates[:, temp_mask].tolist()
    neighbors = [generate_neighbors(pt, size, grid_resolution) for pt in planar_workspace]
    positives = []
    for pt in range(len(planar_workspace)):
        bools = [corner.tolist() in planar_workspace for corner in neighbors[pt]]
        positives.append(all(bools))
    valid_candidates = candidates[positives]
    return valid_candidates


def generate_neighbors(point, size, grid_resolution):
    point = np.array(point)
    n = grid_resolution*np.ceil(size/(2*grid_resolution))
    neast = point + [n, n]
    nwest = point + [-n, n]
    seast = point + [n, -n]
    swest = point + [-n, -n]
    return [neast, nwest, seast, swest]


def fine_startpt_search(candidates, dim):
    '''Run a fine-grained search over candidate starting points by checking reachability of a fine
    grid of points surrounding each candidate starting point.

    '''
    errors = []
    for start_point in candidates:
        grid = make_grid(start_point, dim)
        errors.append(isreachable_grid(grid))
    errors = np.array(errors)
    valid_candidates_idx = np.sum(errors < 1e-2, axis=1) == 225
    return candidates[valid_candidates_idx]


def make_grid(start_point, dim):
    '''Make a grid around a starting point, in either vertical or horizontal planes. As a somewhat
    arbitrary choice, the grid is 15x15.

    '''
    x, y, z = start_point
    if dim == 0:
        z, x, y = start_point
    xmin, xmax = x - 7, x + 8
    ymin, ymax = y - 7, y + 8
    X, Y = np.mgrid[xmin:xmax, ymin:ymax]
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    grid_points_3d = np.insert(grid_points, dim, z, axis=0)
    return grid_points_3d


def isreachable_grid(grid):
    errors = Parallel(n_jobs=24)(delayed(isreachable_point)(grid[:, i]) for i in range(grid.shape[1]))
    return errors


def main():
    # Define search space for the workspace
    grid_resolution = 3   # in cm
    xmax, xmin = 60, 0    # forward and backward
    ymax, ymin = 30, -60  # left and right
    zmax, zmin = 30, -60  # up and down

    X = np.linspace(xmin, xmax, (xmax - xmin)//grid_resolution + 1)
    Y = np.linspace(ymin, ymax, (ymax - ymin)//grid_resolution + 1)
    Z = np.linspace(zmin, zmax, (zmax - zmin)//grid_resolution + 1)

    # Compute whether each point in (X x Y x Z) is reachable
    errors = Parallel(n_jobs=24)(delayed(isreachable_point)(i) for i in itertools.product(X, Y, Z))

    points = np.array(list(itertools.product(X, Y, Z)))
    errors = np.array(errors)
    workspace = points[errors < 1e-2]

    # Run coarse-grained search for candidate starting points
    startpoints_horizontal = coarse_startpt_search(workspace, 20, Z, grid_resolution, 2)
    startpoints_vertical = coarse_startpt_search(workspace, 20, X, grid_resolution, 0)

    # Run fine-grained search for reduced set of candidate starting points
    startpoints_horizontal = fine_startpt_search(startpoints_horizontal, 2)
    startpoints_vertical = fine_startpt_search(startpoints_vertical, 0)

    with h5py.File(os.path.join(PATH_TO_DATA, 'pcr_startingpoints_test.hdf5'), 'w') as file:
        file.create_dataset('horizontal', data=startpoints_horizontal)
        file.create_dataset('vertical', data=startpoints_vertical)

    return

if __name__ == "__main__":
    main()
