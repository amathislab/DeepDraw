# Kai Sandbrink
# 2023-01-27
# This script manually imports and reads some of the velocities

# %% LIBRARY IMPORT

import numpy as np

# %% PARAMETERS

art_ee_file = '/media/data/DeepDraw/revisions/analysis-data/exp402/results/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_4_8-16-16-32_32-32-64-64_7293_1/horall/ee/l0_ee_mets_std_horall_test.npy'
art_vel_file = '/media/data/DeepDraw/revisions/analysis-data/exp402/results/spatial_temporal_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_4_8-16-16-32_32-32-64-64_7293_1/horall/vel/l0_vel_mets_std_horall_test.npy'
#tdt_file = '/media/data/DeepDraw/revisions/analysis-data/exp402/results/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_1/horall/vel/l5_vel_mets_std_horall_test.npy'
#tdt_pv_file = '/media/data/DeepDraw/revisions/analysis-data/exp406/results/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293/spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293_1/horall/vel/l5_vel_mets_std_horall_test.npy'

# %% LOAD IN

#tdt = np.load(tdt_file)
#tdt_pv = np.load(tdt_pv_file)

art_ee = np.load(art_ee_file)
eeevals = art_ee[...,0,1]

art_vel = np.load(art_vel_file)

direvals = art_vel[...,1,1] #dir
velevals = art_vel[...,2,1] #vel
dirvelevals = art_vel[...,3,1] #dir + vel
# %% PRINT

print("MUSCLE-LENGTH SPINDLES")

print(np.median(eeevals[:,0]))
print(np.median(direvals[:,0]))
print(np.median(velevals[:,0]))
print(np.median(dirvelevals[:,0]))


print("MUSCLE-CHANGES SPINDLES")

print(np.median(eeevals[:,1]))
print(np.median(direvals[:,1]))
print(np.median(velevals[:,1]))
print(np.median(dirvelevals[:,1]))

# %%
