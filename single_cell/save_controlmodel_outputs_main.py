#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:06:05 2019

@author: kai
"""

from savelouts_best_controls import main

mnames = [modelname = 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272',
    'temporal_spatial_4_16-16-32-64_64-64-64-64_5272',
    'spatiotemporal_4_8-8-32-64_7292'
    ]

# %% CALL UP MAIN

for name in mnames:
    for i in range(5):
        i = i + 1
        modelname = name + '_%d' %i
