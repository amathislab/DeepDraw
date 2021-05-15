#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 2021

@author: kai

Various utility functions for the DeepDraw pipeline
"""

import re

def sorted_alphanumeric(data):
    '''https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir/48030307#48030307'''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
