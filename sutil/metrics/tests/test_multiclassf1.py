#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 23:03:17 2020

@author: mcampos
"""
import numpy as np
from sutil.metrics.MultiClassF1 import MultiClassF1

predictions = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
values = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
classes = np.unique(values)
print("="*20 + "Test 1: perfect score" + "="*20)
print(MultiClassF1(predictions, values, classes).scores)
print(MultiClassF1(predictions, values, classes).average())



predictions2 = np.array([1, 1, 2, 1, 2, 2, 2, 2, 3, 3, 3, 3])
print("="*20 + "Test 2: one mistake" + "="*20)
print(MultiClassF1(predictions2, values).scores)
print(MultiClassF1(predictions2, values).average())