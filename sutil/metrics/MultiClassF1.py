#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:47:44 2020

@author: mcampos
"""

# -*- coding: utf-8 -*-
import numpy as np

class MultiClassF1:
    def __init__(self, predictions, values, classes = None):
        self.predictions = predictions
        self.values = values
        self.classes = classes if classes is not None else np.unique(values)
        self.scores = {}
        self.calculateF1()
        
    
    def calculateF1(self):
        
        for c in self.classes:
            tp = np.sum((self.predictions == c) & (self.values == c))
            tn = np.sum((self.predictions != c) & (self.values != c))
            fp = np.sum((self.predictions == c) & (self.values != c))
            fn = np.sum((self.predictions != c) & (self.values == c))
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            f1 = 2 * (precision*recall)/(precision+recall)
            self.scores[c] = {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'precision': precision, 'recall': recall, 'f1': f1}
        return self.scores
    
    def average(self):
        precision_total, recall_total, f1_total = 0, 0, 0
        num_classes = 0
        for c in self.classes:
            if c in self.scores:
                num_classes += 1
                precision_total += self.scores[c]['precision']
                recall_total += self.scores[c]['recall']
                f1_total += self.scores[c]['f1']
                
        return {'avg_precision': precision_total/num_classes, 
                'avg_recall': recall_total/num_classes, 
                'avg_f1': f1_total/num_classes}