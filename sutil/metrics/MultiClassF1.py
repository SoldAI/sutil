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
            precision = tp/(tp + fp) if tp + fp > 0 else 0
            recall = tp/(tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision*recall)/(precision+recall) if precision + recall > 0 else 0
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

        if num_classes == 0:
            return {'avg_precision': 0, 
                'avg_recall': 0, 
                'avg_f1': 0}
                
        return {'avg_precision': precision_total/num_classes, 
                'avg_recall': recall_total/num_classes, 
                'avg_f1': f1_total/num_classes}

    def macro_average(self):
        return self.average()

    def micro_average(self):
        tp, tn, fp, fn = 0, 0, 0, 0
        num_classes = 0
        for c in self.classes:
            if c in self.scores:
                num_classes += 1
                tp += self.scores[c]['tp']
                tn += self.scores[c]['tn']
                fp += self.scores[c]['fp']
                fn += self.scores[c]['fn']

        if num_classes == 0:
            return {'avg_precision': 0, 
                'avg_recall': 0, 
                'avg_f1': 0}

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return {'avg_precision': precision, 
                'avg_recall': recall, 
                'avg_f1': f1}

    def report(self):
        if not self.scores:
            self.calculateF1()

        print("{:>8} {:>8} {:>8} {:>8}".format('Class', 'Precision', 'Recall', 'F1'))

        for c, m in self.scores:
            print("{:n} {:2f} {:2f} {:2f}".format(c, m['precision'], m['recall'], m['f1']))
            print("\n")
            avg = self.macro_average()
            print("{:n} {:2f} {:2f} {:2f}".format("Macro Avg", avg['avg_precision'], avg['avg_recall'], avg['avg_f1']))
            avg = self.micro_average()
            print("{:n} {:2f} {:2f} {:2f}".format("Micro Avg", avg['avg_precision'], avg['avg_recall'], avg['avg_f1']))
