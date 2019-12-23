# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

class ModelROC:
    def __init__(self, model, X_test, y_test, legend = 'model'):
        self.model = model
        y_pred = self.getPredictions(X_test)
        fpr, tpr, treshold = roc_curve(y_test, y_pred)
        self.fpr = fpr
        self.tpr = tpr
        self.treshold = treshold
        self.auc = auc(fpr, tpr)
        self.legend = legend
        print("Area under the curve " + self.legend + " " + str(self.auc))
    
    def getPredictions(self, X_test):
        return self.model.predict(X_test)
    
    def plotData(self):
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self.fpr, self.tpr, label= self.legend + ' (area = {:.3f})'.format(self.auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        
    def plot(self):
        plt.figure(1)
        self.plotData()
        plt.show()
        
    def zoom(self, zoom_x, zoom_y):
        plt.figure(2)
        plt.xlim(zoom_x[0], zoom_x[1])
        plt.ylim(zoom_y[0], zoom_y[1])
        self.plotData()
        plt.show()
