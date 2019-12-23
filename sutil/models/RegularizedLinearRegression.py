import numpy as np
import scipy.optimize as op
import sutil.base.Dataset as Dataset
from sutil.models.Model import Model

class RegularizedLinearRegression(Model):

    def __init__(self, theta, alpha, l, **kwargs):
        self.theta = theta
        self.alpha = alpha
        self.l = l
    
    @classmethod
    def fromDataset(cls, data, alpha = 0.1, l=0.1):
        theta = np.rand(data.y.shape)
        return cls(data, theta, alpha, l)
    
    @classmethod
    def fromDataFile(cls, datafile, delimeter, alpha=0.1, l=0.1):
        data = Dataset.fromDataFile(datafile, delimeter)
        theta = np.rand(data.y.shape)
        return cls(data, theta, alpha, l)

    #m denotes the number of examples
    #gradient indicates the gradient matrix
    #regularization is the regularization parameter in order to prevent the over fitting
    #cost is the cotst of the logistic regression funciton
    def getCostAndGradient(self, data, theta):
        X = data.getBiasedX()
        m = data.m
        gradient = np.zeros(np.size(theta))
        h_theta = self.evaluateHypothesis(X, theta).reshape(m, 1)
        differences = h_theta - data.y
        cost_m = differences **2
        regularization = (self.l * np.sum(theta[1 :]**2))/(2*m)
        cost = np.sum(cost_m)/(2*m) + regularization        
        gradient[0] = np.sum(differences * X[:, 0].reshape(len(X), 1))/m
        #We calculate the gradient for the rest of the parameters
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * X[:, j].reshape(len(X), 1))/m + self.l/m * theta[j]
        return cost, gradient
    
    @staticmethod
    def getCost(theta, x, y, l=0):
        m = len(x)
        differences = np.matmul(x, theta).reshape(m, 1) - y
        cost_m = differences **2
        regularization = (l * np.sum(theta[1 :]**2))/(2*m)
        cost = np.sum(cost_m)/(2*m) + regularization
        return cost
    
    @staticmethod
    def getGradient(theta, x, y, l=0):
        gradient = np.zeros(np.size(theta))
        m = len(x)
        differences = np.matmul(x, theta).reshape(m, 1) - y
        gradient[0] = np.sum(differences * x[:, 0].reshape(m, 1))/m
        #We calculate the gradient for the rest of the parameters
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * x[:, j].reshape(m, 1))/m + (l/m) * theta[j]
        return gradient

    def gradienDescent(self, data, iterations):
        theta = self.theta
        cost = []
        gradient = []
        print("****************************Computing gradient descent***************************")
        print(iterations)
        for i in range(iterations):
          costj, gradientj = self.getCostAndGradient(data, theta)
          cost.append(costj)
          gradient.append(gradientj)
          for j in range(len(theta)):
              theta[j] -= self.alpha * gradientj[j]
        self.theta = theta
        return self.theta, cost, gradient
    
    @staticmethod
    def normalEquation(data):
        X = data.getBiasedX()
        theta = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, data.y))
        return theta

    def optimizedGradientDescent(self, data):
        initial_theta = np.zeros(data.n + 1)
        result = op.minimize(fun = RegularizedLinearRegression.getCost, 
                             x0 = initial_theta, 
                             args = (data.getBiasedX(), data.y),
                             method = 'TNC',
                             jac = RegularizedLinearRegression.getGradient)
        return result

    def evaluateHypothesis(self, xi, theta):
        return np.matmul(xi, theta)
    
    def makePrediction(self, x):
        return self.evaluateHypothesis(x, self.theta)
    
    def predict(self, X):
        return self.makePrediction(X)
    
    def trainModel(self, data, iterations = 100):
        return self.gradienDescent(data, iterations)