import numpy as np
import scipy.optimize as op
import scipy as scipy
from sutil.base.Dataset import Dataset
from sutil.models.Model import Model

class RegularizedLogisticRegression(Model):

    def __init__(self, theta, alpha, l, **kwargs):
        self.theta = theta
        self.alpha = alpha
        self.l = l
        self.name = kwargs['name'] if 'name' in  kwargs.keys() else 'Logistic Regression'
        
    
    @classmethod
    def fromDataset(cls, data, alpha = 0.1, l=0.1):
        theta = np.random.random(data.X.shape[1] + 1)
        return cls(data, theta, alpha, l)
    
    @classmethod
    def fromDataFile(cls, datafile, delimeter, alpha=0.1, l=0.1):
        data = Dataset.fromDataFile(datafile, delimeter)
        theta = np.random.random(data.X.shape[1] + 1)
        return cls(data, theta, alpha, l)

    #m denotes the number of examples
    #gradient indicates the gradient matrix
    #regularization is the regularization parameter in order to prevent the over fitting
    #cost is the cotst of the logistic regression funciton
    def getCostAndGradient(self, data, theta):
        gradient = np.zeros(np.size(theta))
        m = data.m
        h_theta = self.computePredictions(data, theta).reshape(m, 1)
        regularization = (self.l * np.sum(theta[1 :]**2))/(2*m)
        
        CK = 0.000009
        h_theta[h_theta == 0] = CK
        h_theta[h_theta == 1] = 1 - CK
        
        cost_m = data.y * np.log(h_theta) + (1 - data.y) * np.log(1 - h_theta)
        
        cost = -1 * np.sum(cost_m)/m + regularization

        #We calculate the gradient of the first regression parameter
        differences = h_theta - data.y
        X = data.getBiasedX()
        gradient[0] = np.sum(differences * X[:, 0].reshape(m, 1))/m
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * X[:, j].reshape(m, 1))/m - self.l/m * theta[j]
        return cost, gradient

    def getCostAndGradient2(self, data, theta):
        m = data.m
        gradient = np.zeros(np.size(theta))
        h_theta = self.computePredictions(data, theta).reshape(m, 1)
        regularization = (self.l * np.sum(theta[1 :]**2))/(2*m)
        cost_m = self.train.y * np.log(h_theta) + (1 - data.y) * np.log(1 - h_theta)
        cost = -1 * np.sum(cost_m)/m + regularization

        #We calculate the gradient of the first regression parameter
        differences = h_theta - data.y
        X = data.getBiasedX()
        gradient[0] = np.sum(differences * X[:, 0].reshape(m, 1))/m
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * X[:, j].reshape(m, 1))/m - self.l/m * theta[j]
        return cost, gradient
    
    @staticmethod
    def getCost(theta, x, y, l=0):
        m = len(x)
        sigmoid = lambda z: 1/ (1 + np.exp(-1 * z))
        #sigmoid = lambda z: scipy.special.expit(z)
        h_theta = sigmoid(np.matmul(x, theta)).reshape(m, 1)
        regularization = (l * np.sum(theta[1 :]**2))/(2*m)
        cost_m = y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta)
        cost = -1 * np.sum(cost_m)/m + regularization
        return cost
    
    @staticmethod
    def getGradient(theta, x, y, l=0):
        gradient = np.zeros(np.size(theta))
        m = len(x)
        sigmoid = lambda z: 1/ (1 + np.exp(-1 * z))
        #input("Wait a second...")
        h_theta = sigmoid(np.matmul(x, theta)).reshape(m, 1)
        differences = h_theta - y
        gradient[0] = np.sum(differences * x[:, 0].reshape(m, 1))/m
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * x[:, j].reshape(m, 1))/m - l/m * theta[j]
        return gradient

    def computePredictions(self, data, theta):
        return self.sigmoid(np.matmul(data.getBiasedX(), theta))

    def evaluateHypotesis(self, xi, theta):
        return self.sigmoid(np.matmul(xi, theta))
                   
    def sigmoid(self, z):
        return scipy.special.expit(z)
        
    def gradienDescent(self, data, iterations):
        theta = self.theta
        cost = []
        gradient = []
        for i in range(iterations):
          costj, gradientj = self.getCostAndGradient(data, theta)
          cost.append(costj)
          gradient.append(gradientj)
          theta[0] -= self.alpha * gradientj[0]
          for j in range(1, len(theta)):
              theta[j] -= self.alpha * (gradientj[j] + (self.l/data.m) * theta[j])
        self.theta = theta
        return self.theta, cost, gradient

    def optimizedGradientDescent(self, data):
        initial_theta = np.zeros(data.n + 1)
        result = op.minimize(fun = RegularizedLogisticRegression.getCost, 
                             x0 = initial_theta, 
                             args = (data.getBiasedX(), data.y),
                             method = 'TNC',
                             jac = RegularizedLogisticRegression.getGradient)
        return result

    def makePrediction(self, x):
        evaluation = self.evaluateHypotesis(x, self.theta)
        pred = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        return pred(evaluation)
    
    def predict(self, X):
        return self.makePrediction(X)
    
    def trainModel(self, data, epochs = 100):
        result = self.optimizedGradientDescent(data)
        self.theta = result.x
        return self.theta