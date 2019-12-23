# -*- coding: utf-8 -*-
import numpy as np
#from mpl_toolkits.mplot3d import axes3d, Axes3D
from sutil.base.Dataset import Dataset
from sutil.models.RegularizedLogisticRegression import RegularizedLogisticRegression
import matplotlib.pyplot as pyplot

def plotBoundary(theta, xlabel="X", ylabel="Y", title="Title", legend=['y = 1', 'y = 0', 'Decision boundary']):
    #Plot Boundary
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            mf = map_feature(np.array(u[i]), np.array(v[j]))
            print(mf)
            print(mf.shape)
            z[i, j] = np.matmul(theta, mf)
            #z[i, j] = (map_feature(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))
    z = z.T
    pyplot.contour(u, v, z)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.legend(legend)
    pyplot.show()

def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)
    return out

#Load the model
datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
d.xlabel = 'Exam 1 score'
d.ylabel = 'Exam 2 score'
d.legend = ['Admitted', 'Not admitted']
#Some gradient descent settings
iterations = 400
theta = np.zeros((d.n + 1, 1))
#alpha = 0.1
alpha = 0.03
l = 0

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
d.plotData()
input('\nProgram paused. Press enter to continue.\n')

#initialize theta
rlr = RegularizedLogisticRegression(theta, alpha, l, train=1)

#Compute and display initial cost and gradient
cost, grad = rlr.getCostAndGradient(d, theta);

print('Cost at initial theta (zeros): \n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

#Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = rlr.getCostAndGradient(d, test_theta.T)

print('\nCost at test theta: \n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('\nProgram paused. Press enter to continue.\n')

theta, cost, gradient = rlr.gradienDescent(d, 100)
print(theta)
fig, ax = pyplot.subplots()
ax.scatter(range(len(cost)), cost, marker='x', c='r')
ax.set(xlabel='Iteration', ylabel='Cost', title='Cost function')
ax.grid()
pyplot.show()
print("Cost found by gradient descent...")
print(cost[-1])
input("Wait...")
result = rlr.optimizedGradientDescent(d)
cost = result.fun
theta = result.x
#Print theta to screen
print('Cost at theta found by fminunc: \n', cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

#Plot Boundary
#plotBoundary(theta, 'Exam 1 score', 'Exam 2 score', ['Admitted', 'Not admitted'])

input('\nProgram paused. Press enter to continue.\n')

prob = rlr.sigmoid(np.matmul(theta, [1, 45, 85]))
print('For a student with scores 45 and 85, we predict an admission probability of \n', prob)
print('Expected value: 0.775mean +/- 0.002\n\n');

rlr.theta = theta
#Compute accuracy on our training set
p = rlr.makePrediction(d.getBiasedX())

sum = 0
for i in range(len(p)):
    if p[i] == d.y[i]:
        sum += 1

print('Train Accuracy: \n', sum/len(p) * 100)

#===========================
# Test model from zero
#===========================
datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
lr = RegularizedLogisticRegression(theta, alpha, l, train=1)
lr.trainModel(d)
lr.score(d.X, d.y)
lr.roc.plot()
lr.roc.zoom((0, 0.4),(0.5, 1.0))