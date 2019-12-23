# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from sutil.base.Dataset import Dataset
from sutil.models.RegularizedLinearRegression import RegularizedLinearRegression
import matplotlib.pyplot as pyplot
#Load the model
datafile = './sutil/datasets/ex1data1.txt'
d = Dataset.fromDataFile(datafile, ',')
d.plotDataRegression('example')
#Some gradient descent settings
iterations = 1500
theta = np.zeros((2, 1))
alpha = 0.01
l = 0

print('Testing the cost function ...')
rlf = RegularizedLinearRegression(theta, alpha, l)
#compute and display initial cost
J, gradient = rlf.getCostAndGradient(d, theta)
print('With theta = [0 ; 0]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 32.07\n')

#further testing of the cost function
J, gradient = rlf.getCostAndGradient(d, np.array([-1, 2]).T)
print('\nWith theta = [-1 ; 2]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 54.24\n')

print('Program paused. Press enter to continue.\n')
input("Press enter to continue...")

print('\nRunning Gradient Descent ...\n')
theta, cost, gradient = rlf.gradienDescent(d, iterations)

print('Theta found by gradient descent:\n')
print(theta);
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')


result = rlf.optimizedGradientDescent(d)
print('Theta found by optimization descent:\n')
print(result.x);
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

input("Press enter to continue...")
#Plot the linear fit
fig, ax = d.getPlotRegression()
ax.plot(d.X, np.matmul(d.getBiasedX(), theta), '-')
#fig.legend('Training data', 'Linear regression')
pyplot.show()

#Predict values for population sizes of 35,000 and 70,000
predict1 = np.matmul([1, 3.5], theta)
print('For population = 35,000, we predict a profit of ', predict1*10000, '\n')
predict2 = np.matmul([1, 7], theta)
print('For population = 70,000, we predict a profit of ',predict2*10000, '\n')

print('Program paused. Press enter to continue.\n')
input("Press enter to continue...")

#============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

#Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(theta0_vals, theta1_vals)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

#Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
    	t = np.array([theta0_vals[i], theta1_vals[j]]).T
    	J_vals[i][j], gradient = rlf.getCostAndGradient(d, t);

print(theta0_vals.shape)
print(theta1_vals.shape)
print(J_vals.shape)
print(J_vals)

input("Press enter to continue...")
#% Because of the way meshgrids work in the surf command, we need to
#% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T;
#% Surface plot
fig2 = pyplot.figure()
#ax = Axes3D(fig2)
ax = fig2.gca(projection='3d')
surf = ax.plot_surface(X, Y, J_vals, cmap=cm.winter, linewidth=1, antialiased=True)
#fig.xlabel('theta_0')
#fig.ylabel('theta_1')

#% Contour plot
pyplot.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
pyplot.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
pyplot.xlabel('theta_0')
pyplot.ylabel('theta_1')
#hold on;
pyplot.plot(theta[0], theta[1], color='red', marker='x', linestyle='dashed', linewidth=2, markersize=12)
pyplot.show()
