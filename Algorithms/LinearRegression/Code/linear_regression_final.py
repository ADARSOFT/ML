import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})

#%% PREPARE DATA
data = pd.read_csv('house.csv')

# Output data (label)
y = data.iloc[:,[-1]]
# Input data (features)
X = data.iloc[:,0:-1]

# *Feature scaling* standardization method (mean for all features is 0 and variance is 1)
X_mean = X.mean()
X_std = X.std() 
X = (X - X_mean) / X_std
     
# Add intercept multiplier (add this after feature scaling)
X.insert(0, 'X0', 1)

# Convert input and output data to np.array (in future we will use to_numpy())
y = y.values
X = X.values

# Calculate input data shape
n,m = X.shape

#%% INITIALIZATION
# Return random floats in the half-open interval [0.0, 1.0) for Slope's. 
# We use array length as X number of columns (m).
theta = np.random.random((1,m))
# Define learning rate for gradient descent
learning_rate = 0.1
# Define maximum iteration number
max_iteration = 400
# Define gradient min norm
min_grad_norm = 0.01

#%% Create function with params 
def GD_LinearRegression(X, theta, y, learning_rate, max_iteration, min_grad_norm, n, lambda_penalty = 1):
	
	x_iteration = []
	x_mse_per_iteration = []
	
	for iter in range(max_iteration):
		# Predictions calculation.
	    # First step is .T -> Transpose, because first matrix have 6 columns, second must have 6 rows
	    # Second step is do 'dot product' in order to calculate predicted response variable by formula in below:
	    # Yi = b0 * 1 + b1 * Xi1 + b2 * Xi2 + ... + bK * XiK + E --> MULTIPLE LINEAR REGRESSION
		pred = np.dot(X, theta.T)
		
	    # Calculate errors per item (i), in next line we will calculate squered sum. RESIDUALS
		residuals = (pred - y)

		# *Ridge regression calculation*
		# SSR + (lambda * theta **2) #(Note: not penalize intercept)
		penalty_vector = calculateRidgeRegressionPenalty(theta, lambda_penalty)
		
		# Gradient descent calculation
		gradient_vector = (np.dot(residuals.T, X) + penalty_vector) / n
		
		# Calculate step_size by multiplying learning_rate and gradient_vector (gradient_vector => data descent to minimum).
		step_size = learning_rate * gradient_vector
		# Calculate new theta - Loss function parameters (old w - step_size) * learning rate
		theta = theta - step_size
		
		# Calculate sum of absolute gradient_vector values
		grad_norm = abs(gradient_vector).sum()
		
		# Cost function - Loss function MSE
		mean_square_error = np.dot(residuals.T, residuals) / n
		
		x_iteration.append(iter)
		x_mse_per_iteration.append(mean_square_error[0][0])
		
		if grad_norm < min_grad_norm or mean_square_error < 10: break
	
		print(iter, grad_norm, mean_square_error)

	showPlot(x_iteration, x_mse_per_iteration)
#%% Ridge regression calculation 
def calculateRidgeRegressionPenalty(theta, lambda_penalty):
	theta_squered = theta**2
	slope_squered = theta_squered[0][1:]
	intercept_squered = theta_squered[0][0:1]
	return np.concatenate([intercept_squered, lambda_penalty * slope_squered])

#%% Visualisation
def showPlot(x_iteration, x_mse_per_iteration):
	plt.plot(x_iteration, x_mse_per_iteration)
	plt.xlabel('No. of iterations')
	plt.ylabel('Mean squered error')
	
#%% Use linear regression with above function 
GD_LinearRegression(X, theta, y, learning_rate, max_iteration, min_grad_norm, n, 1)

#%% PREDICT
data_new = pd.read_csv('house_new.csv')
data_new = (data_new-X_mean)/X_std
data_new.insert(0, 'X0', 1)
prediction = data_new.values.dot(theta.T)
print(prediction)