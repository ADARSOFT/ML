import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})

#%% PREPARE DATA
data = pd.read_csv('boston.csv')

y = data.iloc[:,[-1]]
X = data.iloc[:,0:-1]

# (Feature scaling)
X_mean = X.mean()
X_std = X.std()
# Standardization method (mean for all features is 0)
X = (X - X_mean) / X_std 
     
# Add intercept multiplier (add this after Feature scaling)
X['X0'] = 1 

y = y.values
X = X.values

n,m = X.shape

#%% INITIALIZATION
# Return random floats in the half-open interval [0.0, 1.0) for Slope's. Use array length as X number of columns (m)
theta = np.random.random((1,m))
# Define learning rate for gradient descent
learning_rate = 0.8
# Define maximum iteration number
max_iteration = 1000
# Define gradient min norm
min_grad_norm = 0.01

#%% Create function with params 
def GD_LinearRegression(X, theta, y, learning_rate, max_iteration, min_grad_norm, n, lambda_penalty = 0):
	
	x_iteration = []
	x_mse_per_iteration = []
	
	for iter in range(max_iteration):
		
		for i in range(n):
			
			rnd_idx = np.random.randint(0, n)
			X_i = X[rnd_idx, :].reshape(1, X.shape[1])
			y_i = y[rnd_idx].reshape(1,1)
			
		    # First step is .T -> Transpose, because first matrix have 6 columns, second must have 6 rows
		    # Second step is do 'dot product' in order to calculate predicted response variable by formula in below:
		    # Yi = b0 * 1 + b1 * Xi1 + b2 * Xi2 + ... + bK * XiK + E --> MULTIPLE LINEAR REGRESSION
			pred = np.dot(X_i, theta.T)
			
		    # Calculate errors per item (i), in next line we will calculate squered sum. RESIDUALS
			residuals = (pred - y_i)
			
			# Cost function - Loss function
			# Calculate mean_square_error (first calculate sum of squered residuals and then devide by n to get the mean)
			mean_square_error = np.dot(residuals.T, residuals) / n
			# sum_squered_error = err.T.dot(err)
			
			# Calculate penalty vector
			penalty_vector = (lambda_penalty * theta)
			# Exclude intercept
			penalty_vector[0][X.shape[1]-1] = 0 
			
			# Gradient calculation - Ridge regression - derivative
			gradient_vector = (np.dot(residuals.T, X_i) + penalty_vector) / n
			
			# Calculate step_size by multiplying learning_rate and gradient_vector (gradient_vector => data descent to minimum).
			# step_size = gradient_vector * learning_rate
			# Calculate new theta - Loss function parameters (old w - step_size)
			theta = theta - gradient_vector * learning_rate
			
			# Calculate sum of absolute gradient_vector values
			grad_norm = abs(gradient_vector).sum()
			
			x_iteration.append((iter-1) +i)
			x_mse_per_iteration.append(mean_square_error[0][0])
			
			if grad_norm < min_grad_norm or mean_square_error < 10: break
		
			print(iter, grad_norm, mean_square_error)

	plt.plot(x_iteration, x_mse_per_iteration)
	plt.xlabel('No. of iterations')
	plt.ylabel('Mean squered error')
			
#%% Use linear regression with above function 
GD_LinearRegression(X, theta, y, learning_rate, max_iteration, min_grad_norm, n, 0)

#%% PREDICT
data_new = pd.read_csv('house_new.csv')
data_new = (data_new-X_mean)/X_std
data_new['X0'] = 1
prediction = data_new.values.dot(theta.T)
print(prediction)
