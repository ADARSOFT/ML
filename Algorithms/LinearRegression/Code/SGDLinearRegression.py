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
learning_rate = 0.01
# Define maximum iteration number
max_iteration = 80000
# Define gradient min norm
min_grad_norm = 0.01
# Define batch_size
batch_size = 1
#%% Calculate batch size
def DisplayBatchSize(batch_size, n):
	
	n_batches = int(n / batch_size)
	if n_batches * batch_size < n:
		n_batches = n_batches + 1
	
	print('Number of batches {}'.format(n_batches))
#%%
	
def SGD_LinearRegression(X, theta, y, learning_rate, max_iteration, min_grad_norm, n, batch_size = 1, lambda_penalty = 0):
	DisplayBatchSize(batch_size, n)
	
	x_iteration = []
	x_mse_per_iteration = []
	
	for iter in range(max_iteration):
		# Shuffle source data X and y
		indices = np.random.permutation(n)	
		X_batch_shuffled = X[indices]		
		y_batch_shuffled = y[indices]
		
		# range(start_value, end_value_ step)
		for i in range(0, n, batch_size):
			# take samples for batch size
			X_i = X_batch_shuffled[i: i + batch_size]
			y_i = y_batch_shuffled[i: i + batch_size]
			# calc prediction
			pred_i = np.dot(X_i, theta.T)
			# calc residuals
			residuals_i = pred_i-y_i
			# calc MSE
			mean_square_error = np.dot(residuals_i.T, residuals_i) / n
			
			penalty_vector = (lambda_penalty * theta)
		    # Exclude intercept
			penalty_vector[0][X.shape[1]-1] = 0 
			# calc GD
			gradient_vector_i = (np.dot(residuals_i.T, X_i) + penalty_vector) / n
			# calc step_size
			step_size_i = gradient_vector_i * learning_rate
			# calc new theta weights
			theta = theta - step_size_i
			# calc grad_norm
			grad_norm = abs(gradient_vector_i).sum()
			
			x_iteration.append(batch_size * (iter-1) +i)
			x_mse_per_iteration.append(mean_square_error[0][0])
			
			if grad_norm < min_grad_norm or mean_square_error < 10: break
		
			print(iter, i, grad_norm, mean_square_error)
			
	plt.plot(x_iteration, x_mse_per_iteration)
	plt.xlabel('No. of iterations')
	plt.ylabel('Mean squered error')
			
#%% Use linear regression with above function 
SGD_LinearRegression(X, theta, y, learning_rate, max_iteration, min_grad_norm, n, 100, 8)

#%% PREDICT
data_new = pd.read_csv('house_new.csv')
data_new = (data_new-X_mean)/X_std
data_new['X0'] = 1
prediction = data_new.values.dot(theta.T)
print(prediction)










