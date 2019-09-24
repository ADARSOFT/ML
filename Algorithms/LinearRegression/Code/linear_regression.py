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

y = y.as_matrix()
X = X.as_matrix()

n,m = X.shape

#%% INITIALIZATION
# Return random floats in the half-open interval [0.0, 1.0) for Slope's. Use array length as X number of columns (m)
w = np.random.random((1,m))
# Define learning rate for gradient descent
learning_rate = 0.1
# Define maximum iteration number
max_iteration = 10000
# Define gradient min norm
min_grad_norm = 0.01

#%% Create function with params 
def LearnWithGradientDescent(X, w, y, learning_rate, max_iteration, min_grad_norm, n):
	
	x_iteration = []
	x_mse_per_iteration = []
	
	for iter in range(max_iteration):
	    # First step is .T -> Transpose, because first matrix have 6 columns, second must have 6 rows
	    # Second step is do 'dot product' in order to calculate predicted response variable by formula in below:
	    # Yi = b0 * 1 + b1 * Xi1 + b2 * Xi2 + ... + bK * XiK + E --> MULTIPLE LINEAR REGRESSION
		pred = X.dot(w.T)
		
	    # Calculate errors per item (i), in next line we will calculate squered sum. RESIDUALS
		err = pred-y
		# Calculate mean_square_error (first calculate sum of squered error and then devide by n to get the mean)
		mean_square_error = err.T.dot(err) / n
		# sum_squered_error = err.T.dot(err)
		
		# Loss function
		# Calculate gradient errors per features (this means e1 * xi1 + e2 * xi2 + ... + eK * eiK) / n
		# Mean feature error -> per item (prosecna greska za svaki od feature-a) kolone sa vecim vrednostima imace vecu tezinu gresaka
		grad = err.T.dot(X) / n
		
		# Calculate step_size by multiplying learning_rate and grad (grad => data descent to minimum).
		step_size = grad * learning_rate
		# Calculate new w - Loss function parameters (old w - step_size)
		w = w - step_size
		# Calculate sum of absolute grad values
		grad_norm = abs(grad).sum()
		
		x_iteration.append(iter)
		x_mse_per_iteration.append(mean_square_error[0][0])
		
		if grad_norm < min_grad_norm or mean_square_error < 10: break
	
		print(iter, grad_norm, mean_square_error)

		plt.plot(x_iteration, x_mse_per_iteration)
		plt.xlabel('No. of iterations')
		plt.ylabel('Mean squered error')
		
#%% Use linear regression with above function 
LearnWithGradientDescent(X, w, y, learning_rate, max_iteration, min_grad_norm, n)

#%% PREDICT 
data_new = pd.read_csv('house_new.csv')
data_new = (data_new-X_mean)/X_std
data_new['X0'] = 1
prediction = data_new.as_matrix().dot(w.T)
print(prediction)