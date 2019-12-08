import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#%% Helper functions
def ConvToMinusOneOrPlusOne(data):
	return data*2-1

def ConvToOppositeSign(data):
	return data*(-1)

#%% Prepare data
	
data = pd.read_csv('drugY.csv')
X = data.drop('Drug',axis=1)
y = ConvToMinusOneOrPlusOne(data['Drug'])
n,m = X.shape

# Convert categorical to numerical data
X = pd.get_dummies(X)

# Initial instances weights. Same weights for all instances.
alfas = pd.Series(np.array([1/n]*n), index=data.index)
ensemble_size = 10
weights = np.zeros(ensemble_size)   # tezine modela u ansamblu, koje odredjuju jacinu prilikom glasanja
ensemble = []

#%% Algorithm
for i in range(ensemble_size):
	
	# "weak" learners - work with numerical data
    alg = GaussianNB()
	
	# Define model
    model = alg.fit(X,y, sample_weight=alfas)
	
    # Model error in prediction
    error = (model.predict(X)-y).abs()/2
	
    # Total wighted error with intance weights (sum of instance weights are 1 - alfas)
	# total_error is in range from 0 to 1
    total_error = (error*alfas).sum()
	
	# Classifiers weights calculation (models weight)
	# Log can't use negative numbers as params, if 1 is parama then w is 0, there is no impact in reweighting
	# If w is negative, then his vote we use as opposite in next exp formula
    w = 1/2 * math.log((1-total_error)/total_error)
    
    ensemble.append(model)
    weights[0] = w
    
	# Reweighting instances
	# Step 1: Convert to opposite sign -> model weight and instance weight
	# Step 2: Multiply model weight with instance errors (all converted to opposite signs).
	# Step 3: Pass result as param into np.exp() function
	# Step 4: Multiply old instance weights with np.exp() result
	# If exponent is 0 then we do not change alfa because np.exp(0) = 1
	# If exponent is greater then 0, example: np.exp(0.1) = 1.1051, in that case we increase instance alfa
	# If exponent is lower then 0, example: np.exp(-1) = 0.367879, in that case we decrease instace alfa
    alfas = alfas * np.exp(ConvToOppositeSign(w) * ConvToOppositeSign(ConvToMinusOneOrPlusOne(error)))
	
    # Norm for normalization
    z = alfas.sum()
    alfas = alfas/z
    
    # OVAJ DEO JE POTREBAN SAMO ZA DIJAGNOSTIKU
    predictions = pd.DataFrame([model.predict(X) for model in ensemble]).T
    predictions = np.sign(predictions.dot(weights[:i+1]))
    print('Ensemble with {} models, accuracy: {}'.format(i+1, accuracy_score(y,predictions)))


# EVALUIRAJ SVAKI MODEL POSEBNO
for i,model in enumerate(ensemble):
    print('Model {}, accuracy: {} %'.format(i,accuracy_score(y, np.sign(model.predict(X))) * 100))

# EVALUACIJA CELOG ANSAMBLA
predictions = pd.DataFrame([model.predict(X) for model in ensemble]).T
predictions = np.sign(predictions.dot(weights))
print('Final ensemble accuracy: {} %'.format(accuracy_score(y,predictions)*100))

# https://www.youtube.com/watch?v=UHBmv7qCey4