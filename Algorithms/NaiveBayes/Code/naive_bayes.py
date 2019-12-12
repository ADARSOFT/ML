import pandas as pd
import numpy as np
from math import sqrt
from math import pi
from math import exp

#%%  Fit model
def fit(data, outputClass, alfa):
	model = {}
	# Calculate apriori probability
	apriori = data[outputClass].value_counts()
	# Normalize (total sum apriori prob is equal to 1)
	apriori = apriori / apriori.sum()
	model['apriori'] = apriori.to_frame()
	for atribut in data.drop(outputClass, axis=1).columns:	
		colType = data[atribut].dtype
		if(colType != np.float64 and colType != np.int64):
			res = smoothedAdditiveProbability(data[atribut],data[outputClass], alfa)
			model[atribut] = res
	return model

#%% TASK 2: Smoothed additive probability
# Formula: (count(x) + alfa) / (N + K * alfa) 
# N - Total number
# K - Categories count
# count(x) - How meny x exist in scope N (x frequency)
	 
def smoothedAdditiveProbability(attributData, outputClassData, alfa):
	# Calculate count(x)
	x_freq = pd.crosstab(attributData,outputClassData)
	# Calculate K - categories count
	K = len(x_freq.index)
	# Calculate N
	N = x_freq.sum(axis = 0)
	# Calculate equation
	smoothedAdditiveProb = (x_freq + alfa).div(N + K * alfa)
	
	return smoothedAdditiveProb

#%% TASK 3: Calculate continious numerical attriboute probability 
def continiousNumericProbability(point, continiousAndOutputData, outputClassName):

	dataGroup = continiousAndOutputData.groupby(outputClassName)
	dataGroup = [dataGroup.get_group(x) for x in dataGroup.groups]
	
	for i in range(len(dataGroup)):
		columnData = dataGroup[i].iloc[:,0:1]
		class_mean = (np.sum(columnData) / len(columnData) )[0]
		class_std = np.std(dataGroup[i].iloc[:,0:1].values)
		prob = calculate_PDF(point, class_mean, class_std)
			
	return prob

#%% Gaussian Probability Density function
def calculate_PDF(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

#%% TASK 1: Underflow problem helper
def logarithmTransformation(number):
	# We use exp function, because we have negative numbers, logarithm can't work with negative numbers.
	return np.log(np.exp(number))

#%% Predict values
def predict(model, slucaj, p_outputClass):
	predictResponse = {}
	# Iterate throught model for output class (example "da" or "ne")
	for outputClass in model['apriori'].index:
		probability = 1
		# Iterate throuth features or attributes 
		for atribut in data.drop(p_outputClass, axis=1).columns:
			if atribut == 'apriori':
				probability = probability * model['apriori'].loc[outputClass,:][0]
			else:
				colType = data[atribut].dtype
				conditionProb = 0
				
				if(colType == np.float64 or colType == np.int64):
					# Get probability with continiousNumericProbability function (not from model)
					conditionProb = continiousNumericProbability(slucaj[atribut], data[[atribut, p_outputClass]] , data[p_outputClass])
				else:	
					# Get probability from model for specific slucaj and specific attribute, for specific output class 
					conditionProb = model[atribut].loc[slucaj[atribut]][outputClass]
				probability = logarithmTransformation(probability * conditionProb)

		predictResponse[outputClass]=probability
	return predictResponse

#%% Algorithm usage
# Configuration part
prehlada_path = '../../../data/prehlada.csv'
prehlada_novi_path = '../../../data/prehlada_novi.csv'
drug_path = '../../../data/drug.csv'
prehlada_label = "Prehlada"
drug_label = 'Drug'
label = drug_label

# LoadData (training and test)
data = pd.read_csv(drug_path)
#data_new = pd.read_csv('prehlada_novi.csv')
data_new, data = np.split(data, [int(.05*len(data))])

# Learn model
model = fit(data, label, 1)

# Predictions on test data
for i in range(len(data_new)):
	point = data_new.loc[i]
	prediction = predict(model, point, label)
	data_new.loc[i,'prediction'] = max(prediction, key=lambda x: prediction[x])
	print(data_new.loc[i,'prediction'])
	for klasa in prediction:
		data_new.loc[i,'klasa='+klasa] = prediction[klasa]

print(data_new)

# TO DO: Ostaje mi da sada kada sam izracunao verovatnoce, upisem to u response u model.!!