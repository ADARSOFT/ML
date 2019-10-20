import pandas as pd
import numpy as np
import copy as c
from scipy.stats import norm
import matplotlib.pyplot as plt

#%%  UCENJE
def learn(data, outputClass, alfa):
	model = {}
	apriori = data[outputClass].value_counts()
	apriori = apriori / apriori.sum()
	model['apriori'] = apriori.to_frame()
	for atribut in data.drop(outputClass, axis=1).columns:	
		colType = data[atribut].dtype
		if(colType == np.float64 or colType == np.int64):
			# to do:
			model[atribut] = continiousNumericProbability(data[atribut],data[outputClass])
		else:
			res = smoothedAdditiveProbability(data[atribut],data[outputClass], alfa)
			print(type(res))
			model[atribut] = res
	return model

#%% Smoothed additive probability
def smoothedAdditiveProbability(attributData, outputClassData, alfa):
	attCatOutFreq = pd.crosstab(attributData,outputClassData)
	attCatCount = len(attCatOutFreq.index) 
	counter = attCatOutFreq + alfa
	denominator = attCatOutFreq.sum(axis = 0) + (attCatCount * alfa)
	smoothedAdditiveProb = counter.div(denominator)
	return smoothedAdditiveProb

#%% Calculate continious numerical attriboute probability 
def continiousNumericProbability(data, outputClass):
	dataClasses = data[['K', 'Drug']].groupby('Drug')
	dataClasses = [dataClasses.get_group(x) for x in dataClasses.groups]
	
	for i in range(len(dataClasses)):
		i = 0
		columnData = dataClasses[i].iloc[:,0:1]
		class_mean = (np.count_nonzero(columnData) / np.sum(columnData))[0]
		class_std = np.std(dataClasses[i].iloc[:,0:1])
		for j in range(len(dataClasses[i])):
			x = dataClasses[i].iloc[0][0]
			prob = calculate_PDF(x, class_mean, class_std)

	return 1

#%% Gaussian Probability Density function
def calculate_PDF(x, mean, stdev):
	exponent = np.exp(-((x-mean)**2/(2*stdev**2)))
	result = (1 / (np.sqrt(2*np.pi)*stdev)) * exponent
	return result

#%% PREDVIDJANJE
def predict(model, slucaj):
	predictResponse = {}
	for outputClass in model['apriori'].index:
		probability = 1
		for atribut in model:
			if atribut == 'apriori':
				probability = probability * model['apriori'][outputClass]
			else:
				conditionProb = model[atribut][outputClass][slucaj[atribut]]
				exponentValue = np.exp(conditionProb)
				logarithmOfSumOfExponent = np.log(exponentValue)
				probability = probability * logarithmOfSumOfExponent
		predictResponse[outputClass]=probability
	return predictResponse

#%% KORISCENJE
data = pd.read_csv('drug.csv')
model = learn(data,'Prehlada', 0)
'''
for i in data.drop('Prehlada', axis=1).columns:
	print(model[i])
'''
data_new = pd.read_csv('prehlada_novi.csv')
for i in range(len(data_new)):
	point = data_new.loc[i]
	prediction = predict(model, point)
	data_new.loc[i,'prediction'] = max(prediction, key=lambda x: prediction[x])
	for klasa in prediction:
		data_new.loc[i,'klasa='+klasa] = prediction[klasa]

print(data_new)

'''
Underflow with log and exp solution for the problem.
https://stackoverflow.com/questions/33434032/avoid-underflow-using-exp-and-minimum-positive-float128-in-numpy
https://www.youtube.com/watch?v=-RVM21Voo7Q
Laplace smoothing (additive)
https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf
https://en.wikipedia.org/wiki/Additive_smoothing
Gaussian PDF
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
'''
