import pandas as pd
import numpy as np
import copy as c
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import sqrt
from math import pi
from math import exp
# from scipy.stats import norm # this is for testing purpose
#%%  UCENJE
def learn(data, outputClass, alfa):
	model = {}
	apriori = data[outputClass].value_counts()
	apriori = apriori / apriori.sum()
	model['apriori'] = apriori.to_frame()
	for atribut in data.drop(outputClass, axis=1).columns:	
		colType = data[atribut].dtype
		if(colType != np.float64 and colType != np.int64):
			res = smoothedAdditiveProbability(data[atribut],data[outputClass], alfa)
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

#%% PREDVIDJANJE
def predict(model, slucaj, p_outputClass):
	predictResponse = {}
	for outputClass in model['apriori'].index:
		probability = 1
		for atribut in data.drop(p_outputClass, axis=1).columns:
			if atribut == 'apriori':
				probability = probability * model['apriori'].loc[outputClass,:][0]
			else:
				colType = data[atribut].dtype
				conditionProb = 0
				
				if(colType == np.float64 or colType == np.int64):
					conditionProb = continiousNumericProbability(slucaj[atribut], data[[atribut, p_outputClass]] , data[p_outputClass])
				else:	
					conditionProb = model[atribut].loc[slucaj[atribut]][outputClass]
					
				exponentValue = np.exp(conditionProb)
				logarithmOfSumOfExponent = np.log(exponentValue)
				probability = probability * logarithmOfSumOfExponent
		predictResponse[outputClass]=probability
	return predictResponse

#%% KORISCENJE
data = pd.read_csv('prehlada.csv')
data = pd.read_csv('drug.csv')
data_new, data = np.split(data, [int(.05*len(data))])
label = 'Prehlada'
model = learn(data,label, 1)
'''
for i in data.drop('Prehlada', axis=1).columns:
	print(model[i])
'''
data_new = pd.read_csv('prehlada_novi.csv')
for i in range(len(data_new)):
	point = data_new.loc[i]
	prediction = predict(model, point, label)
	data_new.loc[i,'prediction'] = max(prediction, key=lambda x: prediction[x])
	print(data_new.loc[i,'prediction'])
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
# TO DO: Ostaje mi da sada kada sam izracunao verovatnoce, upisem to u response u model.!!