import pandas as pd
import numpy as np
import copy as c

#%%  UCENJE
def learn(p_data, outputClass, alfa):
	model = {}
	apriori = data[outputClass].value_counts()
	apriori = apriori / apriori.sum()
	model['apriori'] = apriori

	for atribut in data.drop(outputClass, axis=1).columns:
		attCatOutFreq = pd.crosstab(data[atribut],data[outputClass])
		attCatCount = len(attCatOutFreq.index) 
		counter = attCatOutFreq + alfa
		denominator = attCatOutFreq.sum(axis = 0) + (attCatCount * alfa)
		smoothedAdditiveProb = counter.div(denominator)
		model[atribut] = smoothedAdditiveProb

	return model
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
data = pd.read_csv('prehlada.csv')
model = learn(data,'Prehlada',2)

data_new = pd.read_csv('prehlada_novi.csv')
for i in range(len(data_new)):
	point = data_new.loc[i]
	prediction = predict(model, point)
	data_new.loc[i,'prediction'] = max(prediction, key=lambda x: prediction[x])
	for klasa in prediction:
		data_new.loc[i,'klasa='+klasa] = prediction[klasa]

print(data_new)

# Underflow with log and exp solution for the problem.
# https://stackoverflow.com/questions/33434032/avoid-underflow-using-exp-and-minimum-positive-float128-in-numpy
# https://www.youtube.com/watch?v=-RVM21Voo7Q
# Laplace smoothing (additive)
# https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf
# https://en.wikipedia.org/wiki/Additive_smoothing
