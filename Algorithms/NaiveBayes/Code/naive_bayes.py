import pandas as pd
import numpy as np
import math 
import sys
import copy
import extmath

#%%  UCENJE
def learn(podaci, klasa):
	model = {}
	apriori = podaci[klasa].value_counts()
	apriori = apriori / apriori.sum()
	model['apriori'] = apriori

	for atribut in podaci.drop(klasa, axis=1).columns:
		mat_kont = pd.crosstab(podaci[atribut],podaci[klasa])
		mat_kont = mat_kont.div(mat_kont.sum())
		model[atribut] = mat_kont

	return model
#%% PREDVIDJANJE
def predict(model, slucaj):
	predvidjanje = {}
	#slucaj = point 
	for klasa in model['apriori'].index:
		#klasa = 'ne'
		verovatnoca = 1
		for atribut in model:
			if atribut == 'apriori':
				verovatnoca = verovatnoca * model['apriori'][klasa]
			else:
				# Pomnozi prethodni P sa P iz modela za slucaj (njegov atribut)
				caseProb = model[atribut][klasa][slucaj[atribut]]
				verovatnoca = verovatnoca * np.log(np.sum(np.exp(caseProb)))
				if verovatnoca < 0:
					print('Verovatnoca manja od 0: {}'.format(verovatnoca))

		predvidjanje[klasa]=verovatnoca
	return predvidjanje
#%% KORISCENJE
data = pd.read_csv('prehlada.csv')
model = learn(data,'Prehlada')

data_new = pd.read_csv('prehlada_novi.csv')
for i in range(len(data_new)):
	#i = 0
	point = data_new.loc[i]
	prediction = predict(model, point)
	data_new.loc[i,'prediction'] = max(prediction, key=lambda x: prediction[x])
	for klasa in prediction:
		data_new.loc[i,'klasa='+klasa] = prediction[klasa]

print(data_new)
# https://stackoverflow.com/questions/33434032/avoid-underflow-using-exp-and-minimum-positive-float128-in-numpy
# https://www.youtube.com/watch?v=-RVM21Voo7Q
print(np.log(0.444))
print(np.log(3))
print(np.log(-9))
print(np.log(4))