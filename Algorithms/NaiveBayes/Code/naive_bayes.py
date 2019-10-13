import pandas as pd
import sys

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
	for klasa in model['apriori'].index:
		verovatnoca = 1
		for atribut in model:
			if atribut == 'apriori':
				verovatnoca = verovatnoca * model['apriori'][klasa]
			else:
				verovatnoca = verovatnoca * model[atribut][klasa][slucaj[atribut]]

		predvidjanje[klasa]=verovatnoca
	return predvidjanje

#%% KORISCENJE
data = pd.read_csv('prehlada.csv')
model = learn(data,'Prehlada')

data_new = pd.read_csv('prehlada_novi.csv')
for i in range(len(data_new)):
	point = data_new.loc[i]
	prediction = predict(model, point)
	data_new.loc[i,'prediction'] = max(prediction, key=lambda x: prediction[x])
	for klasa in prediction:
		data_new.loc[i,'klasa='+klasa] = prediction[klasa]

print(data_new)
