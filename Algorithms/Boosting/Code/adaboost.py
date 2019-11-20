import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug']*2-1
n,m = X.shape

X = pd.get_dummies(X)   # dummy coding: pretvaranje kategorickih u numericke

alfas = pd.Series(np.array([1/n]*n), index=data.index)   # tezine instanci: za prvi model su sve instance podjednako vazne

ensemble_size = 5
weights = np.zeros(ensemble_size)   # tezine modela u ansamblu, koje odredjuju jacinu prilikom glasanja
ensemble = []

for i in range(ensemble_size):
    alg = GaussianNB()   # "slabi"/jednostavni model 
    model = alg.fit(X,y, sample_weight=alfas)
    error = (model.predict(X)-y).abs()/2    # greska i-tog modela u predvidjanju
    total_error = (error*alfas).sum()       # ukupna (otezana) greska sa tezinama instanci
    w = 1/2 * math.log((1-total_error)/total_error)   # preracunavanje tezina modela
    
    ensemble.append(model)
    weights[i] = w
    
    alfas = alfas * np.exp(-w*(error*2-1)*(-1))   # preracunavanje novih tezina instanci, po formuli
    z = alfas.sum()   # norma za normalizaciju
    alfas = alfas/z
    
    # OVAJ DEO JE POTREBAN SAMO ZA DIJAGNOSTIKU
    predictions = pd.DataFrame([model.predict(X) for model in ensemble]).T
    predictions = np.sign(predictions.dot(weights[:i+1]))
    print('Ensemble with {} models, accuracy: {}'.format(i+1, accuracy_score(y,predictions)))


# EVALUIRAJ SVAKI MODEL POSEBNO
for i,model in enumerate(ensemble):
    print('Model {}, accuracy: {}'.format(i,accuracy_score(y, np.sign(model.predict(X)))))

# EVALUACIJA CELOG ANSAMBLA
predictions = pd.DataFrame([model.predict(X) for model in ensemble]).T
predictions = np.sign(predictions.dot(weights))
print('Final ensemble accuracy: ', accuracy_score(y,predictions))


    
