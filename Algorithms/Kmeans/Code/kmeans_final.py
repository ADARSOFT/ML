import pandas as pd
import numpy as np
import math as mt
from scipy.spatial.distance import cdist
from numpy.linalg import norm

np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})
pd.set_option('chop_threshold',0.01)

#%% LOAD DATA
data = pd.read_csv('../../../Data/life.csv').set_index('country')
k = 4
m,n = data.shape
#%% NORMALIZATION https://en.wikipedia.org/wiki/Standard_score
data_mean = data.mean()
data_std = data.std()
data = (data-data_mean)/data_std
feature_weights = [1,2,1,1,1]

#%% INITIALIZATION
# samle uzima uzorak a ne random value, mada ni ovo nije pogresno.
centroids = data.sample(k).reset_index(drop=True)

#%% ALGORITHM
max_iter = 100
distance_measure_type = 'euclidian'
assign = np.zeros((m,1))
old_quality = float('inf')

for it in range(max_iter):
	quality = np.zeros(k)

	for j in range(m):
		point = data.iloc[j]
		dist = []
		if distance_measure_type == 'city-block':
			dist = abs(point-centroids).mul(feature_weights).sum(axis=1)
		else: #euclidian
			dist = ((point-centroids)**2).mul(feature_weights).sum(axis=1).apply(mt.sqrt)
		assign[j] = np.argmin(dist)	

	for c in range(k):
		subset = data[assign==c]
		centroids.loc[c] = subset.mean()
		quality[c] = subset.var().sum() * len(subset)

	#print(centroids)
	#print(quality.sum(), quality)
	print('Redni broj interacije: {}'.format(it))

	if np.array_equal(quality, old_quality): break
	
	old_quality = quality

print(quality)
