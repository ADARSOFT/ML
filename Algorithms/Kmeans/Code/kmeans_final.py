import pandas as pd
import numpy as np
import math as mt
from scipy.spatial.distance import cdist
from numpy.linalg import norm

np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})
pd.set_option('chop_threshold',0.01)
#%% LOAD DATA
data = pd.read_csv('../../../Data/life.csv').set_index('country')
k = 2
m,n = data.shape
#%% NORMALIZATION https://en.wikipedia.org/wiki/Standard_score
data_mean = data.mean()
data_std = data.std()
data = (data-data_mean)/data_std

current_centroid = data.sample(1).reset_index(drop=True)
centroids = pd.DataFrame()
centroids = centroids.append(current_centroid)
distances_k = pd.DataFrame() #current_centroid.copy()
k_number = 5

for i in range(k_number-1):
	for j in range(m):
	# racunam distance za svaki centroid posebno (city-block metrika) oduzimam od centroida
		distances_k = distances_k.append((abs(data.iloc[j:j+1,:].reset_index(drop=True) - current_centroid)), ignore_index=True)
	# setujem novi centroid, najidaljeniji od starog
	# TO DO: obrisi onaj koji smo uzeli
	idx = np.argmax(distances_k.sum(axis = 1))
	current_centroid = data.iloc[idx:idx+1,:].reset_index(drop=True)
	# ubaci u listu centroida
	centroids = centroids.append(current_centroid)
	distances_k = distances_k.iloc[0:0]

#%% ALGORITHM
def KMeans_experiment(experiment_numbers, feature_weights, max_iteration = 100, dist_measure_type = 'euclidian'):
	res = KMeansExperimentResponse([], np.zeros(experiment_numbers))
	res.CentroidHistory.clear()
	res.BestCentroid.clear()
	for expn in range(experiment_numbers):
		centroids = data.sample(k).reset_index(drop=True)
		res.CentroidHistory.append(centroids.values)
		assign = np.zeros((m,1))
		old_quality = float('inf')
		for it in range(max_iteration):
			quality = np.zeros(k) 
			for j in range(m):
				point = data.iloc[j]
				dist = []
				if dist_measure_type == 'city-block':
					dist = abs(point-centroids).mul(feature_weights).sum(axis=1)
				else: 
					dist = ((point-centroids)**2).mul(feature_weights).sum(axis=1).apply(mt.sqrt)
				assign[j] = np.argmin(dist)	
			for c in range(k):
				subset = data[assign==c]
				centroids.loc[c] = subset.mean()
				quality[c] = subset.var().sum() * len(subset)
			print('Redni broj iteracije: {}'.format(it))
			if np.array_equal(quality, old_quality): break
			old_quality = quality
		print('Experiment number: {} Quality: {}.'.format(expn, quality.sum()))
		res.QualityHistory[expn] = quality.sum()
	res.BestCentroid = res.CentroidHistory[np.argmin(res.QualityHistory)]
	return res	
#%% Call function
res = KMeans_experiment(10, [1,2,1,1,1])
res.CentroidHistory
res.QualityHistory
res.BestCentroid 
#%% Objects
class KMeansExperimentResponse:
	def __init__(self, bestCentroid = [], qualityHistory = [], centroidHistory = []):
		self.BestCentroid = bestCentroid
		self.QualityHistory = qualityHistory
		self.CentroidHistory = centroidHistory