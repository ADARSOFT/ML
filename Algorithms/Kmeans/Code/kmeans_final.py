import pandas as pd
import numpy as np
import math as mt
from scipy.spatial.distance import cdist
from numpy.linalg import norm
import kmeans_lib

np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})
pd.set_option('chop_threshold',0.01)

#%% HELP method
def FindOutermostCentroids(p_k_number):
	current_centroid = data.sample(1)
	centroids = pd.DataFrame()
	centroids = centroids.append(current_centroid)
	distances_k = pd.DataFrame() 
	data_with_no_c = pd.DataFrame()

	for i in range(p_k_number-1):
		data_with_no_c = data[~data.isin(centroids)].dropna()
		for j in range(len(data_with_no_c)):
			distances_k = distances_k.append((abs(data_with_no_c.iloc[j:j+1,:].reset_index(drop=True) - current_centroid.reset_index(drop=True))), ignore_index=True)
		idx = np.argmax(distances_k.sum(axis = 1))
		current_centroid = data.iloc[idx:idx+1,:]
		centroids = centroids.append(current_centroid)
		distances_k = distances_k.iloc[0:0]
	
	return centroids

#%% ALGORITHM fit
def KMeans_Fit(p_k, p_experiment_numbers, p_feature_weights, p_max_iteration = 100, p_dist_measure_type = 'euclidian'):
	res = kmeans_lib.KMeansExperimentResponse([], np.zeros(p_experiment_numbers))
	res.CentroidHistory.clear()
	res.BestCentroid.clear()
	for expn in range(p_experiment_numbers):
		centroids = FindOutermostCentroids(p_k).reset_index(drop=True)
		res.CentroidHistory.append(centroids)
		assign = np.zeros((m,1))
		old_quality = float('inf')
		for it in range(p_max_iteration):
			quality = np.zeros(p_k) 
			for j in range(m):
				point = data.iloc[j]
				dist = []
				if p_dist_measure_type == 'city-block':
					dist = abs(point-centroids).mul(p_feature_weights).sum(axis=1)
				else: 
					dist = ((point-centroids)**2).mul(p_feature_weights).sum(axis=1).apply(mt.sqrt)
				assign[j] = np.argmin(dist)	
			for c in range(p_k):
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

#%% Predict kmeans
def KMeans_Predict(p_centroids, p_data):
	m_m, m_n = p_data.shape
	m_assign = np.zeros((m_m,1))
	for j in range(m_m):
		m_assign[j] = np.argmin(((p_data.iloc[j]-p_centroids)**2).sum(axis=1).apply(mt.sqrt))
	p_data['Cluster'] = m_assign
	return p_data

#%% Silhouette score calculation function
def SilhouetteScore(p_k, p_labeled_data, p_centroids):
	
	m_labeled_data = p_labeled_data
	m_labeled_data['Silhouette_index'] = 0
	m_centroids =  p_centroids
	m_centroids['NearestCluster'] = 0
	m_m, m_n = m_centroids.shape
	
	for i in range(m_m):
		m_centroids['NearestCluster'][i] = np.argmin(((m_centroids.iloc[i] - m_centroids[~ m_centroids.index.isin([i])])**2).sum(axis=1).apply(mt.sqrt))
	
	return m_labeled_data
	
#%% LOAD DATA and params configuration
data = pd.read_csv('../../../Data/boston.csv')#.set_index('country') # ZA BOSTON NE TREBA COUNTRY
m,n = data.shape
data_mean = data.mean()
data_std = data.std()
data = (data-data_mean)/data_std # Standard_score
boston_weights = [1,2,1,1,1,1,1,1,1,1,1,1,1,1]
life_weights = [1,2,1,1,1]

#%% Algorithm usage
res = kmeans_lib.KMeansExperimentResponse([], np.zeros(7))
res = KMeans_Fit(5, 7, boston_weights)
predict_model = KMeans_Predict(res.BestCentroid, data)
# predict_model.groupby(['Cluster']).count().iloc[:,-1]
silhoueteScore = SilhouetteScore(5, predict_model, res.BestCentroid)
	