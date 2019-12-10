import pandas as pd
import numpy as np
import math as mt
import kmeans_lib
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from copy import copy
np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})
pd.set_option('chop_threshold',0.01)

#%% Helper methods
def findInstancesThatAreNotCentroids(isSingleResponse, data, centroids):
	if isSingleResponse:
		# return random single instance that isn't centroid
		return data[~data.isin(centroids)].sample(1)
	else:
		# return all instances that aren't centroids
		return data[~data.isin(centroids)].dropna()

#%% TASK 4 - This method we use when we initialize clusters (already defined number of clusters)
def findOutermostCentroids(p_k_number):
	centroids = pd.DataFrame()
	current_centroid = findInstancesThatAreNotCentroids(True, data, centroids)
	centroids = centroids.append(current_centroid)
	distances_k = pd.DataFrame() 
	data_with_no_c = pd.DataFrame()

	for i in range(p_k_number-1):
		# Get only data without centroids
		data_with_no_c = findInstancesThatAreNotCentroids(False, data, centroids)		
		# Calculate distances from intances to current cluster
		distances_k = abs(data_with_no_c.sub(current_centroid.values.tolist()[0], axis = 'columns'))
		# Take index of outermost instance
		idx = np.argmax(distances_k.sum(axis = 1))
		# Take outermost centroid (instance) from training dataset by index
		current_centroid = data.iloc[idx:idx+1,:]
		# Add centroid to output list
		centroids = centroids.append(current_centroid)
		# Reset distances
		distances_k = distances_k.iloc[0:0]
	
	return centroids
#%% # TASK 0 AND 1 AND 2 (calculate weignted instances with optional measure type (euclidian, city-block)) 
def calculateWeightedDistance(dist_measure_type, point, centroids, feature_weights):
	if dist_measure_type == 'city-block':
		return abs(point-centroids).mul(feature_weights).sum(axis=1)
	else:
		return ((point-centroids)**2).mul(feature_weights).sum(axis=1).apply(mt.sqrt)

#%% Scalar mean distance
def GetScalarMeanDistance(p_cluster_data, p_cluster_point):
	resp = np.mean(((p_cluster_point - p_cluster_data)**2).sum(axis=1).apply(mt.sqrt))
	return resp

#%% CalculateSilhouette index value
def silhouetteIndexCalculation(a,b):	
	if a < b:
		return float(1- a/b)
	elif a > b:
		return float(b/a-1)
	elif a == b:
		return 0
	
#%% ALGORITHM fit
def fit(total_clusters_number, total_experiments_number, p_feature_weights, p_max_iteration = 100, p_dist_measure_type = 'euclidian'):
	res = kmeans_lib.KMeansExperimentResponse([], np.zeros(total_experiments_number))
	res.CentroidHistory.clear()
	res.BestCentroid.clear()
	# TASK 3 (restart algorithm N times and choose best centroid quality)
	for expn in range(total_experiments_number):
		# Cluster initialization, find outermost clusters on first step 
		centroids = findOutermostCentroids(total_clusters_number).reset_index(drop=True)
		res.CentroidHistory.append(centroids)
		assign = np.zeros((m,1))
		old_quality = float('inf')
		for it in range(p_max_iteration):
			quality = np.zeros(total_clusters_number)
		
		    # Calculate witch cluster instance belong (with optinal distances metrics)
			for j in range(m):
				dist = calculateWeightedDistance(p_dist_measure_type, data.iloc[j], centroids, p_feature_weights)
				# Assing instance
				assign[j] = np.argmin(dist)
				
			# Calculate new centroids (mean of subset)
			for c in range(total_clusters_number):
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
def predict(p_centroids, p_data):
	m_m, m_n = p_data.shape
	m_assign = np.zeros((m_m,1))
	for j in range(m_m):
		m_assign[j] = int(np.argmin(((p_data.iloc[j]-p_centroids)**2).sum(axis=1).apply(mt.sqrt)))
	p_data['Cluster'] = m_assign
	return p_data

#%% TASK 5. Silhouette score calculation function
def silhouetteScore(p_k, labeled_data, centroids):
	m_labeled_data = copy(labeled_data)
	m_centroids =  copy(centroids)
	m_labeled_data['IntraClusterMean'] = float(0)
	m_labeled_data['NearestClusterMean'] = float(0)
	m_labeled_data['SilhouetteIndex'] = float(0)
	m_centroids['NearestCluster'] = int(0)
	
	c_row_number, c_column_number = m_centroids.shape
	ld_row_number,ld_column_number = m_labeled_data.shape
	
	# Assign nearest centroids
	for i in range(c_row_number):
		m_centroids['NearestCluster'][i] = np.argmin(((m_centroids.iloc[i] - m_centroids[~ m_centroids.index.isin([i])])**2).sum(axis=1).apply(mt.sqrt))

	for i in range(ld_row_number):
		
		# Extraxt sigle instance into point variable 
		point = m_labeled_data.iloc[i, :-4]
		
		# Calculate self cluster mean distance
		self_cluster_data = m_labeled_data.loc[m_labeled_data['Cluster'] == m_labeled_data.iloc[i]['Cluster']].iloc[:,:-4]
		a = GetScalarMeanDistance(self_cluster_data[~self_cluster_data.index.isin([i])], point)
		m_labeled_data['IntraClusterMean'][i] = a
		
		# Calculate nearest cluster mean distance 
		current_centroid = int(m_labeled_data.iloc[i]['Cluster'])
		nearest_centroid = int(m_centroids.iloc[current_centroid]['NearestCluster'])
		b = GetScalarMeanDistance(m_labeled_data.loc[m_labeled_data['Cluster'] == nearest_centroid].iloc[:,:-4], point)
		m_labeled_data['NearestClusterMean'][i] = b
		
		# Calculate solhouette index
		m_labeled_data['SilhouetteIndex'][i] = silhouetteIndexCalculation(a,b)
	
	return np.mean(m_labeled_data[['SilhouetteIndex']])

#%% LOAD DATA and params configuration
boston_weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
life_weights = [1,1,1,1,1]
visina_tezina_weights = [1,1]
alphas = visina_tezina_weights
experiments = 4
silhoueteScoreHistory = []
init_data = pd.read_csv('../../../Data/visina_tezina.csv')#.set_index('country') # Just for life csv dataset

for i in range(3):
	clusters = i+2
	data = copy(init_data)
	m,n = data.shape
	data_mean = data.mean()
	data_std = data.std()
	data = (data-data_mean)/data_std # Standard_score
	res = kmeans_lib.KMeansExperimentResponse([], np.zeros(experiments))
	res = fit(clusters, experiments, alphas)
	predict_model = predict(res.BestCentroid, data)
	score = sklearn_silhouette_score(predict_model.iloc[:,0:n], predict_model.iloc[:,-1], metric='euclidean')
	silhoueteScore = silhouetteScore(clusters, predict_model, res.BestCentroid)
	silhoueteScoreHistory.append([clusters, score, silhoueteScore])
