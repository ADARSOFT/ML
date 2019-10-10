import pandas as pd
import numpy as np
import math as mt
import kmeans_lib
from sklearn.metrics import silhouette_score

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
		#data_with_no_c = data_with_no_c.iloc[0:0]
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
		m_assign[j] = int(np.argmin(((p_data.iloc[j]-p_centroids)**2).sum(axis=1).apply(mt.sqrt)))
	p_data['Cluster'] = m_assign
	return p_data

#%% Silhouette score calculation function
def GetScalarMeanDistance(p_cluster_data, p_cluster_point):
	#print('-------------------------------------------------')
	#print('Cluster data:')
	#print(p_cluster_data)
	#print('Cluster point')
	#print(p_cluster_point)
	#print('Cluster poin minus cluster data')
	#print(p_cluster_point - p_cluster_data)
	#print('Mean(cluster_point-cluster_data)')
	#print(np.mean(p_cluster_point - p_cluster_data))
	resp = np.mean(((p_cluster_point - p_cluster_data)**2).sum(axis=1).apply(mt.sqrt))
	#print('Mean of Mean: {}'.format(resp))
	return resp

def SilhouetteScore(p_k, p_labeled_data, p_centroids):
	
	# Prepare, add new columns
	#m_labeled_data = predict_model
	#m_centroids =  res.BestCentroid
	m_labeled_data = p_labeled_data
	m_centroids =  p_centroids
	m_labeled_data['IntraClusterMean'] = float(0)
	m_labeled_data['NearestClusterMean'] = float(0)
	m_labeled_data['SilhouetteIndex'] = float(0)
	m_centroids['NearestCluster'] = int(0)
	
	c_row_number, c_column_number = m_centroids.shape
	ld_row_number,ld_column_number = m_labeled_data.shape
	
	# Dodeljujem najblize centroide
	for i in range(c_row_number):
		m_centroids['NearestCluster'][i] = np.argmin(((m_centroids.iloc[i] - m_centroids[~ m_centroids.index.isin([i])])**2).sum(axis=1).apply(mt.sqrt))
	
	# Calculate SilhuetteIndex per observation by formula (a-b) / max(a,b)	
	for i in range(ld_row_number):
		#i = 0
		point = m_labeled_data.iloc[i, :-4]
		self_cluster_data = m_labeled_data.loc[m_labeled_data['Cluster'] == m_labeled_data.iloc[i]['Cluster']].iloc[:,:-4]
		a = GetScalarMeanDistance(self_cluster_data[~self_cluster_data.index.isin([i])], point)
		m_labeled_data['IntraClusterMean'][i] = a
		current_centroid = int(m_labeled_data.iloc[i]['Cluster'])
		nearest_centroid = int(m_centroids.iloc[current_centroid]['NearestCluster'])
		b = GetScalarMeanDistance(m_labeled_data.loc[m_labeled_data['Cluster'] == nearest_centroid].iloc[:,:-4], point)
		m_labeled_data['NearestClusterMean'][i] = b
		
		if a < b:
			m_labeled_data['SilhouetteIndex'][i] = float(1- a/b)
		elif a > b:
			m_labeled_data['SilhouetteIndex'][i] = float(b/a-1)
		elif a == b:
			m_labeled_data['SilhouetteIndex'][i] = 0
	
	return np.mean(m_labeled_data[['SilhouetteIndex']])

#%% LOAD DATA and params configuration
data = pd.read_csv('../../../Data/boston.csv')#.set_index('country') # ZA BOSTON NE TREBA COUNTRY
m,n = data.shape
data_mean = data.mean()
data_std = data.std()
data = (data-data_mean)/data_std # Standard_score
boston_weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
life_weights = [1,1,1,1,1]
visina_tezina_weights = [1,1]

#%% Algorithm usage

res = kmeans_lib.KMeansExperimentResponse([], np.zeros(7))


res = KMeans_Fit(3, 7, boston_weights)
predict_model = KMeans_Predict(res.BestCentroid, data)

score = silhouette_score(predict_model.iloc[:,0:14], predict_model.iloc[:,-1], metric='euclidean')

# predict_model.groupby(['Cluster']).count().iloc[:,-1]
silhoueteScore = SilhouetteScore(3, predict_model, res.BestCentroid)
silhoueteScore

# 5. -0.378873, 2. -0.673033, 3. -0.354913 4. -0.29497 5. -0.326227 10. -0.544737