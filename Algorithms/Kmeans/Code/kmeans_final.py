import pandas as pd
import numpy as np
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

#%% Class
class Kmeans:
	''' Implementing custom Kmeans algorithm'''
	
	def __init__(self, n_clusters, max_iter, random_state = 123):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.random_state = random_state
		
	def init_centroids_m1(self, X):
		centroids = data.sample(k).reset_index(drop=True)
		return centroids
	
	def init_centroids_m2(self, X):
		return 0 # TO DO:
	
	def compute_distance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.n_clusters))
		for k in range(self.n_clusters):
			row_norm = norm(X - centroids.[k, :].values, axis=1)
			distance[:, k] = np.square(row_norm)
		return distance
	
	def find_closest_cluster(self, distance):
		return np.argmin(distance, axis = 1)
	
	def compute_centroids(self, X, labels):
		centroids = np.zeros((self.n_clusters, X.shape[1]))
		for k in range(self.n_clusters):
			mean1 = X[labels == k].mean(axis=0)
			centroids[k, :] = mean1
		return centroids
	def compute_sse(self, X, labels, centroids):
		distance = np.zeros(X.shape[0])
		for k in range(self.n_clusters):
			distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
		return np.sum(np.square(distance))
	
	def fit(self, X):
		self.centroids = self.init_centroids_m1(X)
		for i in range(self.max_iter):
			old_centroids = self.centroids
			distance = self.compute_distance(X, old_centroids)
			self.labels = self.find_closest_cluster(distance)
			self.centroids = self.compute_centroids(X, self.labels)
			if np.all(old_centroids == self.centroids):
				break
			old_centroids = centroids
		self.error = self.compute_sse(X, self.labels, self.centroids)	
	def predict(self, X):
		distance = self.compute_distance(X, self.centroids)
		return self.find_closest_cluster(distance)
	
#%% Call alg
		
alg = Kmeans(n_clusters=2, max_iter=10)
centroids3 = alg.init_centroids_m1(data)
distances3 = alg.compute_distance(data, centroids3)
labels_cls = alg.find_closest_cluster(distances3)
new_centroids = alg.compute_centroids(data, labels_cls)
new_centroids

alg2 = Kmeans(n_clusters=2, max_iter=2)
alg2.fit(data)


#%% INITIALIZATION
# samle uzima uzorak a ne random value, mada ni ovo nije pogresno.
centroids = data.sample(k).reset_index(drop=True)

#%% ALGORITHM
max_iter = 10
assign = np.zeros((m,1))
old_quality = float('inf')


for it in range(max_iter):
	quality = np.zeros(k)

	for j in range(m):
		point = data.iloc[j]
		dist = ((point-centroids)**2).sum(axis=1)
		assign[j] = np.argmin(dist)

	for c in range(k):
		subset = data[assign==c]
		centroids.loc[c] = subset.mean()
		quality[c] = subset.var().sum() * len(subset)

	print(centroids)
	print(quality.sum(), quality)

	if old_quality and abs(old_quality-quality)<0.1: break
	old_quality = quality

