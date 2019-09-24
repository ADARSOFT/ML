import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})
pd.set_option('chop_threshold',0.01)


#%% LOAD DATA
data = pd.read_csv('life.csv').set_index('country')
k = 2
m,n = data.shape

#%% NORMALIZATION
data_mean = data.mean()
data_std = data.std()
data = (data-data_mean)/data_std

#%% INITIALIZATION
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

