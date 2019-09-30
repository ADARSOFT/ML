class KMeansExperimentResponse:
	def __init__(self, bestCentroid = [], qualityHistory = [], centroidHistory = []):
		self.BestCentroid = bestCentroid
		self.QualityHistory = qualityHistory
		self.CentroidHistory = centroidHistory