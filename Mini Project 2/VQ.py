from util import *

class VQ : 
	def __init__(self, input_dimension, number_symbols) :
		self.number_symbols = number_symbols
		self.input_dimension = input_dimension
		self.centroids = np.random.randn(number_symbols,input_dimension)

	def train(self, data_mfcc, iterations) : 
		data_points_list = []
		[data_points_list.append([]) for j in range(self.number_symbols)]

		for i in range(iterations) : 
			for j in range(data_mfcc.shape[0]) : 
				present_data_point = data_mfcc[j].reshape(1,self.input_dimension)
				dist = np.sum((self.centroids - present_data_point)**2,axis = 1)
				centroids_index = np.argmin(dist)
				data_points_list[centroids_index].append(present_data_point)

			for j in range(len(data_points_list)) : 
				updated_centroid = np.zeros((1,self.input_dimension))
				for z in data_points_list[j] :
					updated_centroid = updated_centroid + z 
				if len(data_points_list[j]) != 0 :
					updated_centroid = updated_centroid/len(data_points_list[j])
				else : 
					updated_centroid = self.centroids[j,:]			
				self.centroids[j,:] = updated_centroid

		return self.centroids

	def assign_symbols(self, trained_centroids = None, training_instance = None) : 
		self.centroids = trained_centroids

		symbol_list = np.zeros((1,training_instance.shape[0]))	
		for j in range(training_instance.shape[0]): 
			dist = np.sum((self.centroids - training_instance[j,:])**2,axis = 1)
			centroids_index = np.argmin(dist)
			symbol_list[0,j] = centroids_index
		return symbol_list	