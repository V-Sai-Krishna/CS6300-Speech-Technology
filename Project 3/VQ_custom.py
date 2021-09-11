import numpy as np
def assign_symbols(trained_centroids = None, training_instance = None) : 

		symbol_list = np.zeros((1,training_instance.shape[0]))	
		for j in range(training_instance.shape[0]): 
			dist = np.sum((trained_centroids - training_instance[j,:])**2,axis = 1)
			centroids_index = np.argmin(dist)
			symbol_list[0,j] = centroids_index
		return symbol_list	