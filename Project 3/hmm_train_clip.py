from util import *
from functools import reduce

np.random.seed(42)

class HMM : 
	def __init__(self, number_symbols, number_states) : 
		self.number_symbols = number_symbols
		self.number_states = number_states
		
		self.pi_states = np.random.random((1,number_states))
		self.pi_states = self.pi_states/np.sum(self.pi_states)
		self.log_pi_states = np.log(self.pi_states)

		self.A = np.zeros((number_states, number_states))
		#for i in self.A.shape[0] : 
		#	self.A[i-1:i+1]
		for i in range(number_states) : 
			if i == (number_states - 1) :
				self.A[i,i] = 1 
				break
			self.A[i,i] = 0.6
			self.A[i,i+1] = 0.4
			#self.A[i,:] = self.A[i,:]/np.sum(self.A[i,:])
		#self.A = self.A/np.sum(np.sum(self.A))
		self.A += ((self.A < 1e-300) * 1e-300).astype(np.float64)

		self.log_A = np.log(self.A + 1e-300)
		self.B = np.random.random((number_symbols, number_states))
		for i in range(number_states) : 
			self.B[:,i] = self.B[:,i]/np.sum(self.B[:,i])
		#self.B = self.B/np.sum(np.sum(self.B))
		self.B += ((self.B < 1e-300) * 1e-300).astype(np.float64)
		self.log_B = np.log(self.B)
	
	def train(self, symbol_training_instance) : 
		#print('Length of the sample is : ',symbol_training_instance.shape[1])
		#print('Pi : ',self.pi_states)
		#print('A : ',self.A)
		#print('B : ',self.B)
		alpha = np.zeros((self.number_states, symbol_training_instance.shape[1]))
		beta = np.zeros((self.number_states, symbol_training_instance.shape[1]))
		beta[:,-1] = np.ones(self.number_states)

		for i in range(self.number_states) :
			alpha[i,0] = self.log_pi_states[0,i] + self.log_B[int(symbol_training_instance[0,0]),i]

		gamma = np.full((self.number_states, symbol_training_instance.shape[1]),-300,dtype = np.float64)
		eta = np.full((self.number_states, self.number_states, symbol_training_instance.shape[1]),-300,dtype = np.float64) 

		for t in range(1,symbol_training_instance.shape[1]) : 
			for j in range(self.number_states) :
				#alpha[j,t] = self.log_B[int(symbol_training_instance[0,t]),j] + np.log(np.sum(np.exp(alpha[:,t-1].reshape((1,-1)) + self.log_A[:,j].reshape((1,-1)))) + 1e-300)
				alpha[j,t] = self.log_B[int(symbol_training_instance[0,t]),j] + np.max(alpha[:,t-1].reshape((1,-1)) + self.log_A[:,j].reshape((1,-1))) + np.log(np.sum(np.exp(alpha[:,t-1].reshape((1,-1)) + self.log_A[j,:].reshape((1,-1)) - np.max(alpha[:,t-1].reshape((1,-1)) + self.log_A[j,:].reshape((1,-1))))))

		for t in range(symbol_training_instance.shape[1]-2,-1,-1) : 
			for j in range(self.number_states) :
				#beta[j,t] = np.log(np.sum(np.exp((self.log_A[i, :] + self.log_B[int(symbol_training_instance[0,t+1]),:] + beta[:,t+1]))) + 1e-300)
				beta[j,t] = np.max(self.log_A[i, :] + self.log_B[int(symbol_training_instance[0,t+1]),:] + beta[:,t+1]) + np.log(np.sum(np.exp(self.log_A[i,:] + self.log_B[int(symbol_training_instance[0,t+1]),:] + beta[:,t+1] -np.max(self.log_A[i, :] + self.log_B[int(symbol_training_instance[0,t+1]),:] + beta[:,t+1])) ))
				#for  k in range(self.number_states) : 
				#	beta[j,t]  = np.log(np.exp(beta[j,t]) + np.exp(np.log(self.A[j,k] + 1e-300) + beta[k,t+1] + np.log(self.B[int(symbol_training_instance[0,t+1]),k] + 1e-300)) + 1e-300)### If wrong change to A[k,i+1]

		
		for t in range(symbol_training_instance.shape[1]) :
			#gamma[:,t] = alpha[:,t] + beta[:,t] - np.log(np.sum(np.exp(alpha[:,t]+beta[:,t]))+1e-300)
			gamma[:,t] =   alpha[:,t] + beta[:,t] - np.max(alpha[:,t]+beta[:,t]) - np.log(np.sum(np.exp(alpha[:,t]+beta[:,t] - np.max(alpha[:,t]+beta[:,t]))))
			#total = 0
			# for i in range(self.number_states) : 
			# 	gamma[i,t] = alpha[i,t] + beta[i,t]
			# 	total = total + (alpha[i,t] + beta[i,t])
			# gamma[:,t] = gamma[:,t] - total
		
		for t in range(symbol_training_instance.shape[1]-1) :
			total = 0
			for i in range(self.number_states) : 
				for j in range(self.number_states) : 
					eta[i,j,t] = alpha[i,t] + self.log_A[j,i] + beta[j,t+1] + self.log_B[int(symbol_training_instance[0,t+1]),j] 		
					#total = total + np.exp(alpha[i,t] + self.log_A[i,j] + beta[j,t+1] + self.log_B[int(symbol_training_instance[0,t+1]),j] )		  			
			#eta[:,:,t] = eta[:,:,t] - np.log(total + 1e-300) 
			eta[:,:,t] = eta[:,:,t] - np.max(eta[:,:,t]) - np.log(np.sum(np.exp(eta[:,:,t] - np.max(eta[:,:,t])))) 
		
			# for i in range(self.number_states) : 
			# 	for j in range(self.number_states) : 
			# 		eta[i,j,t] = np.exp(alpha[i,t] + np.log(self.A[i,j] + 1e-300) + beta[j,t+1] + np.log(self.B[int(symbol_training_instance[0,t+1]),j] + 1e-300))		
			# 		total = total + np.exp(alpha[i,t]  + np.log(self.A[i,j] + 1e-300) + beta[j,t+1] + np.log(self.B[int(symbol_training_instance[0,t+1]),j] + 1e-300))		  			
			# # eta += ((eta < 1e-300) * 1e-300).astype(np.float64)
			# total += ((total < 1e-300) * 1e-300).astype(np.float64)
			# eta[:,:,t] = eta[:,:,t] / total 
		
		### Updating A,B,lambda


		self.log_pi_states = gamma[:,0].reshape((1,-1))

		for i in range(self.number_states) : 
			for j in range(self.number_states) : 
				numerator, denominator = 1e-300,1e-300
				for t in range(symbol_training_instance.shape[1]) : 
					numerator = numerator + np.exp(eta[i,j,t] - np.max(eta[i,j,:]))
					denominator = denominator + np.exp(gamma[i,t] - np.max(gamma[i,:]))	
				self.log_A[j,i] = np.max(eta[i,j,:]) + np.log(numerator) - np.max(gamma[i,:]) - np.log(denominator)	 			

		for i in range(self.number_states) : 
			for j in range(self.number_symbols) : 
				numerator,denominator = 1e-300,1e-300
				num_instances = np.where(j == symbol_training_instance)
				if len(num_instances[0]) != 0 :
					num_max = np.max(gamma[i,num_instances[1]])
				else : 
					num_max = 0	
				for t in range(symbol_training_instance.shape[1]) :
					denominator = denominator + np.exp(gamma[i,t] - np.max(gamma[i,:]))
					if j == symbol_training_instance[0,t] :
						numerator = numerator + np.exp(gamma[i,t] - num_max)
						
				self.log_B[j,i] = num_max + np.log(numerator) - np.max(gamma[i,:]) - np.log(denominator)	
		self.pi_states = self.log_pi_states
		self.A = self.log_A
		self.B = self.log_B			

	def test(self, symbol_test_instance ) :
		
		alpha = np.zeros((self.number_states, symbol_test_instance.shape[1]))
		for i in range(self.number_states) :
			alpha[i,0] = self.log_pi_states[0,i] + self.log_B[int(symbol_test_instance[0,0]),i]  
		for t in range(1,symbol_test_instance.shape[1]) : 
			for j in range(self.number_states) :
				#alpha[j,t] = self.log_B[int(symbol_training_instance[0,t]),j] + np.log(np.sum(np.exp(alpha[:,t-1].reshape((1,-1)) + self.log_A[:,j].reshape((1,-1)))) + 1e-300)
				alpha[j,t] = self.log_B[int(symbol_test_instance[0,t]),j] + np.max(alpha[:,t-1].reshape((1,-1)) + self.log_A[j,:].reshape((1,-1))) + np.log(np.sum(np.exp(alpha[:,t-1].reshape((1,-1)) + self.log_A[j,:].reshape((1,-1)) - np.max(alpha[:,t-1].reshape((1,-1)) + self.log_A[j,:].reshape((1,-1))))))

		# for i in range(1,symbol_test_instance.shape[1]) : 
		# 	for j in range(self.number_states) :
		# 		alpha[j,i] = np.log(self.B[int(symbol_test_instance[0,i]),j] + 1e-300) + np.log(np.dot(np.exp(alpha[:,i-1]).reshape((1,-1)),self.A[:,j].reshape((-1,1)))[0,0] + 1e-300)	

		return np.sum(alpha[:,-1])		