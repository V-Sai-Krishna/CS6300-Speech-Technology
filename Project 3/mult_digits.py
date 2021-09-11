import numpy as np 
import matplotlib.pyplot as plt
import sys
from util import *
from VQ_custom import * 

def test(pi,A,B,number_states, symbol_test_instance ) :
		
		alpha = np.zeros((number_states, symbol_test_instance.shape[1]))
		for i in range(number_states) :
			alpha[i,0] = pi[i] + B[int(symbol_test_instance[0,0]),i]  
		for t in range(1,symbol_test_instance.shape[1]) : 
			for j in range(number_states) :
				alpha[j,t] = B[int(symbol_test_instance[0,t]),j] + np.max(alpha[:,t-1].reshape((1,-1)) + A[:,j].reshape((1,-1))) + np.log(np.sum(np.exp(alpha[:,t-1].reshape((1,-1)) + A[:,j].reshape((1,-1)) - np.max(alpha[:,t-1].reshape((1,-1)) + A[:,j].reshape((1,-1))))))		
		return np.sum(alpha[:,-1])					


number_states = sys.argv[1]
number_symbols = sys.argv[2]

print('number of states : ', number_states)
print('number of symbols : ',number_symbols)

source = '/Users/kommi/Desktop/HMM/'+'weights_log_'+number_symbols+'_'+number_states

digits = ['1','2','8','o','z']
numbers_list = []
number_list_A = []
number_list_B = []
number_list_pi = []

digit_A = []
digit_B = []
digit_pi = []

for i in digits : 
	source_digit = source + '/' + i + '/'
	with open(source_digit + 'A.txt', 'r') as f_A : 
		A = np.loadtxt(f_A)
		digit_A.append(A)
	with open(source_digit + 'B.txt', 'r') as f_B : 
		B = np.loadtxt(f_B)
		digit_B.append(B)
	with open(source_digit + 'pi.txt', 'r') as f_pi : 	
		pi = np.loadtxt(f_pi)
		digit_pi.append(pi)
print(digit_pi[0].shape)

for x in digits : 
	number = x 
	numbers_list.append(number)

	#number_A = np.zeros((int(number_states),int(number_states)))
	number_A = np.full((int(number_states),int(number_states)),-300)
	#number_B = np.zeros((int(number_symbols),int(number_states)))
	number_B = np.full((int(number_symbols),int(number_states)),-300)
	#number_pi = np.zeros(int(number_states))
	number_pi = np.full(int(number_states),-300)

	number_A[:int(number_states),:int(number_states)] = digit_A[digits.index(x)]
	number_B[:,:int(number_states)] = digit_B[digits.index(x)]
	


	number_pi[:int(number_states)] = digit_pi[digits.index(x)]

	number_list_A.append(number_A)
	number_list_B.append(number_B)
	number_list_pi.append(number_pi)
	for y in digits : 
		number = x + y 
		numbers_list.append(number)

		# number_A = np.zeros((2*int(number_states),2*int(number_states)))
		# number_B = np.zeros((int(number_symbols), 2*int(number_states)))
		# number_pi = np.zeros(2*int(number_states))

		number_A = np.full((2*int(number_states),2*int(number_states)),-300)
		number_B = np.full((int(number_symbols), 2*int(number_states)),-300)
		number_pi = np.full(2*int(number_states),-300)

		number_A[:int(number_states),:int(number_states)] = digit_A[digits.index(x)]
		number_A[int(number_states)-1,int(number_states)] = -1.609
		number_A[int(number_states)-1,int(number_states)-1] = -0.2231
		number_A[int(number_states):2*int(number_states),int(number_states):2*int(number_states)] = digit_A[digits.index(y)]

		number_B[:,:int(number_states)] = digit_B[digits.index(x)]
		number_B[:,int(number_states):2*int(number_states)] = digit_B[digits.index(y)]


		number_pi[:int(number_states)] = digit_pi[digits.index(x)]

		number_list_A.append(number_A)
		number_list_B.append(number_B)
		number_list_pi.append(number_pi)
		for z in digits : 

			number = x+y+z
			numbers_list.append(number)

			# number_A = np.zeros((3*int(number_states),3*int(number_states)))
			# number_B = np.zeros((int(number_symbols), 3*int(number_states)))
			# number_pi = np.zeros(3*int(number_states))

			number_A = np.full((3*int(number_states),3*int(number_states)),-300)
			number_B = np.full((int(number_symbols), 3*int(number_states)),-300)
			number_pi = np.full(3*int(number_states),-300)

			number_A[:int(number_states),:int(number_states)] = digit_A[digits.index(x)]
			number_A[int(number_states)-1,int(number_states)] = -1.609
			number_A[int(number_states)-1,int(number_states)-1] = -0.2231
			
			number_A[int(number_states):2*int(number_states),int(number_states):2*int(number_states)] = digit_A[digits.index(y)]
			number_A[int(number_states)-1,int(number_states)] = -1.609
			number_A[int(number_states)-1,int(number_states)-1] = -0.2231
			number_A[2*int(number_states):3*int(number_states),2*int(number_states):3*int(number_states)] = digit_A[digits.index(z)]

			number_B[:,:int(number_states)] = digit_B[digits.index(x)]
			number_B[:,int(number_states):2*int(number_states)] = digit_B[digits.index(y)]
			number_B[:,2*int(number_states):3*int(number_states)] = digit_B[digits.index(z)]

			number_pi[:int(number_states)] = digit_pi[digits.index(x)]

			number_list_A.append(number_A)
			number_list_B.append(number_B)
			number_list_pi.append(number_pi)

source_centroid = source + '/centroids.txt'
centroids = None
with open(source_centroid,'r') as f_centroid : 
	centroids = np.loadtxt(f_centroid)

test_files = []
path_dir = '/Users/kommi/Desktop/HMM/test/*.mfcc'  
[test_files.append(j) for j in glob.glob(path_dir)] 

list_symbols = []
test_numbers = []
for k in test_files : 
	test_instances = np.random.randn(1,38)
	test_numbers.append(k.split('/')[-1].split('.')[0])
	with open(k,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		array_points = np.array(list_points_refined).reshape(rows,columns)
		test_instances = np.concatenate((test_instances, array_points),axis = 0)

	symbols = assign_symbols(centroids,test_instances)
	list_symbols.append(symbols)

for j in range(len(list_symbols)) : 
	prob_list_number = [] 
	numbers_list_copy = numbers_list.copy()
	for i in range(len(numbers_list)) : 
		prob_list_number.append(test(number_list_pi[i],number_list_A[i],number_list_B[i],number_list_A[i].shape[0],list_symbols[j]))
	print('Actual : ',test_numbers[j])
	
	print('Predicted 1 : ',numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	numbers_list_copy.remove(numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	prob_list_number.remove(max(prob_list_number))
	
	print('Predicted 2 : ',numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	numbers_list_copy.remove(numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	prob_list_number.remove(max(prob_list_number))

	print('Predicted 3 : ',numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	numbers_list_copy.remove(numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	prob_list_number.remove(max(prob_list_number))

	print('Predicted 4 : ',numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	numbers_list_copy.remove(numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	prob_list_number.remove(max(prob_list_number))

	print('Predicted 5 : ',numbers_list_copy[prob_list_number.index(max(prob_list_number))])
	print()
