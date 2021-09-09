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

predicted_3_digit_list = []
score_3_digit_list = []
for j in range(len(list_symbols)) : 
	test_instance_part1 = list_symbols[j][0][0:int(len(list_symbols[j][0])/3)].reshape(1,-1)
	test_instance_part2 = list_symbols[j][0][int(len(list_symbols[j][0])/3):int(2*len(list_symbols[j][0])/3)].reshape(1,-1)	
	test_instance_part3 = list_symbols[j][0][int(2*len(list_symbols[j][0])/3):-1].reshape(1,-1)

	predicted_digit1_index = [test(digit_pi[0],digit_A[0],digit_B[0],int(number_states),test_instance_part1),test(digit_pi[1],digit_A[1],digit_B[1],int(number_states),test_instance_part1),test(digit_pi[2],digit_A[2],digit_B[2],int(number_states),test_instance_part1),test(digit_pi[3],digit_A[3],digit_B[3],int(number_states),test_instance_part1),test(digit_pi[4],digit_A[4],digit_B[4],int(number_states),test_instance_part1)]			
	predicted_digit2_index = [test(digit_pi[0],digit_A[0],digit_B[0],int(number_states),test_instance_part2),test(digit_pi[1],digit_A[1],digit_B[1],int(number_states),test_instance_part2),test(digit_pi[2],digit_A[2],digit_B[2],int(number_states),test_instance_part2),test(digit_pi[3],digit_A[3],digit_B[3],int(number_states),test_instance_part2),test(digit_pi[4],digit_A[4],digit_B[4],int(number_states),test_instance_part2)]			
	predicted_digit3_index = [test(digit_pi[0],digit_A[0],digit_B[0],int(number_states),test_instance_part3),test(digit_pi[1],digit_A[1],digit_B[1],int(number_states),test_instance_part3),test(digit_pi[2],digit_A[2],digit_B[2],int(number_states),test_instance_part3),test(digit_pi[3],digit_A[3],digit_B[3],int(number_states),test_instance_part3),test(digit_pi[4],digit_A[4],digit_B[4],int(number_states),test_instance_part3)]

	predicted_digit1 = digits[predicted_digit1_index.index(max(predicted_digit1_index))]			
	predicted_digit2 = digits[predicted_digit2_index.index(max(predicted_digit2_index))]
	predicted_digit3 = digits[predicted_digit3_index.index(max(predicted_digit3_index))]

	predicted_3_digit_list.append(predicted_digit1+predicted_digit2+predicted_digit3)
	score_3_digit_list.append((max(predicted_digit1_index)+max(predicted_digit2_index)+max(predicted_digit3_index))/3)

predicted_2_digit_list = []
score_2_digit_list = []
for j in range(len(list_symbols)) : 
	test_instance_part1 = list_symbols[j][0][0:int(len(list_symbols[j][0])/2)].reshape(1,-1)
	test_instance_part2 = list_symbols[j][0][int(len(list_symbols[j][0])/2):-1].reshape(1,-1)	

	predicted_digit1_index = [test(digit_pi[0],digit_A[0],digit_B[0],int(number_states),test_instance_part1),test(digit_pi[1],digit_A[1],digit_B[1],int(number_states),test_instance_part1),test(digit_pi[2],digit_A[2],digit_B[2],int(number_states),test_instance_part1),test(digit_pi[3],digit_A[3],digit_B[3],int(number_states),test_instance_part1),test(digit_pi[4],digit_A[4],digit_B[4],int(number_states),test_instance_part1)]			
	predicted_digit2_index = [test(digit_pi[0],digit_A[0],digit_B[0],int(number_states),test_instance_part2),test(digit_pi[1],digit_A[1],digit_B[1],int(number_states),test_instance_part2),test(digit_pi[2],digit_A[2],digit_B[2],int(number_states),test_instance_part2),test(digit_pi[3],digit_A[3],digit_B[3],int(number_states),test_instance_part2),test(digit_pi[4],digit_A[4],digit_B[4],int(number_states),test_instance_part2)]			


	predicted_digit1 = digits[predicted_digit1_index.index(max(predicted_digit1_index))]			
	predicted_digit2 = digits[predicted_digit2_index.index(max(predicted_digit2_index))]

	predicted_2_digit_list.append(predicted_digit1+predicted_digit2)
	score_2_digit_list.append((max(predicted_digit1_index)+max(predicted_digit2_index))/2)

predicted_1_digit_list = []
score_1_digit_list = []

for j in range(len(list_symbols)) : 
	test_instance_part1 = list_symbols[j][0].reshape(1,-1)
	

	predicted_digit1_index = [test(digit_pi[0],digit_A[0],digit_B[0],int(number_states),test_instance_part1),test(digit_pi[1],digit_A[1],digit_B[1],int(number_states),test_instance_part1),test(digit_pi[2],digit_A[2],digit_B[2],int(number_states),test_instance_part1),test(digit_pi[3],digit_A[3],digit_B[3],int(number_states),test_instance_part1),test(digit_pi[4],digit_A[4],digit_B[4],int(number_states),test_instance_part1)]			

	predicted_digit1 = digits[predicted_digit1_index.index(max(predicted_digit1_index))]			

	predicted_1_digit_list.append(predicted_digit1)
	score_1_digit_list.append((max(predicted_digit1_index)))

for j in range(len(list_symbols)) :
	print('Actual Number : ',test_numbers[j])
	score_list = [score_1_digit_list[j],score_2_digit_list[j],score_3_digit_list[j]]
	digit_predicted = [predicted_1_digit_list[j],predicted_2_digit_list[j],predicted_3_digit_list[j]]
	print('Predicted Number : ',digit_predicted[score_list.index(max(score_list))])
	# if score_2_digit_list[j] > score_3_digit_list[j] : 
	# 	print('Predicted Number : ',predicted_2_digit_list[j])
	# else : 	
	# 	print('Predicted Number : ',predicted_3_digit_list[j])