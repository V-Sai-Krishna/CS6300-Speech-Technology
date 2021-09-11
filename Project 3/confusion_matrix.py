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

	number_A = np.zeros((int(number_states),int(number_states)))
	number_B = np.zeros((int(number_symbols),int(number_states)))
	number_pi = np.zeros(int(number_states))

	number_A[:int(number_states),:int(number_states)] = digit_A[digits.index(x)]
	number_B[:,:int(number_states)] = digit_B[digits.index(x)]
	
	number_pi[:int(number_states)] = digit_pi[digits.index(x)]

	number_list_A.append(number_A)
	number_list_B.append(number_B)
	number_list_pi.append(number_pi)
	

source_centroid = source + '/centroids.txt'
centroids = None
with open(source_centroid,'r') as f_centroid : 
	centroids = np.loadtxt(f_centroid)

req_test_directories = []
true_digits = [] 

test_source = '/Users/kommi/Desktop/HMM/Isolated_Digits/'
directories = ['1','2','8','o','z'] 
for i in directories : 
	path_dir = test_source + '/' + i + '/dev/*.mfcc'
	[req_test_directories.append(j) for j in glob.glob(path_dir)] 
	if i == '1' :	
		[true_digits.append(0) for j in glob.glob(path_dir)]
	elif i == '2' :
		[true_digits.append(1) for j in glob.glob(path_dir)]
	elif i == '8' :
		[true_digits.append(2) for j in glob.glob(path_dir)]
	elif i == 'o' : 
		[true_digits.append(3) for j in glob.glob(path_dir)]
	elif i == 'z' :
		[true_digits.append(4) for j in glob.glob(path_dir)]				 

confusion_matrix = np.zeros((5,5))
test_instance = None
predicted_digits = []
i = 0
for test_file in req_test_directories :
	with open(test_file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		test_instance = np.array(list_points_refined).reshape(rows,columns)
		symbols_test_instances = assign_symbols(centroids,test_instance)
		predicted = [test(number_list_pi[0],number_list_A[0],number_list_B[0],int(number_states),symbols_test_instances) , test(number_list_pi[1],number_list_A[1],number_list_B[1],int(number_states),symbols_test_instances) ,test(number_list_pi[2],number_list_A[2],number_list_B[2],int(number_states),symbols_test_instances) ,test(number_list_pi[3],number_list_A[3],number_list_B[3],int(number_states),symbols_test_instances) ,test(number_list_pi[4],number_list_A[4],number_list_B[4],int(number_states),symbols_test_instances)]
		predicted_digits.append(predicted.index(max(predicted)))
		confusion_matrix[predicted.index(max(predicted)),true_digits[i]] = confusion_matrix[predicted.index(max(predicted)),true_digits[i]] + 1
	i = i+ 1	
array_true_digits = np.array(true_digits)				
array_predicted_digits = np.array(predicted_digits)

accuracy = np.sum(array_predicted_digits == array_true_digits)/len(true_digits)
print('Development Accuracy : ',accuracy)
print(confusion_matrix)
newpath = '/Users/kommi/Desktop/HMM/confusion_matrix_dev'
if not os.path.exists(newpath):
	os.makedirs(newpath)
np.savetxt(newpath+'/'+str(number_states)+'_'+str(number_symbols)+'.txt',confusion_matrix)
req_test_directories = []
true_digits = [] 

test_source = '/Users/kommi/Desktop/HMM/Isolated_Digits/'
directories = ['1','2','8','o','z'] 
for i in directories : 
	path_dir = test_source + '/' + i + '/train/*.mfcc'
	[req_test_directories.append(j) for j in glob.glob(path_dir)] 
	if i == '1' :	
		[true_digits.append(0) for j in glob.glob(path_dir)]
	elif i == '2' :
		[true_digits.append(1) for j in glob.glob(path_dir)]
	elif i == '8' :
		[true_digits.append(2) for j in glob.glob(path_dir)]
	elif i == 'o' : 
		[true_digits.append(3) for j in glob.glob(path_dir)]
	elif i == 'z' :
		[true_digits.append(4) for j in glob.glob(path_dir)]				 

confusion_matrix = np.zeros((5,5))
test_instance = None
predicted_digits = []
i=0
for test_file in req_test_directories :
	with open(test_file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		test_instance = np.array(list_points_refined).reshape(rows,columns)
		symbols_test_instances = assign_symbols(centroids,test_instance)
		predicted = [test(number_list_pi[0],number_list_A[0],number_list_B[0],int(number_states),symbols_test_instances) , test(number_list_pi[1],number_list_A[1],number_list_B[1],int(number_states),symbols_test_instances) ,test(number_list_pi[2],number_list_A[2],number_list_B[2],int(number_states),symbols_test_instances) ,test(number_list_pi[3],number_list_A[3],number_list_B[3],int(number_states),symbols_test_instances) ,test(number_list_pi[4],number_list_A[4],number_list_B[4],int(number_states),symbols_test_instances)]
		predicted_digits.append(predicted.index(max(predicted)))
		confusion_matrix[predicted.index(max(predicted)),true_digits[i]] = confusion_matrix[predicted.index(max(predicted)),true_digits[i]] + 1
	i = i+1 	
array_true_digits = np.array(true_digits)				
array_predicted_digits = np.array(predicted_digits)
# print(array_true_digits)
# print(array_predicted_digits)

accuracy = np.sum(array_predicted_digits == array_true_digits)/len(true_digits)
print('Train Accuracy : ',accuracy)
print(confusion_matrix)
newpath = '/Users/kommi/Desktop/HMM/confusion_matrix_train'
if not os.path.exists(newpath):
	os.makedirs(newpath)
np.savetxt(newpath+'/'+str(number_states)+'_'+str(number_symbols)+'.txt',confusion_matrix)