from VQ import * 
from util import *
from hmm_train import *

source = '/Users/kommi/Desktop/HMM/Isolated_Digits'

# weightsave = '/Users/kommi/Desktop/HMM/weights_10'
# if not os.path.exists(weightsave):
# 	os.makedirs(weightsave)

directories = ['1','2','8','o','z']
req_directories = []

for i in directories : 
	path_dir = source + '/' + i + '/dev/*.mfcc'
	[req_directories.append(j) for j in glob.glob(path_dir)]  

data_dev_points = [] 

for file in req_directories : 
	with open(file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		array_points = np.array(list_points_refined).reshape(rows,columns)
		for i in range(array_points.shape[0]) : 
			data_dev_points.append(list(array_points[i,:]))

iterations = 15
number_symbols = int(sys.argv[1])
dimensions = 38

data_dev_points = np.array(data_dev_points)
kmeans = VQ(dimensions,number_symbols)		

centroids = kmeans.train(data_dev_points,iterations)

states = int(sys.argv[2])
print("Number of states : ",states)
print("Number of symbols in each state : ",number_symbols)

weightsave = '/Users/kommi/Desktop/HMM/weights_log_'+sys.argv[1]+'_'+sys.argv[2]
if not os.path.exists(weightsave):
	os.makedirs(weightsave)
train_iterations = int(sys.argv[3])

np.savetxt(weightsave+'/centroids.txt',centroids)
############## DIGIT 1 #################
req_train_directories = []
directories = ['1'] 
for i in directories : 
	path_dir = source + '/' + i + '/train/*.mfcc'
	[req_train_directories.append(j) for j in glob.glob(path_dir)] 

train_instances = np.random.randn(1,38)
for file in req_train_directories : 
	with open(file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		array_points = np.array(list_points_refined).reshape(rows,columns)
		train_instances = np.concatenate((train_instances, array_points),axis = 0)
train_instances = [train_instances]

symbols_train_instances = []
for i in range(len(train_instances)) : 
	print(i)
	symbols = kmeans.assign_symbols(centroids,train_instances[i])
	symbols_train_instances.append(symbols)

hmm_1 = HMM(number_symbols,states)

for i in range(train_iterations) : 
	print('Training Started -------------------- Epoch Number : ',i)
	for j in range(len(symbols_train_instances)) :
		hmm_1.train(symbols_train_instances[j])
		
print('Training Completed for digit 1')	
newpath=weightsave+"/"+directories[0]
if not os.path.exists(newpath):
	os.makedirs(newpath)
file1=newpath+"/pi.txt"
np.savetxt(file1,hmm_1.log_pi_states)
file1=newpath+"/A.txt"
np.savetxt(file1,hmm_1.log_A)
file1=newpath+"/B.txt"
np.savetxt(file1,hmm_1.log_B)
################ DIGIT 2 ################

req_train_directories = []
directories = ['2'] 
for i in directories : 
	path_dir = source + '/' + i + '/train/*.mfcc'
	[req_train_directories.append(j) for j in glob.glob(path_dir)] 

train_instances = np.random.randn(1,38)
for file in req_train_directories : 
	with open(file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		array_points = np.array(list_points_refined).reshape(rows,columns)
		train_instances = np.concatenate((train_instances, array_points),axis = 0)
train_instances = [train_instances]

symbols_train_instances = []
for i in range(len(train_instances)) : 
	symbols = kmeans.assign_symbols(centroids,train_instances[i])
	symbols_train_instances.append(symbols)

hmm_2 = HMM(number_symbols,states)

for i in range(train_iterations) : 
	print('Training Started -------------------- Epoch Number : ',i)
	for j in range(len(symbols_train_instances)) :
		hmm_2.train(symbols_train_instances[j])
		
print('Training Completed for digit 2')
newpath=weightsave+"/"+directories[0]
if not os.path.exists(newpath):
	os.makedirs(newpath)
file1=newpath+"/pi.txt"
np.savetxt(file1,hmm_2.log_pi_states)
file1=newpath+"/A.txt"
np.savetxt(file1,hmm_2.log_A)
file1=newpath+"/B.txt"
np.savetxt(file1,hmm_2.log_B)
############ DIGIT 8 ####################

req_train_directories = []
directories = ['8'] 
for i in directories : 
	path_dir = source + '/' + i + '/train/*.mfcc'
	[req_train_directories.append(j) for j in glob.glob(path_dir)] 

train_instances = np.random.randn(1,38)
for file in req_train_directories : 
	with open(file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		array_points = np.array(list_points_refined).reshape(rows,columns)
		train_instances = np.concatenate((train_instances, array_points),axis = 0)
train_instances = [train_instances]

symbols_train_instances = []
for i in range(len(train_instances)) : 
	symbols = kmeans.assign_symbols(centroids,train_instances[i])
	symbols_train_instances.append(symbols)

hmm_8 = HMM(number_symbols,states)

for i in range(train_iterations) : 
	print('Training Started -------------------- Epoch Number : ',i)
	for j in range(len(symbols_train_instances)) :
		hmm_8.train(symbols_train_instances[j])
		
print('Training Completed for digit 8')	
newpath=weightsave+"/"+directories[0]
if not os.path.exists(newpath):
	os.makedirs(newpath)
file1=newpath+"/pi.txt"
np.savetxt(file1,hmm_8.log_pi_states)
file1=newpath+"/A.txt"
np.savetxt(file1,hmm_8.log_A)
file1=newpath+"/B.txt"
np.savetxt(file1,hmm_8.log_B)

############## DIGIT O #####################

req_train_directories = []
directories = ['o'] 
for i in directories : 
	path_dir = source + '/' + i + '/train/*.mfcc'
	[req_train_directories.append(j) for j in glob.glob(path_dir)] 

train_instances = np.random.randn(1,38)
for file in req_train_directories : 
	with open(file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		array_points = np.array(list_points_refined).reshape(rows,columns)
		train_instances = np.concatenate((train_instances, array_points),axis = 0)
train_instances = [train_instances]

symbols_train_instances = []
for i in range(len(train_instances)) : 
	symbols = kmeans.assign_symbols(centroids,train_instances[i])
	symbols_train_instances.append(symbols)

hmm_o = HMM(number_symbols,states)

for i in range(train_iterations) : 
	print('Training Started -------------------- Epoch Number : ',i)
	for j in range(len(symbols_train_instances)) :
		hmm_o.train(symbols_train_instances[j])
		
print('Training Completed for digit O')	
newpath=weightsave+"/"+directories[0]
if not os.path.exists(newpath):
	os.makedirs(newpath)
file1=newpath+"/pi.txt"
np.savetxt(file1,hmm_o.log_pi_states)
file1=newpath+"/A.txt"
np.savetxt(file1,hmm_o.log_A)
file1=newpath+"/B.txt"
np.savetxt(file1,hmm_o.log_B)
################## DIGIT Z #####################

req_train_directories = []
directories = ['z'] 
for i in directories : 
	path_dir = source + '/' + i + '/train/*.mfcc'
	[req_train_directories.append(j) for j in glob.glob(path_dir)] 

train_instances = np.random.randn(1,38)
for file in req_train_directories : 
	with open(file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		array_points = np.array(list_points_refined).reshape(rows,columns)
		train_instances = np.concatenate((train_instances, array_points),axis = 0)
train_instances = [train_instances]

symbols_train_instances = []
for i in range(len(train_instances)) : 
	symbols = kmeans.assign_symbols(centroids,train_instances[i])
	symbols_train_instances.append(symbols)

hmm_z = HMM(number_symbols,states)

for i in range(train_iterations) : 
	print('Training Started -------------------- Epoch Number : ',i)
	for j in range(len(symbols_train_instances)) :
		hmm_z.train(symbols_train_instances[j])
		
print('Training Completed for digit z')	
newpath=weightsave+"/"+directories[0]
if not os.path.exists(newpath):
	os.makedirs(newpath)
file1=newpath+"/pi.txt"
np.savetxt(file1,hmm_z.log_pi_states)
file1=newpath+"/A.txt"
np.savetxt(file1,hmm_z.log_A)
file1=newpath+"/B.txt"
np.savetxt(file1,hmm_z.log_B)
####################### TEST ######################

req_train_directories = []
true_digits = [] 

directories = ['1','2','8','o','z'] 
for i in directories : 
	path_dir = source + '/' + i + '/train/*.mfcc'
	[req_train_directories.append(j) for j in glob.glob(path_dir)] 
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

train_instance = None
predicted_digits = []
for test_file in req_train_directories :
	with open(test_file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		train_instance = np.array(list_points_refined).reshape(rows,columns)
		symbols_train_instances = kmeans.assign_symbols(centroids,train_instance)
		predicted = [hmm_1.test(symbols_train_instances) , hmm_2.test(symbols_train_instances) ,hmm_8.test(symbols_train_instances) ,hmm_o.test(symbols_train_instances) ,hmm_z.test(symbols_train_instances)]
		predicted_digits.append(predicted.index(max(predicted)))

array_true_digits = np.array(true_digits)				
array_predicted_digits = np.array(predicted_digits)

accuracy = np.sum(array_predicted_digits == array_true_digits)/len(true_digits)
print('Train Data accuracy : ',accuracy)

req_train_directories = []
true_digits = [] 

directories = ['1','2','8','o','z'] 
for i in directories : 
	path_dir = source + '/' + i + '/dev/*.mfcc'
	[req_train_directories.append(j) for j in glob.glob(path_dir)] 
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

train_instance = None
predicted_digits = []
for test_file in req_train_directories :
	with open(test_file,'r') as f : 
		data = f.read()
		list_points = data.split(' ')
		list_points = [i.strip() for i in list_points]

		rows,columns  = int(list_points[1]),int(list_points[0])
		list_points_refined = [ float(i) for i in list_points[2:]]
		train_instance = np.array(list_points_refined).reshape(rows,columns)
		symbols_train_instances = kmeans.assign_symbols(centroids,train_instance)
		predicted = [hmm_1.test(symbols_train_instances) , hmm_2.test(symbols_train_instances) ,hmm_8.test(symbols_train_instances) ,hmm_o.test(symbols_train_instances) ,hmm_z.test(symbols_train_instances)]
		predicted_digits.append(predicted.index(max(predicted)))

array_true_digits = np.array(true_digits)				
array_predicted_digits = np.array(predicted_digits)

accuracy = np.sum(array_predicted_digits == array_true_digits)/len(true_digits)
print('Development Data Accuracy : ',accuracy)
