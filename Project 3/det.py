import os
import numpy as np 
import matplotlib.pyplot as plt

source = '/Users/kommi/Desktop/HMM/roc_train'
number_states_list = [15,15,15,15]
number_symbols_list = [30,50,70,90]

plt.figure(1)
false_positive_list = []
true_positive_list = []

for j in range(len(number_states_list)) :

	with open(source+'/false_positive_'+str(number_states_list[j])+'_'+str(number_symbols_list[j])+'.txt','r') as f : 
		false_positive = np.loadtxt(f)
		false_positive_list.append(false_positive)
	with open(source+'/true_positive_'+str(number_states_list[j])+'_'+str(number_symbols_list[j])+'.txt','r') as f :	
		true_positive = np.loadtxt(f)
		true_positive_list.append(true_positive)

for j in range(len(true_positive_list)) :
	plt.plot(false_positive_list[j],true_positive_list[j])
plt.title('Combined ROC curve on train data for 15 state isolated HMMs')
plt.legend(('30 symbols','50 symbols','70 symbols','90 symbols'))
plt.xlabel('False Positive Rate ----->')
plt.ylabel('True Positive Rate ----->')
plt.grid()
plt.show()	

plt.figure(2)
for j in range(len(true_positive_list)) :
	plt.loglog(false_positive_list[j],1-true_positive_list[j])
plt.title('Combined ROC curve on train data for 15 state isolated HMMs')
plt.legend(('30 symbols','50 symbols','70 symbols','90 symbols'))
plt.xlabel('False Positive Rate ----->')
plt.ylabel('False Negative Rate ----->')
plt.grid()
plt.show()		