import matplotlib.pyplot as plt

symbols = [30,50,70,90]
accuracy_test = [0.75,0.88,0.83,0.81]
accuracy_train = [0.86,0.96,0.97,0.97]

plt.figure(0)
plt.plot(symbols,accuracy_train)
plt.plot(symbols,accuracy_test)
plt.grid()
plt.title('Accuracy vs Symbols Plot for 15 state HMM')
plt.xlabel('Number of symbols in one state ----->')
plt.ylabel('Accuracy ----->')
plt.legend(('Accuracy on training data','Accuracy on development data'))
plt.show()