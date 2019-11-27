import pandas
import numpy as np
from io import StringIO
import time



filename = 'ann-test.data'
names = ['colom1', 'colom2', 'colom3', 'colom4', 'colom5', 'colom6', 'colom7', 'colom8', 'colom9', 'colom10',
         'colom11', 'colom12', 'colom13', 'colom14', 'colom15', 'colom16', 'colom17', 'colom18', 'colom19', 'colom20', 'colom21','classLabel']



data = pandas.read_csv(filename, names=names, delim_whitespace=True)

data_array = np.array(data)    #convert the pandas to numpy array


training_inputs = np.array(data_array[:, :21])				#take all features from data except result column

training_outputs = np.array(data_array[:,21]).T-1		        # take result(class) column

np.random.seed(42)

feature_set = np.array(training_inputs)

labels = np.array(training_outputs, dtype=int)


one_hot_labels = np.zeros((feature_set.shape[0], 3))



for i in range(feature_set.shape[0]):
    one_hot_labels[i, labels[i]] = 1


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
	correct = 0
	length = 0
	for i in range(len(actual)):
		length += 1
		for j in range(len(actual[i])):
			if predicted[i][j] >= 0.50:
				predicted[i][j] = 1
			if actual[i][j] == predicted[i][j]:
				correct += 1
	return correct / float(length) * 100.0

def accuracy_class1(actual, predicted):
	correct_class = 0
	class_length = 0
	for i in range(len(actual)):
		for j in range(len(actual[i])):
			if actual[i][0] == 1:
				class_length +=1
			if predicted[i][0] >= 0.56:
				predicted[i][0] = 1
			if actual[i][0] == predicted[i][0]:
				correct_class +=1
	accuracy_class = correct_class / float(class_length) * 100.0
	return accuracy_class

def accuracy_class2(actual, predicted):
	correct_class = 0
	class_length = 0
	for i in range(len(actual)):
		for j in range(len(actual[i])):
			if actual[i][1] == 1:
				class_length +=1
			if predicted[i][1] >= 0.56:
				predicted[i][1] = 1
			if actual[i][1] == predicted[i][1]:
				correct_class +=1
	accuracy_class = correct_class / float(class_length) * 100.0
	return accuracy_class

def accuracy_class3(actual, predicted):
	correct_class = 0
	class_length = 0
	for i in range(len(actual)):
		for j in range(len(actual[i])):
			if actual[i][2] == 1:
				class_length +=1
			if predicted[i][2] >= 0.56:
				predicted[i][2] = 1
			if actual[i][2] == predicted[i][2]:
				correct_class +=1
	accuracy_class = correct_class / float(class_length) * 100.0
	return accuracy_class

wh = np.loadtxt("weightsHidden.txt")
wo = np.loadtxt("weightsOutput.txt")
bh = np.loadtxt("biasHidden.txt")
bo = np.loadtxt("biassOutput.txt")

zh = np.dot(feature_set, wh) + bh  #output of hidden layer after multiplication
ah = sigmoid(zh)  #take sigmoid of result

    # Phase 2
zo = np.dot(ah, wo) + bo    #hidden layers weihgt*a0 (values of aoutput layer)
ao = softmax(zo)

result = softmax(ao)

acc=accuracy_metric(one_hot_labels, softmax(ao))
acc_class1=accuracy_class1(one_hot_labels, softmax(ao))
acc_class2=accuracy_class2(one_hot_labels, softmax(ao))
acc_class3=accuracy_class3(one_hot_labels, softmax(ao))

np.savetxt('result.txt', result, fmt='%f')
print(acc)
print("Accuracy class1: ", acc_class1)
print("Accuracy class2: ", acc_class2)
print("Accuracy class3: ", acc_class3)



