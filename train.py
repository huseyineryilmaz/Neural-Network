import pandas
import numpy as np
from io import StringIO
import time



filename = 'ann-train.data'
names = ['colom1', 'colom2', 'colom3', 'colom4', 'colom5', 'colom6', 'colom7', 'colom8', 'colom9', 'colom10',
         'colom11', 'colom12', 'colom13', 'colom14', 'colom15', 'colom16', 'colom17', 'colom18', 'colom19', 'colom20', 'colom21','classLabel']

data = pandas.read_csv(filename, names=names, delim_whitespace=True)    #Reads all parameters from dataset file respect to column names


data_array = np.array(data)    #convert the pandas to numpy array


training_inputs = np.array(data_array[:, :21])				#take all features from data except result classLabels

training_outputs = np.array(data_array[:,21]).T-1		        # take classLabels from daataset

np.random.seed(42)

feature_set = np.array(training_inputs)

labels = np.array(training_outputs, dtype=int)


one_hot_labels = np.zeros((feature_set.shape[0], 3))			#creates a one_hot_label array  #one_hot_label array using for multi-class ann problems so it a good choose for our problem



for i in range(feature_set.shape[0]):
    one_hot_labels[i, labels[i]] = 1					#implement one_hot_label array



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
			if predicted[i][j] >= 0.56:
				predicted[i][j] = 1
			if actual[i][j] == predicted[i][j]:
				correct += 1
	return correct / float(length) * 100.0


instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 40
output_labels = 3

wh = np.random.rand(attributes,hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)
bo = np.random.randn(output_labels)
lr = 10e-4      # learning rate it equals to 0.001

error_cost = []

t0 = time.time()


for epoch in range(50000):
############# feedforward

    # Phase 1
    zh = np.dot(feature_set, wh) + bh  #output of hidden layer after multiplication
    ah = sigmoid(zh)  #take sigmoid of result

    # Phase 2
    zo = np.dot(ah, wo) + bo    #hidden layers weihgt*a0 (values of aoutput layer)
    ao = softmax(zo)
########## Back Propagation

########## Phase 1

    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)
    

    acc = 0
    if epoch == 49999:
        acc=accuracy_metric(one_hot_labels, softmax(ao))	
        np.savetxt('weightsHidden.txt', wh, fmt='%f')
        np.savetxt('weightsOutput.txt', wo, fmt='%f')
        np.savetxt('biasHidden.txt', bh, fmt='%f')
        np.savetxt('biassOutput.txt', bo, fmt='%f')
        print('Hidden layer node number: ', hidden_nodes)
        print('Threshold: ', 0.56)
        print('accuracy= ', acc)
        print('Loss function value: ', loss)


    if epoch % 200 == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        error_cost.append(loss)

    
t1 = time.time()

total = t1-t0

def convert(seconds): 
    min, sec = divmod(seconds, 60) 
    hour, min = divmod(min, 60) 
    return "%d:%02d:%02d" % (hour, min, sec) 
      

print("Algorithm complated in " + str(convert(total)) + " minutes (hour:min:sec)")
