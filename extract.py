import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

def Extract(file,splitFact):
	print("Beginning the extraction process...")
	with open(file,'r') as f:
		data = f.read()
		data = data.split('\n')
	
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	
	data = shuffle(data)
	
	l = len(data)
	
	trainData = int(l * (1-splitFact))
	testData = int(l - trainData)
	
	print("Dividing data into {} training and {} testing samples..".format(trainData,testData))
	
	for i in range(trainData):
		if data[i] != '':
			x_train.append(np.array(data[i].split(',')[1:], dtype=np.float32))
			y_train.append(np.array(data[i].split(',')[0], dtype=np.int8))

	for i in range(testData):
		if data[i+trainData] != '':
			x_test.append(np.array(data[i+trainData].split(',')[1:], dtype=np.float32))
			y_test.append(np.array(data[i+trainData].split(',')[0], dtype=np.int8))
			
	x_train = np.array(x_train)/255
	y_train = to_categorical(np.array(y_train))
	x_test = np.array(x_test)/255
	y_test = to_categorical(np.array(y_test))
	
	del data
	
	return (x_train,y_train),(x_test,y_test)
	