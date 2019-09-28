from flask import Flask, render_template,request
import cv2
import numpy as np
import re
import base64
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

def init(): 
	global sess
	global graph

	sess = tf.Session()
	set_session(sess)
	loaded_model = load_model("model.h5")
	print("Loaded Model from disk")

	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	graph = tf.get_default_graph()
	return loaded_model,graph

model, graph = init()

app = Flask(__name__)

labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

@app.route('/')
def index():
	return render_template("index.html")

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

@app.route('/predict/',methods=['GET','POST'])
def predict():
	global sess
	global graph
	imgData = request.get_data()
	convertImage(imgData)
	x = cv2.imread('output.png',0) #0 for single channel read
	x = np.invert(x)
	x = cv2.resize(x,(28,28))/255
	x = x.reshape(1,28,28,1)

	with graph.as_default():
		set_session(sess)
		out = model.predict(x)
		pred = int(np.argmax(out,axis=1))
		response = labels[pred] +" (" +  str(round(out[0][pred] * 100,2)) + " %)"
		print(response)
		return response

    
if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=8888)