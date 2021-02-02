from flask import Flask, render_template,request
#import cv2
from PIL import Image
import numpy as np
import re
import base64
import os
from tensorflow.keras.models import load_model


model = load_model("model.h5")
print("Loaded Model from disk")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

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

@app.route('/predict/',methods=['POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)
	#x = cv2.imread('output.png',0) #0 for single channel read
	img = Image.open('output.png').convert('L')
	wpercent = (28/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	x = np.array(img.resize((28,hsize), Image.ANTIALIAS))
	x = np.invert(x)
	x = x.reshape(28,28)/255
	x = x.reshape(1,28,28,1)



	out = model.predict(x)
	pred = int(np.argmax(out,axis=1))
	response = labels[pred] +" (" +  str(round(out[0][pred] * 100,2)) + " %)"
	print(response)
	return response

"""    
if __name__ == '__main__':  
	#port = int(os.environ.get('PORT', 8888))
	port = 80
	app.run(debug=True, port=port)
"""
