# Character-Recognition-using-Keras-and-Flask
A simple web app which can recognize handwritten english character, using keras (with tensorflow backend)
 - note : You need to downlaod dataset from [here](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format) , extract .csv file and rename it as A_Z_Data.csv
 
 run "pip install -r requirements.txt" to install dependencies

to run the app, run following command :  uwsgi --socket 0.0.0.0:5000 --protocol=http -w wsgi:app
this will run app on port 5000

all source code and trained model is included...!!
