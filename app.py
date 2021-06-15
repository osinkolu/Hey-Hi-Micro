# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 03:31:02 2021

@author: Professor
"""

# first impor the needed flask libraries
from flask import Flask, request, render_template

# import my tools from fastai insead of the conventional pickle.

#from fastai import *
#import fastai
from fastai.vision.all import *

# import pathlib, i needed this during the test on my local pc.
import pathlib


# i intend to use Alexadr rybnikov's dnn face detector that he hid in opencv.
import cv2
d_image = "web_images/avatar.png"
image_path = os.path.join('static',d_image)

# quicly set poxipath for my testing.
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# just in case you like to know, i'm using fastai version 2.2.5

# start the backend.
app = Flask(__name__)


@app.route('/') # this means when i call the home page, by default it loads the home.
def home():
    return render_template('index.html',user_image=image_path)

#load in my models via fastai's load_learner method. 
#gender_model = load_learner('Gender_model/export.pkl')
#race_model = load_learner('Race_model/export.pkl')
#age_model = load_learner('Age_model/export.pkl')

# Time to infer from the model
@app.route('/predict',methods=['POST','GET'])
#Predict method that uses the trained model to predict the results for the picture we uploaded

def predict():
    file = request.files['file'] #get file
    #Store the uploaded images in a temporary folder
    if file:
        filename = file.filename
        file.save("static/"+filename)
        #file.save("temporary/"+filename)
        
        image_to_predict = "static/"+filename
        print(type(filename))
    
    # let's check if there's actually a face in the model.

        image = cv2.imread(image_to_predict) # give opencv the image.
        image_path = os.path.join("static",filename )
        print(image_path)
        prototxtPath = "face_detector/deploy.prototxt" #path to rybnikov's prototxtpath
        weightspath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"  #path to rybnikov's weightspath
        
        """ NOTE THAT YOUR ADDRESS OF PROTOTXTPATH AND WEIGHTSPATH SHOULD BE SIMILAR TO THAT OF OpenCV's I.E ...../face_detector/ the file to load"""
        
        #Setting up the network
        net = cv2.dnn.readNet(prototxtPath, weightspath)# opencv loads model up.
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300,300), (104.0,177.0,123.0)) #imagenet stats
        
        print("processor is computing object detections....")# lol, of course this is for my backend machine
        
        #give input to network
        net.setInput(blob)
        
        #foward detections
        detections = net.forward()

        confidence = detections[0][0][0][2]
        
        # 0.5 aint bad for the worst picture taken of the face, lol, so let's make 0.5 the threshold
        if confidence > 0.5:
            print("i found a face, and i'm {}% sure about it".format(confidence)) # for my backend.
            
            face_finder = "Yeah! i found your face in the image, and"
        else:
            face_finder = "I couldn't find your face in the image, but still"
            
            print(confidence)

        #Getting the prediction from the fastai models
        gender = gender_model.predict(image_to_predict)
        race = race_model.predict(image_to_predict)
        age =  age_model.predict(image_to_predict)
        
        return render_template('index.html', prediction_text="{} I predicted your gender as {} and your race as {} and your age should be around {} years give or take".format(face_finder,gender[0].title(),race[0].title(), int(age[1][0])),user_image=image_path)
    else:
        return render_template('index.html', prediction_text="I'm not sure you've selected any picture yet, i didnt find anything.", user_image=image_path)
# The guy that calls the whole shot!!"""
if __name__ =="__main__":
    app.run()
