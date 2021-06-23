# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:37:18 2021

@author: Professor
"""

# first import the needed fastapi libraries
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Unlike Flask, Fastai needs uvicorn to handle server stuffs
import uvicorn

# Never import * when deploying a code, I'm doing this cos this code is just for fun, only import what you need.
from fastai.vision.all import *
from io import BytesIO
#import cv2

# import pathlib, i needed this during the test on my local pc.
#import pathlib
# quicly set poxipath for my testing.
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath



# start the backend, an intersting mix of Fastapi and Jinja 2.
app = FastAPI() # call fastapi
app.mount("/static", StaticFiles(directory="static"), name="static") #we'll need fastapi to locate the static folder, that's where important stuff is.
templates = Jinja2Templates(directory="templates") # Jinja 2 needs to know where the HTML code is


@app.get('/', response_class = HTMLResponse) # this means when i call the home page, by default it loads the home.
async def index(request:Request):
    #return ("This guy works well")
    return templates.TemplateResponse("index.html", {"request": request, "user_image":"/static/web_images/avatar.png"})

#load in my models via fastai's load_learner method. 
gender_model = load_learner('Gender_model/export.pkl')
race_model = load_learner('Race_model/export.pkl')
age_model = load_learner('Age_model/export.pkl')

# Time to infer from the model
@app.post('/predict',response_class = HTMLResponse)
async def predict(request:Request, file:UploadFile = File(...)):
    if file.content_type[:5] == "image":
        
        picture = file.file.read()
        image = np.array(Image.open(BytesIO(picture)))
        prototxtPath = "face_detector/deploy.prototxt" #path to rybnikov's prototxtpath
        weightspath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"  #path to rybnikov's weightspath
            
        """ NOTE THAT YOUR ADDRESS OF PROTOTXTPATH AND WEIGHTSPATH SHOULD BE SIMILAR TO THAT OF OpenCV's I.E ...../face_detector/ the file to load"""
            
        #Setting up the network
        import cv2
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
        image_path = "static/temporary.jpg"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, image) #i'll have to save the image temporarily to display in my HTML code
        
        #Getting the prediction from the fastai models
        gender = gender_model.predict(picture)
        race = race_model.predict(picture)
        age =  age_model.predict(picture)
            
        
        return templates.TemplateResponse("index.html", {"request": request,"prediction_text": "{} I predicted your gender as {} and your race as {} and your age should be around {} years give or take".format(face_finder,gender[0].title(),race[0].title(), int(age[1][0])),"user_image":image_path})
    else:
        return templates.TemplateResponse('index.html', {"request": request,"prediction_text":"I'm not sure you've selected any picture yet, i didn't find anything.", "user_image":"/static/web_images/avatar.png"})
        
# LET UVICORN RUN THE WHOLE THING.

if __name__ == '__main__':
    uvicorn.run(app, host = "0.0.0.0", port = 5000)
