from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path)
    im=convert_image_to_array(img)
    np_image_li = np.array(im, dtype=np.float16) / 225.0
    npp_image = np.expand_dims(np_image_li, axis=0)
    # Preprocessing the image
   # x = image.img_to_array(img)
    #np_image_li = np.array(x, dtype=np.float16) / 225.0
    #npp_image = np.expand_dims(np_image_li, axis=0)
    # x = np.true_divide(x, 255)
   # x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(npp_image)
    if np.argmax(preds)==0:
        preds = "A"
    elif np.argmax(preds)==1:
        preds = "B"
    else :
        preds = "C"
    return preds

def convert_image_to_array(img):
    try:
        image = cv2.imread(img)
        if image is not None :
            image = cv2.resize(image, target_size=(224, 224))   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        #label_list=['Bacterialblight' 'Blast' 'Brownspot' 'Tungro']
        preds = model_predict(file_path, model)
        
        #result = np.where(preds==np.max(preds))
        #print("probability:"+str(np.max(result))+"\n"+label_list_[itemindex[1][0]])

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

