
import cv2
import torchvision.models as models
from PIL import Image
import numpy as np
import torch
from keras.models import load_model
import joblib
import pickle
from PIL import Image
from numpy import asarray,array,expand_dims,reshape,load,max
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


# Load the model
device = torch.device("cpu")
saved_model_path = "models/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=saved_model_path, force_reload=True)
facenetmodel= load_model('models/facenet_keras.h5')
label_decoder = joblib.load('models/label_encoder-combined.joblib')
svmmodel = pickle.load(open('models/finalized_model_combined.sav', 'rb'))
label_decoder_mask = joblib.load('models/label_encoder-combined.joblib')
svmmodel_mask = pickle.load(open('models/finalized_model_combined.sav', 'rb'))


#Generalize the data and extract the embeddings
def extract_embeddings(model,face_pixels):
  face_pixels = face_pixels.astype('float32')  #convert the entire data to float32(base)
  mean = face_pixels.mean()                    #evaluate the mean of the data
  std  = face_pixels.std()                     #evaluate the standard deviation of the data
  face_pixels = (face_pixels - mean)/std
  samples = expand_dims(face_pixels,axis=0)    #expand the dimension of data
  yhat = model.predict(samples)
  return yhat[0]



# Read the frame
img = cv2.imread('test_images/unnimasked3.jpg')
cv2.imshow('orig_img',img)
cv2.waitKey(0)
# # Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect the faces
faces = [i for i in model(gray, 416).xyxy[0].tolist()]
if len(faces)==0:
    faces = [[0,0,0,0,0,0]]
# Draw the rectangle around each face
for x, y, x2, y2,m,col in faces:
        if col==1:
            store_face = img[int(y)-20:int(y2)+5,int(x)-20:int(x2)+10]
            cv2.imshow('extracted_imag',store_face)
            cv2.waitKey(0)
            image1 = Image.fromarray(store_face,'RGB')
            image1 = image1.resize((160,160))             #resize the image
            # face_array = asarray(image1)
            testx = asarray(image1)
            testx = testx.reshape(-1,160,160,3)
            new_testx = asarray([extract_embeddings(facenetmodel,test_pixels) for test_pixels in testx])
            in_encode = Normalizer(norm='l2')
            new_testx = in_encode.transform(new_testx)
            predict_test = svmmodel_mask.predict(new_testx)
            print(predict_test)
            if len(predict_test)!=0:
                predict_test = label_decoder_mask.inverse_transform(predict_test)[0]
            else:
                predict_test = "Could not recognize"
            cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, predict_test+' Masked ', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,cv2.LINE_AA)
            # cv2.putText(img, 'Masked ', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,cv2.LINE_AA)

        else:
            store_face = img[int(y)-20:int(y2)+5,int(x)-20:int(x2)+10]
            cv2.imshow('extracted_imag',store_face)
            cv2.waitKey(0)
            image1 = Image.fromarray(store_face,'RGB')
            image1 = image1.resize((160,160))             #resize the image
            # face_array = asarray(image1)
            testx = asarray(image1)
            testx = testx.reshape(-1,160,160,3)
            new_testx = asarray([extract_embeddings(facenetmodel,test_pixels) for test_pixels in testx])
            in_encode = Normalizer(norm='l2')
            new_testx = in_encode.transform(new_testx)
            predict_test = svmmodel.predict(new_testx)
            print(predict_test)
            if len(predict_test)!=0:
                predict_test = label_decoder.inverse_transform(predict_test)[0]
            else:
                predict_test = "Could not recognize"
            cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, predict_test+' Not Masked', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,cv2.LINE_AA)
            # cv2.putText(img, 'Not Masked', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,cv2.LINE_AA)


# # Display
cv2.imshow('img', img)
# Stop if escape key is pressed
cv2.waitKey(0)