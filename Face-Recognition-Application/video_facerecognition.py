
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
model = torch.hub.load('ultralytics/yolov5', 'custom', path=saved_model_path)
facenetmodel= load_model('models/facenet_keras.h5')
label_decoder = joblib.load('models/label_encoder-combined.joblib')
svmmodel = pickle.load(open('models/finalized_model_combined.sav', 'rb'))

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2


#Generalize the data and extract the embeddings
def extract_embeddings(model,face_pixels):
  face_pixels = face_pixels.astype('float32')  #convert the entire data to float32(base)
  mean = face_pixels.mean()                    #evaluate the mean of the data
  std  = face_pixels.std()                     #evaluate the standard deviation of the data
  face_pixels = (face_pixels - mean)/std
  samples = expand_dims(face_pixels,axis=0)    #expand the dimension of data
  yhat = model.predict(samples)
  return yhat[0]




# To capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    # # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = [i for i in model(gray, 416).xyxy[0].tolist()]
    print('faces',len(faces))
    if len(faces)==0:
        faces = [[25,0,25,10,0,0]]
    # Draw the rectangle around each face
    try:
        for x, y, x2, y2,m,col in faces:
                if col==1:
                    store_face = img[int(y)-20:int(y2)+5,int(x)-20:int(x2)+10]
                    image1 = Image.fromarray(store_face,'RGB')
                    image1 = image1.resize((160,160))             #resize the image
                    # face_array = asarray(image1)
                    testx = asarray(image1)
                    testx = testx.reshape(-1,160,160,3)
                    new_testx = asarray([extract_embeddings(facenetmodel,test_pixels) for test_pixels in testx])
                    in_encode = Normalizer(norm='l2')
                    new_testx = in_encode.transform(new_testx)
                    predict_test = svmmodel.predict(new_testx)
                    if len(predict_test)!=0:
                        predict_test = label_decoder.inverse_transform(predict_test)[0]
                    else:
                        predict_test = "Could not recognize"
                    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, predict_test+' Masked ', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,cv2.LINE_AA)

                else:
                    store_face = img[int(y)-20:int(y2)+5,int(x)-20:int(x2)+10]
                    image1 = Image.fromarray(store_face,'RGB')
                    image1 = image1.resize((160,160))             #resize the image
                    # face_array = asarray(image1)
                    testx = asarray(image1)
                    testx = testx.reshape(-1,160,160,3)
                    new_testx = asarray([extract_embeddings(facenetmodel,test_pixels) for test_pixels in testx])
                    in_encode = Normalizer(norm='l2')
                    new_testx = in_encode.transform(new_testx)
                    predict_test = svmmodel.predict(new_testx)
                    if len(predict_test)!=0:
                        predict_test = label_decoder.inverse_transform(predict_test)[0]
                    else:
                        predict_test = "Could not recognize"
                    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img, predict_test+' Not Masked', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,cv2.LINE_AA)
    except:
        pass
    # # Display
    cv2.imshow('img', img)
    out = cv2.VideoWriter('recorded_videos/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (416,416))
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()