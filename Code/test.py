import pyaudio
import wave
import pydub
import itertools
import random
from pydub import AudioSegment
import time
from pydub.silence import split_on_silence
import keras
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model


import librosa

import warnings
warnings.filterwarnings("ignore")

model=load_model('C:/collegework/Ethnicity_DL/Training/weights-improvement-cnn.hdf5')
#model=load_model('C:/collegework/Ethnicity_DL/Training/weights-improvement-ann.hdf5')
adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999, epsilon=None,decay=0.0,amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

p="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

data_path = "C:/collegework/Ethnicity_DL/Test_images"



image_data_list=[]
size_data=[]
lengths=[]


image_list=os.listdir(data_path)
lengths.append(len(image_list))    
for image in image_list:        
    img = cv2.imread(data_path+'/'+image,0)
    image = cv2.resize(img, (100, 100))
    rects = detector(img, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
    shape=list(shape)            
    image_data_list.append(np.array(shape))

size_data.append(len(image_list))
print(image_data_list[0].shape)
image_data = np.array(image_data_list)
image_data = image_data.astype('float32')
image_data /= 256
image_data=np.expand_dims(image_data, axis=1)
    
(model.predict(image_data))
g=(model.predict_classes(image_data))[0]

if int(g)==0:
   print("The person is African")
elif int(g)==1:
   print("The person is Hispanic")  
elif int(g)==2:
   print("The person is Caucasian")
elif int(g)==3:
   print("The person is Dravidian") 
else:
   print("The person is Mongoloid")


    
    





