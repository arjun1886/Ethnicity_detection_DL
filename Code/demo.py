from imutils import face_utils
import dlib
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D
from keras.optimizers import adam
import keras
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

p="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

data_path = "C:/collegework/Ethnicity_DL/Frontalized/"
data_dir_list = os.listdir(data_path)
print(data_dir_list)
map=[[1 for i in range(2)] for j in range(5)]

for i in range(len(data_dir_list)):
    if data_dir_list[i]=='AFRICAN':
       map[i][0]='AFRICAN'
       map[i][1]=0
    elif data_dir_list[i]=='HISPANIC':
       map[i][0]='HISPANIC'
       map[i][1]=1
    elif data_dir_list[i]=='CAUCASIAN':
       map[i][0]='CAUCASSIAN'
       map[i][1]=2
    elif data_dir_list[i]=='DRAVIDIAN':
       map[i][0]='DRAVIDIAN'
       map[i][1]=3
    else:
       map[i][0]='MONGOLOID'
       map[i][1]=4
       
num_classes = 5
image_data_list=[]
size_data=[]
lengths=[]

for dataset in data_dir_list:
    image_list=os.listdir(data_path+'/'+ dataset)
    lengths.append(len(image_list))    
    for image in image_list:        
        img = cv2.imread(data_path+'/'+ dataset +'/'+image,0)
        img = cv2.resize(img, (200, 200))
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
print(image_data.shape)
num_classes = 5
num_of_samples = image_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

i=1
while(i<5):
     lengths[i]=lengths[i]+lengths[i-1]
     i+=1

print(lengths)

for k in range(0,lengths[0]):
    labels[k]=map[0][1]


i=1
while(i<5):
     for j in range(lengths[i-1],lengths[i]):
         labels[j]=map[i][1]
     i+=1

Y= np_utils.to_categorical(labels,5)
x,y = shuffle(image_data,Y, random_state=5)

X_train,X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=4)
X_train,X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=4) 
shape=X_train[0].shape
print(shape)

model = Sequential()
model.add( Conv2D(64, (1,1), input_shape=shape, name="input"))
model.add(Activation('relu'))
model.add( Conv2D(64, (1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(5,activation='softmax',name='op'))

adam=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

filepath = 'C:/collegework/Ethnicity_DL/training/weights-improvement-cnn-frontalized.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(X_train,y_train, epochs=2000, batch_size=32, callbacks=callback_list, validation_data=(X_val,y_val))

score = model.evaluate(X_test, y_test, verbose=0)
cnn_score=score[1]
print('the testing accuracy is',score[1])
test_image = X_test

(model.predict(test_image))
g=model.predict_classes(test_image)

del model

model = Sequential()
model.add(Dense(32, input_shape=shape, name="input"))
model.add(Dropout(0.33))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Dropout(0.33))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.33))
model.add(Dense(5,activation='softmax',name='op'))

adam=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

filepath = 'C:/collegework/Ethnicity_DL/training/weights-improvement-ann.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(X_train,y_train, epochs=2000, batch_size=32, callbacks=callback_list, validation_data=(X_val,y_val))

score = model.evaluate(X_test, y_test, verbose=0)
ann_score=score[1]
print('the testing accuracy is',ann_score)
test_image = X_test

(model.predict(test_image))
print("the predicted classes by ANN model",model.predict_classes(test_image))



del model


print("the predicted classes by CNN model",g)

print('the testing accuracy is for cnn model is:',cnn_score)
print('the testing accuracy is for ann model is:',ann_score)
