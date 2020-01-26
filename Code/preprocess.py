import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os

this_path = os.path.dirname(os.path.abspath(__file__))

def demo(path):
    
    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
    
    img = cv2.imread(path, 1)
    
    lmarks = feature_detection.get_landmarks(img)
    
    if(len(lmarks)>0):
       proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
    
       eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    
       frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
       cv2.imwrite(path,frontal_raw)
    


import os
import warnings
warnings.filterwarnings("ignore")

data_path = "C:/collegework/Ethnicity_DL/Original"
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
c=0
image_data_list=[]
size_data=[]
lengths=[]
 
for dataset in data_dir_list:
    image_list=os.listdir(data_path+'/'+ dataset)
    
    lengths.append(len(image_list))
        

    for image in image_list:
        c+=1
        print(c)
        demo(data_path+'/'+ dataset +'/'+image)
