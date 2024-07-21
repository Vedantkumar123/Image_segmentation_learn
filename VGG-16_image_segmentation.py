import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

from keras.models import Sequential,Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16
# model=Unet(backbone_name='resnet',encoder_weights='imagenet',classes=3,activation='softmax',input_shape=(None,None,6))
# model=Unet(backbone_name='resnet34',encoder_weights='imagenet',encoder_freeze=True)
# model.compile('Adam','binary_crossentropy',['binary_accuracy'])

# print(os.lisdir('images/'))
size_x=1024
size_y=996

train_images=[]

for directory_path in glob.glob("images/train_images"):
  for img_path in glob.glob(os.path.join(directory_path,"*.tif")):
    img =cv2.imread(img_path,cv2.IMREAD_COLOR)
    img =cv2.resize(img,(size_x,size_y))
    img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    train_images.append(img)

train_images=np.array(train_images)

train_masks=[]
for directory_path in glob.glob("images/train_masks"):
  for mask_path in glob.glob(os.path.join(directory_path,"*.tif")):
    mask =cv2.imread(mask_path,0)
    mask =cv2.resize(mask,(size_x,size_y))
    # mask =cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    train_images.append(mask)

train_masks=np.array(train_masks)

x_train=train_images
y_train=train_masks
y_train=np.expand_dims(y_train,axis=3)

vgg_model=VGG16(weights="imagenet",include_top = False,input_shape=(size_x,size_y,3))

for layer in vgg_model.layers:
  layer.trainable=False

vgg_model.summary()

new_model=Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block1_conv2').output)
new_model.summary()
features = new_model.predict(x_train)
square = 8
ix=1
for _ in range(square):
  for _ in range(square):
    ax=plt.subpplot(square,square,ix)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(features[0,:,:,ix-1],cmap='gray')
    ix+=1
plt.show()    

x=features
x=x.reshape(-1,x.shape[3])
y=y_train.reshape(-1)
dataset=pd.Dataframe(x)
dataset['label']=y
print(dataset['label'].unique())
print(dataset['label'].value_counts())
dataset = dataset[dataset['label']!=0]
x_for_rf=dataset.drop(labels=['label'],axis=1)
y_for_rf=dataset['label']

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50,random_state=42)
model.fit(x_for_rf,y_for_rf)
filename='rf_model.sav'
pickle.dump(model,open(filename,'wb'))