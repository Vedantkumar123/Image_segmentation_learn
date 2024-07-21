import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import random

img_width=128
img_height=128
img_channels=3
seed=42
np.random.seed=seed

#Build the model U-NET
inputs=tf.keras.layers.Input((img_width,img_height,img_channels))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#Contracting path
c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(s)
c1=tf.keras.layers.Dropout(0.1)(c1)
c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c1)
p1=tf.keras.layers.MaxPooling2D((2,2))(c1)

c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(p1)
c2=tf.keras.layers.Dropout(0.1)(c2)
c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c2)
p2=tf.keras.layers.MaxPooling2D((2,2))(c2)

c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(p2)
c3=tf.keras.layers.Dropout(0.2)(c3)
c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c3)
p3=tf.keras.layers.MaxPooling2D((2,2))(c3)

c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(p3)
c4=tf.keras.layers.Dropout(0.2)(c4)
c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c4)
p4=tf.keras.layers.MaxPooling2D((2,2))(c4)

c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(p4)
c5=tf.keras.layers.Dropout(0.3)(c5)
c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c5)
p5=tf.keras.layers.MaxPooling2D((2,2))(c5)

#Expansive path
u6=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
u6=tf.keras.layers.concatenate([u6,c4])
c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(u6)
c6=tf.keras.layers.Dropout(0.2)(c6)
c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c6)

u7=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6)
u7=tf.keras.layers.concatenate([u7,c3])
c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(u7)
c7=tf.keras.layers.Dropout(0.1)(c7)
c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c7)

u8=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7)
u8=tf.keras.layers.concatenate([u8,c2],axis=3)
c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(u8)
c8=tf.keras.layers.Dropout(0.1)(c8)
c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c8)

u9=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8)
u9=tf.keras.layers.concatenate([u9,c1])
c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(u9)
c9=tf.keras.layers.Dropout(0.1)(c9)
c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer="he_normal",padding='same')(c9)

outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c9)


model=tf.keras.Model(inputs=[inputs],outputs=[outputs])
model.compile(optimizer='adam',loss="binary_crossentropy z",metrics=['accuracy'])
model.summary()

train_path="C:/Users/KIIT/OneDrive/Desktop/Vedant_Official/vedant projects and works/Datasets/Data_science_bowl_2018/stage1_train/"
test_path="C:/Users/KIIT/OneDrive/Desktop/Vedant_Official/vedant projects and works/Datasets/Data_science_bowl_2018/stage1_test/"
train_ids=next(os.walk(train_path))[1]
test_ids=next(os.walk(test_path))[1]

x_train=np.zeros((len(train_ids),img_height,img_width,img_channels),dtype=np.uint8)
y_train=np.zeros((len(train_ids),img_height,img_width,1),dtype=bool)


print("resizing images and masks")
for n,id_ in tqdm(enumerate(train_ids),total=len(train_ids)):
  path = train_path+id_
  img = imread(path+'/images/'+id_+'.png')[:,:,:img_channels]
  img = resize(img,(img_height,img_width),mode='constant',preserve_range=True)
  x_train[n]=img
  mask=np.zeros((img_height,img_width,1),dtype=bool)
  for mask_file in next(os.walk(path+'/masks/'))[2]:
    mask_=imread(path+'/masks/'+mask_file)
    mask_=np.expand_dims(resize(mask_,(img_height,img_width),mode='constant',preserve_range=True),axis=-1)
    mask=np.maximum(mask,mask_)
  y_train[n]=mask


x_test=np.zeros((len(test_ids),img_height,img_width,img_channels),dtype=np.uint8)
sizes_test=[]
print("resizing test images")
for n,id_ in tqdm(enumerate(test_ids),total=len(test_ids)):
  path = test_path+id_
  img = imread(path+'/images/'+id_+'.png')[:,:,:img_channels]
  sizes_test.append([img.shape[0],img.shape[1]])
  img = resize(img,(img_height,img_width),mode='constant',preserve_range=True)
  x_test[n]=img


#model checkpoint
checkpointer=tf.keras.callbacks.ModelCheckpoint('Test_model.h5',verbose=1,save_best_only=True)
callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
           tf.keras.callbacks.TensorBoard(log_dir='dir')
           ]

results=model.fit(x_train,y_train,validation_split=0.1,batch_size=16,epochs=25,callbacks=callbacks)


idx=random.randint(0,len(x_train))
preds_train=model.predict(x_train[:int(x_train.shape[0]*0.9)],verbose=1)
preds_val=model.predict(x_train[int(x_train.shape[0]*0.9):],verbose=1)
preds_test=model.predict(x_test,verbose=1)

preds_train_t=(preds_train>0.5).astype(np.uint8)
preds_val_t=(preds_val>0.5).astype(np.uint8)
preds_test_t=(preds_test>0.5).astype(np.uint8)


#performing a sanity check on random training samples
ix=random.randint(0,len(preds_train_t))
imshow(x_train[ix])
plt.show()
imshow(np.squeeze(y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()


#performing a sanity check on random validation samples
ix=random.randint(0,len(preds_val_t))
imshow(x_train[int(x_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(y_train[int(y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()