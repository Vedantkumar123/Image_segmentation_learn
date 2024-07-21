import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
import glob 
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

backbone='resnet34'
preprocess_input=sm.get_preprocessing(backbone)

size_x=256
size_y=256

train_images=[]

for dir_path in glob.glob('train path'):
    for img_path in glob.glob(os.path.join(dir_path,"*.png")):
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        train_images.append(img)
train_images = np.array(train_images)

train_masks=[]

for dir_path in glob.glob('mask path'):
    for mask_path in glob.glob(os.path.join(dir_path,"*.png")):
        mask = cv2.imread(mask_path,0)
        train_masks.append(mask)

train_masks = np.array(train_masks)


x=train_images
y=train_masks
y=np.expand_dims(y,axis=3)


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42)

#preprocess input
x_train=preprocess_input(x_train)
x_val=preprocess_input(x_val)

model=sm.Unet(backbone,encoder_weights='imagenet')
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['mse'])

print(model.summary())

history=model.fit(
    x_train,
    y_train,
    batch_size=8,
    epochs=10,
    verbose=1,
    validation_data=(x_val,y_val)
)
loss=history.history('loss')
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'y',label="Training loss")
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.plot()


from tensorflow import keras
model=keras.models.load_model('membrane.h5',compile=False)

intersection = np.logical_and(y_test,y_pred)
union = np.logical_or(y_test,y_pred)
iou_score = np.sum(intersection)/np.sum(union)
print(iou_score)



