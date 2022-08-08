
"""
Segmentation MOdels library info:
#https://github.com/qubvel/segmentation_models
#Recommended for colab execution
TensorFlow ==2.1.0
keras ==2.3.1
pip install segmentation-models
For this demo it is working on a local workstation...
Python 3.5
TensorFlow ==1.
keras ==2
"""

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
from os import path
import numpy as np
from matplotlib import pyplot as plt

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


results_path="/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/Results" # path to store model results
if path.exists(results_path) == False:
  os.mkdir(results_path)

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 256 #Resize images (height  = X, width = Y)
SIZE_Y = 256

#Capture training image info as a list
train_images = []
img_paths = []
for directory_path in glob.glob("/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_AUG"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        #print(img_path)      
        img_paths.append(img_path)       

img_paths.sort()
for img_path in img_paths:
   img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)       
   img = cv2.resize(img, (SIZE_Y, SIZE_X))
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
   train_images.append(img)
   #train_labels.append(label)

#Convert list to array for machine learning processing        
train_images = np.array(train_images)


#Capture mask/label info as a list
train_masks = [] 
mask_paths = []

for directory_path in glob.glob("/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST_AUG"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):        
        mask_paths.append(mask_path)       

mask_paths.sort()
for mask_path in mask_paths:
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)       
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    train_masks.append(mask)
    #train_labels.append(label)

#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

#Use customary x_train and y_train variables
X = train_images
Y = train_masks
#Y = np.expand_dims(Y, axis=3) #May not be necessary.. leftover from previous code 


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

print(model.summary())


history=model.fit(x_train, 
          y_train,
          batch_size=32, 
          epochs=15,
          verbose=1,
          validation_data=(x_val, y_val))



#accuracy = model.evaluate(x_val, y_val)
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

os.chdir(results_path)

model.save('/content/TCC_Cell_Semantic_Segmentation/modelUNET.h5')


from tensorflow import keras
model = keras.models.load_model('/content/TCC_Cell_Semantic_Segmentation/modelUNET.h5', compile=False)

#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread('/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/02/t000.tif', cv2.IMREAD_UNCHANGED)       
test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

test_img = np.expand_dims(test_img, axis=0)


prediction = model.predict(test_img)

#View and Save segmented image
prediction_image = prediction.reshape([256,256,3])
new_img = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)
#cv2_imshow(new_img)
plt.imshow(new_img, cmap='gray')
output_path = results_path + 'test0_segmented.jpg'
plt.imsave(output_path, new_img, cmap='gray')