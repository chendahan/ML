
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2
import os
from PIL import Image
import scipy.io
import plotly.graph_objects as go
import random, re, math
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf, tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""# Get Data"""

#images
!wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
!tar zxvf 102flowers.tgz

#labels as int
!wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

#Image segmentations
!wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz
!tar zxvf 102segmentations.tgz

"""# Preprocessing

## Map Value To Flower Name
"""

#load labels as numbers
mat = scipy.io.loadmat('imagelabels.mat')
#labels map - number to name 
CLASSES = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily',
          'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 
          'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',
          'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 
          'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 
          'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 
          'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
          'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura',
          'geranium', 'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
          'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania',
          'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 
          'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 
          'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia',
          'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']   
#map photo name to flower name 
photo_to_flower={}
for filename in sorted(glob.glob("jpg/*.jpg")):
  photo_to_flower[filename]=CLASSES[mat['labels'][0][int(filename.split('_')[1].split('.')[0])-1]-1]

#create one hot label for each image and split to train test using stratify 
_images = []       
_labels = []
for item in photo_to_flower.items():
  filepath=item[0]
  label=item[1]
  _images.append(filepath)
  _labels.append(label)
      
# Converting labels into One Hot matrix
train_labels_onehot = pd.get_dummies(_labels).values


# Splitting data into train and test dataset
train_images,test_images,train_labels,test_labels = train_test_split(_images,train_labels_onehot,test_size=0.4,stratify=train_labels_onehot)
# Splitting Training data into train and val dataset
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.35,stratify=train_labels )

#randomly add for each image in train data augmentation/segmentations(augmentation- vertical flip/horizontal flip/random noise)
!mkdir augm
seg_images=[]
seg_labels=[]
for idx,(img,label) in enumerate(zip(train_images,train_labels)):
  img_imread = cv2.imread(img)
  rand=random.randint(1, 4)
  #augmentation horizontal flip
  if(rand == 1):
    img_h_f=np.fliplr(img_imread)
    path='augm/h_f_'+img.split('/')[1]
    cv2.imwrite(path,img_h_f)
  #augmentation vertical flip
  elif(rand == 2):
    img_v_f=np.flipud(img_imread)
    path='augm/h_v_'+img.split('/')[1]
    cv2.imwrite(path,img_v_f)
  #random noise
  elif(rand == 3):
    noise_img=random_noise(img_imread,mode='s&p', seed=None, clip=True)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    path='augm/noise_'+img.split('/')[1]
    cv2.imwrite(path,noise_img)
  #segmentations
  elif(rand == 4):
    path='segmim/segmim_'+img.split('/')[1].split('_')[1]
  seg_images.append(path)
  seg_labels.append(label)
  #orig img
  seg_images.append(img)
  seg_labels.append(label)
train_images=seg_images
train_labels=seg_labels

#image path to cv2
def process_data(images,labels):
  shape = (224,224)  
  X_=[]
  y_=[]
  for idx,(img,label) in enumerate(zip(images,labels)):
    img = cv2.imread(img)
    img=cv2.resize(img,shape)#Resize all images to shape
    img=cv2.normalize(img, None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)# normalize 
    X_.append(img)
    y_.append(label)
  X_ = np.array(X_)
  y_ = np.array(y_)
  return X_,y_

#process train+val+test- resize images according to the network input size 
X_train,y_train=process_data(train_images,train_labels)
X_val,y_val=process_data(val_images,val_labels)
X_test,y_test=process_data(test_images,test_labels)

#create a model
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(224, 224, 3))
#Freeze the first layer
feature_extractor.trainable = False
dense=layers.Dense(102, activation='softmax')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Attach a classification head
model = tf.keras.Sequential([
  feature_extractor,
  dense])

model.compile(
              loss='categorical_crossentropy', 
              metrics=['acc'],
              optimizer='adam'
             )
model.summary()

epochs=15
# Training the model
history = model.fit(X_train,y_train,epochs=epochs,batch_size=34,validation_data=(X_val,y_val),callbacks=[early_stopping])

#eval Test data - loss: 0.3630 - acc: 0.8999
model.evaluate(X_test,y_test)

#accuracy : train + val line 
epochs_=list(range(1,epochs+1))

train_acc = history.history['acc']
val_acc = history.history['val_acc']

plt.clf()
fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs_,
                    y=train_acc,
                    name='Train'))

fig.add_trace(go.Scatter(x=epochs_,
                    y=val_acc,
                    name='val'))

fig.update_layout(height=500, 
                  width=700,
                  title='Accuracy',
                  xaxis_title='Epoch',
                  yaxis_title='Accuracy')
fig.show()

#crossentropy loss : train + val line 
crossentropy_loss_train = history.history['loss']
crossentropy_loss_val = history.history['val_loss']

plt.clf()
fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs_,
                    y=crossentropy_loss_train,
                    name='Train'))

fig.add_trace(go.Scatter(x=epochs_,
                    y=crossentropy_loss_val,
                    name='val'))

fig.update_layout(height=500, 
                  width=700,
                  title='Crossentropy',
                  xaxis_title='Epoch',
                  yaxis_title='crossentropy')
fig.show()

#fine-tuning
# Unfreeze the base model
feature_extractor.trainable = True
dense.trainable=False
model.compile(optimizer=keras.optimizers.Adam(1e-5), #low learning rate
              loss='categorical_crossentropy', 
              metrics=['acc']
             )

history_fine_tuning = model.fit(X_train,y_train,epochs=20,batch_size=34,validation_data=(X_val,y_val),callbacks=[early_stopping])

model.evaluate(X_test,y_test)

"""## Train Test Split"""

# # TODO: Create a pipeline for each set.
# IMAGE_RES = 224

# def format_image(image, label):
#   image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
#   return image, label

# BATCH_SIZE = 32

# train_batches = training_set.cache().shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

# validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

# test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

# #split data to folders according to flower type 
# from shutil import copyfile
# os.mkdir('data/')
# for i in photo_to_flower.items():
#   name= i[1].replace(" ", "")
#   path='data/'+name
#   if not os.path.isdir(path):
#     os.mkdir(path)
#   copyfile(i[0], path+'/'+i[0].split('/')[1])

# # !pip install tf-nightly
# #train test split 
# image_size = (180, 180)
# batch_size = 32

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "data",
#     validation_split=0.2,
#     subset="training",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size,
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "data",
#     validation_split=0.2,
#     subset="validation",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size,
# )
