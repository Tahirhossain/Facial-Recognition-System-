import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split


EPOCHS=10



model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
#

# In[5]:


# Load VGG Face model weights
#model.load_weights('vgg_face_weights.h5')
# model.load_weights('colab_Best_Weights_VGG16_.h5')
# model=load_model('colab_Best_Weights_VGG16_.h5')
#model= load_model('lt_face_classifier_model.h5')
model= load_model("colab_Best_Weights_VGG16_.h5")


#model.summary()


# In[7]:


# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)


# In[9]:


#Prepare Training Data
import os
x_=[]
y_=[]
person_folders=os.listdir('dataset_face/')
person_rep=dict()
for i,person in enumerate(person_folders):
  person_rep[i]=person
  image_names=os.listdir('dataset_face/'+person+'/')
  print('Loading data of: ',person)
  for image_name in tqdm(image_names):
    img=load_img('dataset_face/'+person+'/'+image_name,target_size=(224,224))
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    img_encode=vgg_face(img)
    x_.append(np.squeeze(K.eval(img_encode)).tolist())
    y_.append(i)




x_train=np.array(x_)
y_train=np.array(y_)

# Save test and train data for later use
np.save('_data',x_)
np.save('_labels',y_)
#
#
# # In[15]:


# Load saved data
x_=np.load('_data.npy')
y_=np.load('_labels.npy')
#
x_train,x_test,y_train,y_test=train_test_split(x_,y_,test_size=0.20, stratify=y_, random_state=42)
print('Shaper of train data:',x_train.shape)
print('Shaper of test data:',x_test.shape)


# In[16]:
from tensorflow.keras import layers
from tensorflow.keras import initializers


# kernel_initializer='glorot_uniform'
# Softmax regressor to classify images based on encoding 
classifier_model=Sequential()
classifier_model.add(Dense(units=500,input_dim=x_train.shape[1],kernel_initializer=initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Zeros()))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('relu'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=300,kernel_initializer=initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Zeros()))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('relu'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=150,kernel_initializer=initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Zeros()))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('relu'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=75,kernel_initializer=initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Zeros()))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('relu'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=20,kernel_initializer=initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Zeros()))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('relu'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=10 ,kernel_initializer=initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Zeros()))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='adam',metrics=['accuracy'])
classifier_model.summary()


# In[17]:


H=classifier_model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_test,y_test))


# In[19]:


# Save model for later use
tf.keras.models.save_model(classifier_model,'lt2_face_classifier_model.h5')



print("[INFO] saving mask detector model...")
import matplotlib.pyplot as plt
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy and Val_Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_acc.png")
plt.show()


N = EPOCHS
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Val_Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot_loss.png")
plt.show()
