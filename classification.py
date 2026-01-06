"""
Horses and humans image classification
By Ali Al Nasseri
This documentation will help with using this code.
The following code has 3 different models and each model is commented out.
I have provided comments indicating where each model starts and ends.
You can uncomment any model to use it.
These models were made in visual studio code so there might be minor adjustments needed if you are using google colab or jupyter notebook.
The main difference in code that I have noticed is in the output layer where in google colab i needed change it to the following:

model.add(Dense(1,activation='sigmoid'))

Any other issues might be fixed by installing some libraries that are not pre-installed in google colab or jupyter notebook.
I am using this code in a virtual environment so some libraries might need to be installed manually.
the dataset can be found at the following link: https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset
Important note: This code will not compile unless one of the models is uncommented.
"""

# importing necessary libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,UpSampling2D, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# defining directories
trainDir = "train"
testDir = "test"

# checking number of images in each category
print(len(os.listdir("train/horses")))
print(len(os.listdir("train/humans")))
print(len(os.listdir("test/horses")))
print(len(os.listdir("test/humans")))

# training data augmentation methods
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

#model 1 code - start
"""
model = Sequential()

# loading training data
train_data = train_datagen.flow_from_directory(
    trainDir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# validation data augmentation methods
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# loading validation data
validation_data = validation_datagen.flow_from_directory(
    trainDir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#creating the CNN model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(128,128,3))) # input layer 32 filters
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid')) # output layer

# compiling the model
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

# fitting the model
history = model.fit(train_data, epochs=20, validation_data=validation_data)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    testDir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

# loading predictions
predictions = model.predict(test_data)
"""
#model 1 code - end


#model 2 code - start
"""
model = Sequential()

# loading training data
train_data = train_datagen.flow_from_directory(
    trainDir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# validation data augmentation methods
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# loading validation data
validation_data = validation_datagen.flow_from_directory(
    trainDir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#creating the CNN model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(128,128,3))) # input layer 32 filters
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid')) # output layer

# compiling the model
model.compile(optimizer=Adam(learning_rate=0.005),loss='binary_crossentropy',metrics=['accuracy'])

# fitting the model
history = model.fit(train_data, epochs=20, validation_data=validation_data)


test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    testDir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

# loading predictions
predictions = model.predict(test_data)
"""
#model 2 code - end

#model 3 code - start
"""
# loading training data
train_data = train_datagen.flow_from_directory(
    trainDir,
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# validation data augmentation methods
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# loading validation data
validation_data = validation_datagen.flow_from_directory(
    trainDir,
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# useing transfer learning with ResNet50
resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(
    include_top = False, # using my own images so keep it False
    input_shape = (256,256,3),
    pooling = 'max', classes = 2,
    weights = 'imagenet')

for layer in pretrained_model.layers:
    layer.trainable = False # disable training for pre-trained layers to keep weights unchanged

# adding layers to the model
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation = 'relu'))
resnet_model.add(Dense(1, activation = 'sigmoid'))

# compiling the model
resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

# fitting the model
history = resnet_model.fit(train_data, epochs=20, validation_data=validation_data)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    testDir,
    target_size=(256,256),
    batch_size=32,
    class_mode='binary'
)

# loading predictions
predictions = resnet_model.predict(test_data)
"""
#model 3 code - end

#plotting - start
"""

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

"""
#plotting - end

true_labels = test_data.classes
predicted_labels = (predictions > 0.5).astype(int)

cm = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(true_labels, predicted_labels))
