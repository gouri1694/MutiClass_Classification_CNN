import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation,Dropout
from tensorflow.keras.layers import Conv3D, MaxPooling3D,Conv2D, MaxPooling2D, LSTM, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers

# from keras.utils.np_utils import to_categorical
# y = to_categorical(y,num_classes = 5)
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (128,128,3)))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))
# Adding a second convolutional layer
classifier.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.3))
# 3rd Convolutional Layer
classifier.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.3))
# 4th Convolutional Layer
classifier.add(Conv2D(filters=256,kernel_size = (3,3),padding="Same",activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))
# 5th Convolutional Layer
classifier.add(Conv2D(filters=512,kernel_size = (3,3),padding="Same",activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.3))

classifier.add(Flatten())
# 1st Fully Connected Layer
classifier.add(Dense(1024,activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(BatchNormalization())
# Add output layer
classifier.add(Dense(5,activation="softmax"))

classifier.summary() # print summary my model
classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Train,val,test split
# import splitfolders
# splitfolders.ratio(r"C:\Users\DELL\Documents\CNN\CustomImageClassification\flowerRecognition\flowers", output=r"C:\Users\DELL\Documents\CNN\CustomImageClassification\flowerRecognition/dataset_flowers", seed=1337, ratio=(.7, .2, .1), group_prefix=None, move=False)
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\DELL\Documents\CNN\CustomImageClassification\flowerRecognition/dataset_flowers/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'C:\Users\DELL\Documents\CNN\CustomImageClassification\flowerRecognition/dataset_flowers/val',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 300,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save("model.h5")
print("Saved model to disk")



