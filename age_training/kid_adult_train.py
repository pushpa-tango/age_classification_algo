from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, AveragePooling2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# K.set_image_dim_ordering('th')
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (128,128,3), activation = 'relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# Adding a third convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# Adding a fourth convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# Adding a fifth convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),  strides=(2,2), padding='same', data_format='channels_last'))

# Adding a sixth convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# Seventh layer
classifier.add(Convolution2D(32, (3, 3), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# # Eighth layer
# classifier.add(Convolution2D(32, (3, 3), padding='same', activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# # Nineth layer
# classifier.add(Convolution2D(32, (3, 3), padding='same', activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# #Tenth layer
# classifier.add(Convolution2D(32, (3, 3), padding='same', activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2), padding='same', data_format='channels_last'))

# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.save('model/kid_adult_model_arch.txt')

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/dataset/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
 
test_set = test_datagen.flow_from_directory('/dataset/test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (5400/32),
                         epochs = 300,
                         validation_data = test_set,
                         validation_steps = (1080/32))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

mc = ModelCheckpoint('model/kid_adult_weights.h5', monitor='val_loss', mode='min', save_best_only=True)
# classifier.save('retrain2_model_arch.h5')
classifier.save_weights('model/kid_adult_weights.h5')
print('Saved')

