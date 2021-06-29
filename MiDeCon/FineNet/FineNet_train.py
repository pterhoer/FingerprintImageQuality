"""Code for FineNet in paper "Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge" at ICB 2018
  https://arxiv.org/pdf/1712.09401.pdf

  If you use whole or partial function in this code, please cite paper:

  @inproceedings{Nguyen_MinutiaeNet,
    author    = {Dinh-Luan Nguyen and Kai Cao and Anil K. Jain},
    title     = {Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge},
    booktitle = {The 11th International Conference on Biometrics, 2018},
    year      = {2018},
    }
"""

import sys,os
sys.path.append(os.path.realpath('../FineNet'))

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from FineNet_model import FineNetmodel, plot_confusion_matrix
from keras.layers.core import Dropout, Dense
from keras.models import Model, save_model
from tensorflow.keras.applications.resnet50 import preprocess_input

import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime


#os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['KERAS_BACKEND'] = 'tensorflow'


output_dir = '../output_FineNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), output_dir)
log_dir = os.path.join(os.getcwd(), output_dir + '/logs')

# Training parameters
batch_size = 128
epochs = 80
num_classes = 2

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model size, patch
model_type = '_dropout_'


# =============== DATA loading ========================

train_path = '../Dataset/train/'
test_path = '../Dataset/test/'

input_shape = (160, 160, 3)

# Using data augmentation technique for training
datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=180,
        # randomly shift images horizontally
        width_shift_range=0.5,
        # randomly shift images vertically
        height_shift_range=0.5,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True)

datagen1 = ImageDataGenerator( rescale=None,
      preprocessing_function = preprocess_input)
datagen2 = ImageDataGenerator(rescale=1. / 255)

train_batches = datagen.flow_from_directory(train_path, target_size=(input_shape[0], input_shape[1]), classes=['minu', 'non_minu'], batch_size=batch_size)

#get number of samples
count = 0
for root, dirs, files in os.walk(train_path):
    for each in files:
        count += 1

# Feed data from directory into batches
test_gen = ImageDataGenerator()
test_batches = test_gen.flow_from_directory(test_path, target_size=(input_shape[0], input_shape[1]), classes=['minu', 'non_minu'], batch_size=batch_size)

#get number of validation samples
count1 = 0
for root, dirs, files in os.walk(test_path):
    for each in files:
        count1 += 1

# =============== end DATA loading ========================

# original: 30, 60, 150, 180
epo = [30, 60, 150, 180]
def lr_schedule(epoch):
    """Learning Rate Schedule
    """
    lr = 1e-3
    if epoch > epo[3]:
        lr *= 1e-7
    elif epoch > epo[2]:
        lr *= 1e-6
    elif epoch > epo[1]:
        lr *= 1e-5
    elif epoch > epo[0]:
        lr *= 1e-4
    print('Learning rate: ', lr)
    return lr




#============== Define model ==================

model = FineNetmodel(num_classes = num_classes,
                     pretrained_path = '../Models/FineNet.h5',
                                          input_shape=input_shape)

drop_rate = 0.3
# add Dropout
dense = model.layers[-1]
model_out = Model(model.input, model.layers[-2].output)
model_out.summary()
x = model_out.output
dropout = Dropout(rate=drop_rate)(x, training=False)
prediction = dense(dropout)
model = Model(inputs=model.input, outputs=prediction)


# Save model architecture
#plot_model(model, to_file='./modelFineNet.pdf',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

#model.summary()

#============== End define model ==============


#============== Other stuffs for loging and parameters ==================
model_name = 'FineNet_%s_model.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

filepath = os.path.join(save_dir, model_name)


# Show in tensorboard
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

# Prepare callbacks for model saving and for learning rate adjustment.
#checkpoint = ModelCheckpoint(filepath=filepath,
                             #monitor='val_acc',
                             #verbose=1,
                             #save_best_only=True)


lr_scheduler = LearningRateScheduler(lr_schedule)


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler, tensorboard]

#============== End other stuffs  ==================

# Begin training
hist = model.fit_generator(train_batches,
                    validation_data=test_batches,
                    steps_per_epoch= count // batch_size,
                    validation_steps= count1 // batch_size,
                    epochs=epochs, verbose=1,
                    callbacks=callbacks)

# history
df = pd.DataFrame.from_dict(hist.history)
df.to_csv(os.path.join(save_dir, 'hist.csv'), encoding='utf-8', index=False)
    
# save model
save_model(model, filepath)


# Plot confusion matrix
score = model.evaluate_generator(test_batches, steps=(count1 // batch_size)+1)
print 'Test accuracy:', score[1]

mod = count1 % batch_size
if mod > 0:
    mod = 1
else: mod = 0

predictions = model.predict_generator(test_batches, steps=((count1 // batch_size)+mod))

test_labels = test_batches.classes[test_batches.index_array]

cm = confusion_matrix(test_labels, np.argmax(predictions,axis=1))
cm_plot_labels = ['minu','non_minu']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
