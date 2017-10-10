import numpy as np 
import pandas as pd 

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2

from random import shuffle

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

import matplotlib.pyplot as plt

df_train = pd.read_csv('/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/labels.csv')
df_test = pd.read_csv('/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/sample_submission.csv')

TRAIN_DIR = '/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/train/{}.jpg'
TEST_DIR = '/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/test/{}.jpg'



print(df_train.head(10))


targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)


im_size = 90
MODEL_NAME = 'model1'

x_train = []
y_train = []
x_test = []

'''
i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread(TRAIN_DIR.format(f))
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1

for f in tqdm(df_test['id'].values):
    img = cv2.imread(TEST_DIR.format(f))
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_test.append(cv2.resize(img, (im_size, im_size)))


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255.


print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)

np.save('y_train_raw.npy', y_train_raw)
np.save('x_train_raw.npy', x_train_raw)
np.save('x_test.npy', x_test)

'''

y_train_raw = np.load('y_train_raw.npy')
x_train_raw = np.load('x_train_raw.npy')
x_test = np.load('x_test.npy')

print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)

num_class = y_train_raw.shape[1]

#print(x_train_raw)

xtrain = x_train_raw[:-1000]
ytrain = y_train_raw[:-1000]
 
xtest = x_train_raw[-1000:]
ytest = y_train_raw[-1000:]


X = xtrain
Y = ytrain

test_x = xtest
test_y = ytest


###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

print('before_cal')

####convnet = input_data(shape=[None, im_size, im_size, 3], name='input')

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, im_size, im_size, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')
# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)
# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')
# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')
# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)
# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')
# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)
# 8: Fully-connected layer with two outputs
network = fully_connected(network, num_class, activation='softmax')
# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.0005, metric=acc)
# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model111.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
###################################
# Train model for 100 epochs
###################################
model.fit(X, Y, validation_set=(test_x, test_y), batch_size=500,n_epoch=100, run_id='model111', show_metric=True)
model.save('model111_final.tflearn')



model.load('model111_final.tflearn')


#print(one_hot.head())
label_list = list(one_hot.columns.values)

#print(label_list)


with open('submission_file_color.csv','w') as f:
    f.write('{},{}'.format('id',(str(label_list[0])+',')))
    for ii in range(1, num_class-1):
            f.write('{}'.format(label_list[ii]))
            f.write('{}'.format(','))

    f.write('{}\n'.format(label_list[-1]))



with open('submission_file_color.csv','a') as f:
    for find in tqdm(df_test['id'].values):
        img_num = find
        img_data = cv2.imread(TEST_DIR.format(find))
        img_data = cv2.resize(img_data, (im_size, im_size))
        img_data = np.array(img_data)
        img_data = np.array(img_data, np.float32) / 255.

        model_out = model.predict([img_data])
        a = model_out.tolist()
        f.write('{},{}'.format(img_num,(str(a[0][0])+',')))

        for ii in range(1, num_class-1):
            f.write('{}'.format(a[0][ii]))
            f.write('{}'.format(','))

        f.write('{}\n'.format(a[0][-1]))
        #print(a[0][0])
        #print('split')





