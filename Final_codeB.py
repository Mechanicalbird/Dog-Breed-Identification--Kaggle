import cv2
import numpy as np
import pandas as pd
import numpy as np
import os
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
import warnings
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


TRAIN_DIR = '/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/train'
TEST_DIR = '/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/test'
labels = pd.read_csv('/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/labels.csv')
sample_submission = pd.read_csv('/home/mohammad/Desktop/ML/DogBreedIdentification/Dog-Breed-Identification/sample_submission.csv')

IMG_SIZE = 50
LR = 1e-3



### count thr unique value
def count_the_unique_values (data_set, column):
    description = data_set.describe(include="all")
    unique = description.iloc[1][column]
    return unique

### Make the unique values data as a row
def process_the_label_data(column,num_classes,sort,data_set):
    selected_breed_list = list(data_set.groupby(column).count().sort_values(by=sort, ascending=False).head(num_classes).index)
    data_set = data_set[data_set[column].isin(selected_breed_list)]
    data_set['target'] = 1
    #labels['rank'] = labels.groupby(column).rank()[sort]
    labels_pivot = data_set.pivot(sort, column, 'target').reset_index().fillna(0)
    return labels_pivot

print('### Data Preprocessing ###')

print(len(listdir(TRAIN_DIR)), len(labels))
print(len(listdir(TEST_DIR)), len(sample_submission))
print('#####################################################')

print(labels.head())
print('#####################################################')

NUM_CLASSES = count_the_unique_values (labels,'breed')
print('number of unique value = ',NUM_CLASSES)
print('#####################################################')
labels_row  = process_the_label_data('breed',NUM_CLASSES,'id',labels)


labels_use = df = labels_row.drop('id', axis=1)



def create_train_data(start, end, file_num):
    train_data = []
    i = 0
    for x in range(start, end):
        print(x)  
        img_name = labels_row.iloc[start][0]+'.jpg'
        path = os.path.join(TRAIN_DIR, img_name)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img), np.array(labels_use)])

    file_name='train_data'+file_num+'.npy'
    print(file_name)
    shuffle(train_data)
    np.save(file_name, train_data)
    return train_data

def manage_RAM_memory(memory, count):    
    for x in range(0, memory):
        start_data=x*count
        end_data=count+start_data
        file_num_data=x+1
        create_train_data(start_data, end_data, str(file_num_data))

def load_the_data_from_the_files(memory):
    train_data=np.load('train_data1.npy')
    print('train_data1.npy')
    np.array(labels_use)
    for x in range (2,memory+1):
        file_name= 'train_data'+str(x)+'.npy'
        print(file_name)
        train_data_new= np.load(file_name)
        train_data=np.concatenate((train_data, train_data_new), axis=0)
    return train_data


final_data=int(len(labels_row))
memory_parts=20
count_data= int(final_data/memory_parts)

manage_RAM_memory(memory_parts, count_data)
train_data=load_the_data_from_the_files(memory_parts)

print('sample_data_size',count_data)
print('final_data_shape = ',train_data.shape)

np.save('train_data_all.npy', train_data)


train = train_data[:-500]
test = train_data[-500:]


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

model.load(MODEL_NAME)





















