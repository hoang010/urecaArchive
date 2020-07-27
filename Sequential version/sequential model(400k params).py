# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# import cv2

# img = cv2.imread('/kaggle/input/train_images/train_images/10000.jpg')
# print(img)
# os.getcwd()
# os.listdir('/kaggle/input/')
# print(os.listdir("../input/"))

# !pip install tensorflow==2.0.0-alpha0


import cv2
import random
import numpy as np
from math import sqrt
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D,\
    Concatenate, BatchNormalization, LeakyReLU, AveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def interval(a, b):
    sess = tf.Session()

    x1, x2 = a
    x3, x4 = b
    if sess.run(tf.math.less(x3,x1)):
        if sess.run(tf.math.less(x4,x1)):
            return 0
        else:
            return tf.math.minimum(x2, x4) - x1

    else:   # x3 >= x1
        if sess.run(tf.math.less(x2,x3)):
            return 0
        else:
            return tf.math.minimum(x2, x4) - x3

def iou(box1, box2):
    xmin = K.maximum(box1[0], box2[0])
    ymin = K.maximum(box1[1], box2[1])
    xmax = K.minimum(box1[2], box2[2])
    ymax = K.minimum(box1[3], box2[3])

    w = K.maximum(0.0, xmax - xmin)
    h = K.maximum(0.0, ymax - ymin)

    intersection = w * h

    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersection

    return intersection/union * 100


def batch_iou():
    def batch_iou_2(y_true, y_pred):
        list_of_iou = []
        result = 0
        for i in range(batch_size):
            list_of_iou.append(iou(y_true[i],y_pred[i]))
        return K.mean(tf.convert_to_tensor(list_of_iou, dtype = tf.float32))
    return batch_iou_2
# data
img_paths = open('/kaggle/input/img_train_final.txt').read().split()
label_paths = open('/kaggle/input/lab_train_final.txt').read().split()

# variables
img_h = 360
img_w = 640
decay = 5e-4
learning_rate = 1e-3
adam =  Adam(lr=learning_rate)
#sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
leaky = LeakyReLU(alpha=0.1)
batch_size = 16
epochs = 25



# Model

input_layer = Input(shape=(360,640,3))

conv1 = Conv2D(32, (3,3), strides=(2,2), padding ='valid')(input_layer)
norm1 = BatchNormalization()(conv1)
actv1 =  LeakyReLU(alpha=0.1)(norm1)

conv2 = Conv2D(64, (3,3), strides=(1,1), padding ='valid')(actv1)
norm2 = BatchNormalization()(conv2)
actv2 =  LeakyReLU(alpha=0.1)(norm2)

conv3 = Conv2D(64, (3,3), strides=(1,1), padding ='valid')(actv2)
norm3 = BatchNormalization()(conv3)
actv3 =  LeakyReLU(alpha=0.1)(norm3)

max1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(actv3)

conv4 = Conv2D(80, (1,1), strides=(1,1), padding ='valid',activation='linear')(max1)
norm4 = BatchNormalization()(conv4)
actv4 =  LeakyReLU(alpha=0.1)(norm4)

conv5 = Conv2D(192, (3,3), strides=(1,1), padding ='valid',activation='linear')(actv4)
norm5 = BatchNormalization()(conv5)
actv5 =  LeakyReLU(alpha=0.1)(norm5)


## node 1 -- layer 17
k = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(actv5)
print(k.shape)



for x in range(2):
    k = Conv2D(32, (3,3), strides=(1,1), padding ='same',activation='linear')(k)
    k = BatchNormalization()(k)
    k =  LeakyReLU(alpha=0.1)(k)
    
    k = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(k)
    k = BatchNormalization()(k)
    k =  LeakyReLU(alpha=0.1)(k)
    
    k = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(k)
    k = BatchNormalization()(k)
    k =  LeakyReLU(alpha=0.1)(k)
    
    k = MaxPool2D(pool_size=(3,3), strides=(2,2), padding = 'valid')(k)
    
    if (x==1):
        k = Conv2D(4, (9, 18), strides=(1,1), padding ='valid',activation='linear')(k)
        k = Flatten()(k)

        
        
    
model = Model(input_layer, k)
model.summary()

batch_iou = batch_iou()

model.compile(optimizer=adam, loss='mse', metrics=['accuracy', batch_iou])

def load_label(label_f):
    line = open(label_f).read().split('\n')
    
    label = line[0].split(' ')
    
    return np.asarray(label, dtype='float32')


def load_batch(img_paths, label_paths):

    images = np.zeros((batch_size, img_h, img_w, 3))
    batch_labels = np.zeros((batch_size,4))

    indx = 0
    
    for imgFile, labelFile in zip(img_paths, label_paths):
        img = cv2.imread(imgFile).astype(np.float32, copy=False)
        images[indx] = img

        label = load_label(labelFile)
        batch_labels[indx] = label

        indx +=1

    return images, batch_labels
        

def generator(img_names, gt_names, batch_size):
    # Create empty arrays to contain batch of features and labels#

    assert len(img_names) == len(gt_names), "Number of images and ground truths not equal"

    nbatches, n_skipped_per_epoch = divmod(len(img_names), batch_size)

    if True:
        #permutate images
        shuffled = list(zip(img_names, gt_names))
        random.shuffle(shuffled)
        img_names, gt_names = zip(*shuffled)

    nbatches, n_skipped_per_epoch = divmod(len(img_names), batch_size)

    count = 1
    epoch = 0

    while 1:

        epoch += 1
        i, j = 0, batch_size

        #mini batches within epoch
        mini_batches_completed = 0

        for _ in range(nbatches):
            #print(i,j)
            img_names_batch = img_names[i:j]
            gt_names_batch = gt_names[i:j]

            try:
                #get images and ground truths
                imgs, gts = load_batch(img_names_batch, gt_names_batch)

                mini_batches_completed += 1
                yield (imgs, gts)

            except IOError as err:

                count -= 1

            i = j
            j += batch_size
            count += 1




train_generator = generator(img_paths, label_paths, batch_size)
model.fit_generator(train_generator, steps_per_epoch = len(img_paths)//batch_size, epochs = 25, verbose =1)


model.save('my_model_v2.h5')

