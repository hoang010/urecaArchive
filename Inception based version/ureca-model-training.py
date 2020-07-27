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
max2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(actv5)

## path 1
conv4 = Conv2D(64, (1,1), strides=(1,1), padding ='same',activation='linear')(max1)
norm4 = BatchNormalization()(conv4)
actv4 =  LeakyReLU(alpha=0.1)(norm4)

conv5 = Conv2D(48, (1,1), strides=(1,1), padding ='same',activation='linear')(actv4)
norm5 = BatchNormalization()(conv5)
actv5 =  LeakyReLU(alpha=0.1)(norm5)

conv6 = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(actv5)
norm6 = BatchNormalization()(conv6)
actv6 =  LeakyReLU(alpha=0.1)(norm6)
###


## path 2
conv7 = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(max1)
norm7 = BatchNormalization()(conv7)
actv7 =  LeakyReLU(alpha=0.1)(norm7)

conv8 = Conv2D(48, (3,3), strides=(1,1), padding ='same',activation='linear')(actv7)
norm8 = BatchNormalization()(conv8)
actv8 =  LeakyReLU(alpha=0.1)(norm8)
###


## path 3
conv9 = Conv2D(64, (5,5), strides=(1,1), padding ='same',activation='linear')(max1)
norm9 = BatchNormalization()(conv9)
actv9 =  LeakyReLU(alpha=0.1)(norm9)
###


## path 4
avgp1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding = 'same')(max1)

conv10 = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(avgp1)
norm10 = BatchNormalization()(conv10)
actv10 =  LeakyReLU(alpha=0.1)(norm10)
###



## node 2
conc1 = Concatenate()([actv6, actv8, actv9, actv10])

## path 1
conv11 = Conv2D(48, (3,3), strides=(1,1), padding ='same',activation='linear')(conc1)
norm11 = BatchNormalization()(conv11)
actv11 =  LeakyReLU(alpha=0.1)(norm11)

maxp4 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding = 'valid')(actv11)

conv12 = Conv2D(128, (3,3), strides=(1,1), padding ='same',activation='linear')(maxp4)
norm12 = BatchNormalization()(conv12)
actv12 =  LeakyReLU(alpha=0.1)(norm12)

conv13 = Conv2D(256, (3,3), strides=(1,1), padding ='same',activation='linear')(actv12)
norm13 = BatchNormalization()(conv13)
actv13 =  LeakyReLU(alpha=0.1)(norm13)
###


## path 2
conv14 = Conv2D(64, (5,5), strides=(1,1), padding ='same',activation='linear')(conc1)
norm14 = BatchNormalization()(conv14)
actv14 =  LeakyReLU(alpha=0.1)(norm14)

conv27 = Conv2D(64, (3,3), strides=(2,2), padding ='valid',activation='linear')(actv14)
norm27 = BatchNormalization()(conv27)
actv27 =  LeakyReLU(alpha=0.1)(norm27)

conv15 = Conv2D(128, (5,5), strides=(1,1), padding ='same',activation='linear')(actv27)
norm15 = BatchNormalization()(conv15)
actv15 =  LeakyReLU(alpha=0.1)(norm15)

conv16 = Conv2D(128, (5,5), strides=(1,1), padding ='same',activation='linear')(actv15)
norm16 = BatchNormalization()(conv16)
actv16 =  LeakyReLU(alpha=0.1)(norm16)
###


## path 3
conv17 = Conv2D(64, (3,3), strides=(2,2), padding ='valid',activation='linear')(conc1)
norm17 = BatchNormalization()(conv17)
actv17 =  LeakyReLU(alpha=0.1)(norm17)
###


## path 4
avgp2 = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding = 'valid')(conc1)

conv18 = Conv2D(64, (1,1), strides=(1,1), padding ='same',activation='linear')(avgp2)
norm18 = BatchNormalization()(conv18)
actv18 =  LeakyReLU(alpha=0.1)(norm18)

conv21 = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(actv18)
norm21 = BatchNormalization()(conv21)
actv21 =  LeakyReLU(alpha=0.1)(norm21)
###


## path 4
maxp3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding = 'valid')(conc1)

conv19 = Conv2D(64, (1,1), strides=(1,1), padding ='same',activation='linear')(maxp3)
norm19 = BatchNormalization()(conv19)
actv19 =  LeakyReLU(alpha=0.1)(norm19)

conv20 = Conv2D(64, (1,1), strides=(1,1), padding ='same',activation='linear')(actv19)
norm20 = BatchNormalization()(conv20)
actv20 =  LeakyReLU(alpha=0.1)(norm20)
###


conc2 = Concatenate()([actv13, actv16, actv17, actv21, actv20])


conv21 = Conv2D(128, (3,3), strides=(1,1), padding ='same',activation='linear')(conc2)
norm21 = BatchNormalization()(conv21)
actv21 =  LeakyReLU(alpha=0.1)(norm21)

conv22 = Conv2D(256, (3,3), strides=(1,1), padding ='same',activation='linear')(actv21)
norm22 = BatchNormalization()(conv22)
actv22 =  LeakyReLU(alpha=0.1)(norm22)

conv23 = Conv2D(128, (3,3), strides=(1,1), padding ='same',activation='linear')(actv22)
norm23 = BatchNormalization()(conv23)
actv23 =  LeakyReLU(alpha=0.1)(norm23)

maxp5 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding = 'valid')(actv23)


conv24 = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(maxp5)
norm24 = BatchNormalization()(conv24)
actv24 =  LeakyReLU(alpha=0.1)(norm24)

conv25 = Conv2D(128, (3,3), strides=(1,1), padding ='same',activation='linear')(actv24)
norm25 = BatchNormalization()(conv25)
actv25 =  LeakyReLU(alpha=0.1)(norm25)

conv26 = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(actv25)
norm26 = BatchNormalization()(conv26)
actv26 =  LeakyReLU(alpha=0.1)(norm26)

maxp6 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding = 'valid')(actv26)


conv27 = Conv2D(64, (3,3), strides=(1,1), padding ='same',activation='linear')(maxp6)
norm27 = BatchNormalization()(conv27)
actv27 = LeakyReLU(alpha=0.1)(norm27)

conv28 = Conv2D(128, (3,3), strides=(1,1), padding ='same',activation='linear')(actv27)
norm28 = BatchNormalization()(conv28)
actv28 = LeakyReLU(alpha=0.1)(norm28)

conv29 = Conv2D(128, (3,3), strides=(1,1), padding ='same',activation='linear')(actv28)
norm29 = BatchNormalization()(conv29)
actv29 = LeakyReLU(alpha=0.1)(norm29)


conv30 = Conv2D(32, (10,18), strides=(1,1), padding ='valid',activation='linear')(actv29)

conv31 = Conv2D(4, (1,1), strides=(1,1), padding ='valid',activation='linear')(conv30)

out = Flatten()(conv31)


model = Model(input_layer, out)
batch_iou = batch_iou()

model.compile(optimizer=adam, loss='mse', metrics=['accuracy', batch_iou])
model.summary()


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
model.fit_generator(train_generator, steps_per_epoch = len(img_paths)//batch_size,
                    epochs = epochs, verbose =1)


model.save('my_model_v2.h5')

