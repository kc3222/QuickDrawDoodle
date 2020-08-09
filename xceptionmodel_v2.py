# Makes changes in learning rates and callbacks

import numpy as np
import six
import tensorflow as tf
import time
import os
import sklearn
#%matplotlib inline
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
import ast
import datetime as dt
import seaborn as sns
import cv2
import pandas as pd
import keras
from keras.layers import *
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet import preprocess_input

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

#start = dt.datetime.now()

DP_DIR = '/home/ttc290/kiet/train_simplified_divided'  # thư mục chứa dữ liệu

BASE_SIZE = 256 # kích thước gốc của ảnh
NCSVS = 100 # số lượng files csv mà chúng ta đã chia ở bước trên
NCATS = 340 # số lượng category (số lớp mà chúng ta cần phân loại)
STEPS = 1000 # số bước huấn luyện trong 1 epoch
EPOCHS = 120 # số epochs huấn luyện
size = 128 # kích thước ảnh training đầu vào
batchsize = 128
np.random.seed(seed=42) # cài đặt seed 
tf.set_random_seed(seed=42) # cài đặt seed 

def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

input_layer = Input((size, size, 1))

common_layer = Reshape((size*size,), input_shape=(size, size, 1), name='common_layer')(input_layer)
common_layer = RepeatVector(3)(common_layer)
common_layer = Reshape((3, size, size), name='rs')(common_layer)
common_layer = Permute((3, 2, 1))(common_layer)

pretrained_model = Xception(weights='imagenet', include_top=True, input_tensor=common_layer, classes=1000)
layer = pretrained_model.layers[-2].output

layer = Dense(NCATS, activation='softmax')(layer)

model = Model(input_layer, layer)

common_layer = Reshape((size*size,), input_shape=(size, size, 1), name='common_layer')(input_layer)
common_layer = RepeatVector(3)(common_layer)
common_layer = Reshape((3, size, size), name='rs')(common_layer)
common_layer = Permute((3, 2, 1))(common_layer)

model.compile(optimizer=Adam(lr=0.003), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

def draw_cv2(raw_strokes, size=128, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                
                yield x, y

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x

valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))
# NCSVS - 1 = 99 (không lấy file cuối cùng vì file đó dùng cho tập validation)

filepath = '/scratch/ttc290/kiet/xceptionmodel_v2_120epoch/accuracy_{val_categorical_accuracy:.10f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.7, patience=7, mode='max', cooldown=3, verbose=1),
    checkpoint
]

history = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS,
    validation_data=(x_valid, y_valid),
    callbacks=callbacks
)

# Saving history
import json

with open(os.path.join('/home/ttc290/kiet', 'xceptionmodel_v2_120epoch.json'), 'w') as f:
    json.dump(history.history, f)
