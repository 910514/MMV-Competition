from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# 損失函數
def l2_loss(y_true, y_pred):
    diff = y_true - y_pred
    #print(f"Difference (diff): {diff.numpy()}")
    #print(f"y_true (label): {y_true}")
    #print(f"y_pred (label_pred): {y_pred}")
    loss = tf.reduce_mean(tf.square(diff))
    #print(f"loss: {loss}")
    return loss

model = load_model('ourCNNmodel_forTL_300_epoch_Dropout_0.7.h5', custom_objects={'l2_loss': l2_loss})

a = np.load('swipRight_segment_231.npy')

a = np.transpose(a, [3, 1, 2, 0])

model.predict(a)