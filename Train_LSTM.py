# %%
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv1D, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
import time
import numpy as np

# %%
main_input = tf.keras.layers.Input(shape=(35,128))
# x = tf.reshape(main_input, (1, 35, 128))
x = tf.keras.layers.LSTM(units=128, activation="tanh", time_major=False, return_sequences=True)(main_input)
#x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(12, activation=None)(x)
x = tf.keras.layers.Softmax()(x)

model = tf.keras.Model(inputs=main_input, outputs=x)

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# %%
predictions_train = np.load('LSTM_STAGE1/predictions_train.npy')
labels_train = np.load('LSTM_STAGE1/predictions_labels_train.npy')

predictions_val = np.load('LSTM_STAGE1/predictions_val.npy')
labels_val = np.load('LSTM_STAGE1/predictions_labels_val.npy')

predictions_test = np.load('LSTM_STAGE1/predictions_test.npy')
labels_test = np.load('LSTM_STAGE1/predictions_labels_test.npy')

# %%
data_train = []
data_train_labels = []

for i in range(1000000):
    if i + 35 > len(predictions_train):
        break
    data_train.append(predictions_train[i:i + 35, :])
    data_train_labels.append(labels_train[i:i+35, :])

print(np.shape(data_train))
print(np.shape(data_train_labels))

data_val = []
data_val_labels = []

for i in range(1000000):
    if i + 35 > len(predictions_val):
        break
    data_val.append(predictions_val[i:i + 35, :])
    data_val_labels.append(labels_val[i:i+35, :])

print(np.shape(data_val))
print(np.shape(data_val_labels))

data_test = []
data_test_labels = []

for i in range(1000000):
    if i + 35 > len(predictions_test):
        break
    data_test.append(predictions_test[i:i + 35, :])
    data_test_labels.append(labels_test[i:i+35, :])
print(np.shape(data_test))
print(np.shape(data_test_labels))

# %%
import numpy as np

def slice_35(data, sequence_length=35):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length, :])
    return np.array(sequences)

# %%
def load_and_select_data(num_selections, num_bg_classes=3, num_gesture_classes=9, validation_split=0.1):
    # Load datasets
    path1 = "LSTM_STAGE1/Predict_BackGround_MV_809.npy"
    data1 = np.load(path1)

    path2 = "LSTM_STAGE1/Predict_BackGround_MV_809_pat.npy"
    data2 = np.load(path2)

    path3 = "LSTM_STAGE1/Predict_BackGround_falldown.npy"
    data3 = np.load(path3)

    total_elements_data1 = data1.shape[-1]
    total_elements_data2 = data2.shape[-1]
    total_elements_data3 = data3.shape[-1]

    weight1 = 1.0
    weight2 = 10.0
    weight3 = 17.0

    weights = [weight1] * total_elements_data1 + [weight2] * total_elements_data2 + [weight3] * total_elements_data3
    total_weight = np.sum(weights)
    probabilities = np.array(weights) / total_weight

    cdf = np.cumsum(probabilities)
    uniform_random_numbers = np.random.uniform(0, cdf[-1], size=num_selections)
    selected_indices = np.searchsorted(cdf, uniform_random_numbers)

    selected_data = []
    selected_labels = []

    bg_labels_12 = np.zeros((num_bg_classes, 12))
    bg_labels_12[np.arange(num_bg_classes), np.arange(num_bg_classes) + 9] = 1

    for index in selected_indices:
        if index < total_elements_data1:
            selected_data.append(data1[index])
            selected_labels.append(bg_labels_12[index % num_bg_classes])
        elif index < total_elements_data1 + total_elements_data2:
            index2 = index - total_elements_data1
            selected_data.append(data2[index2])
            selected_labels.append(bg_labels_12[index2 % num_bg_classes])
        else:
            index3 = index - (total_elements_data1 + total_elements_data2)
            selected_data.append(data3[index3])
            selected_labels.append(bg_labels_12[index3 % num_bg_classes])

    selected_data = np.array(selected_data)
    selected_labels = np.array(selected_labels)

    num_val = int(num_selections * validation_split)
    num_train = num_selections - num_val

    x_train = selected_data[:num_train]
    y_train = selected_labels[:num_train]
    x_val = selected_data[num_train:]
    y_val = selected_labels[num_train:]

    return (x_train, y_train), (x_val, y_val)



# %%
def AGIprogressBar(count, total,start):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    duration=time.time()-start
    print('\r[%s] %s%s ...%s sec' % (bar, percents, '%', duration),end=' ')

# %%
import time
import numpy as np

Batch = 6720
epochs = 500

rec = []
st = time.time()

for ep in range(epochs):
    print(f'EP: {ep + 1}')
    (num_train_selections, num_bg_classes, num_gesture_classes) = (150000, 3, 9)
    ((x_train, y_train), (x_val, y_val)) = load_and_select_data(num_selections=num_train_selections, num_bg_classes=num_bg_classes, num_gesture_classes=num_gesture_classes)
    y_train = slice_35(y_train, sequence_length=35)
    y_val = slice_35(y_val, sequence_length=35)
    x_train = slice_35(x_train, sequence_length=35)
    x_val = slice_35(x_val, sequence_length=35)

    x_train_combined = np.concatenate((data_train, x_train), axis=0)
    y_train_combined = np.concatenate((data_train_labels, y_train), axis=0)
    
    for i in range(len(x_train_combined) // Batch):
        rand = np.random.randint(0, len(x_train_combined), size=Batch)
        AGIprogressBar(i, len(x_train_combined) // Batch, st)

        x_batch = np.array([x_train_combined[idx] for idx in rand])
        y_batch = np.array([y_train_combined[idx] for idx in rand])

        resultSt = model.train_on_batch(x_batch, y_batch)

    pre = model.predict(x_val, verbose=0)

    # Calculate accuracy
    acc = np.sum(np.argmax(pre, axis=2) == np.argmax(y_val, axis=2)) / y_val.size 
    print(f'ACC = {acc}')
    rec.append(acc)

    if (ep + 1) % 10 == 0:
        model.save(f'saved_model_stage2_CNN/CNN_epoch_{ep + 1}_stage1.h5')
        print(f'Model saved at epoch {ep + 1}')