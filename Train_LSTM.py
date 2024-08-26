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
optimizer=tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

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
# 背景隨機採樣
def load_and_select_data(num_selections, num_bg_classes=3, num_gesture_classes=9, validation_split=0.1, TimeSp=35):
    path1 = "LSTM_STAGE1/BackGround_falldown.npy"
    data1 = np.load(path1)
    labels1 = np.full(len(data1), fill_value=9)  # Background 1

    path2 = "LSTM_STAGE1/BackGround_MV_809_pat.npy"
    data2 = np.load(path2)
    labels2 = np.full(len(data2), fill_value=10)  # Background 2

    path3 = "LSTM_STAGE1/BackGround_MV_809.npy"
    data3 = np.load(path3)
    labels3 = np.full(len(data3), fill_value=11)  # Background 2

    def slice35(data, labels, TimeSp):
        segments = []
        segment_labels = []
        for j in range(len(data)):
            if j + TimeSp > len(data):
                break
            segment = data[j:j + TimeSp, :]
            segments.append(segment)
            segment_labels.append(labels[j])
        
        return np.array(segments), np.array(segment_labels)

    data1_segments, labels1_segments = slice35(data1, labels1, TimeSp)
    data2_segments, labels2_segments = slice35(data2, labels2, TimeSp)
    data3_segments, labels3_segments = slice35(data3, labels3, TimeSp)

    all_segments = np.concatenate([data1_segments, data2_segments, data3_segments], axis=0)
    all_labels = np.concatenate([labels1_segments, labels2_segments, labels3_segments], axis=0)

    class_proportions = {
        9: 0.3,
        10: 0.3,
        11: 0.3
    }

    num_classes = num_bg_classes + num_gesture_classes
    selected_segments = []
    selected_labels = []

    for class_id, proportion in class_proportions.items():
        class_indices = np.where(all_labels == class_id)[0]
        num_class_samples = int(proportion * num_selections)

        num_class_samples = min(num_class_samples, len(class_indices))

        selected_class_indices = np.random.choice(class_indices, size=num_class_samples, replace=False)
        selected_segments.extend(all_segments[selected_class_indices])
        selected_labels.extend(all_labels[selected_class_indices])

    selected_segments = np.array(selected_segments)
    selected_labels = np.array(selected_labels)

    num_val_samples = int(validation_split * len(selected_segments))
    split_index = len(selected_segments) - num_val_samples

    val_segments = selected_segments[split_index:]
    val_labels = selected_labels[split_index:]

    train_segments = selected_segments[:split_index]
    train_labels = selected_labels[:split_index]

    one_hot_train_labels = np.zeros((len(train_labels), TimeSp, num_classes), dtype=np.float32)
    one_hot_val_labels = np.zeros((len(val_labels), TimeSp, num_classes), dtype=np.float32)

    for i in range(len(train_labels)):
        one_hot_train_labels[i, :, train_labels[i]] = 1

    for i in range(len(val_labels)):
        one_hot_val_labels[i, :, val_labels[i]] = 1

    return train_segments, one_hot_train_labels, val_segments, one_hot_val_labels


# %%
def AGIprogressBar(count, total,start):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    duration=time.time()-start
    print('\r[%s] %s%s ...%s sec' % (bar, percents, '%', duration),end=' ')

# %%
Batch = 24
epochs = 10
rec = []
losses = []
st = time.time()

for ep in range(epochs):
    num_train_selections = 151300
    x_train, y_train, x_val, y_val = load_and_select_data(num_selections=num_train_selections)

    x_train_combined = np.concatenate((data_train, x_train), axis=0)
    y_train_combined = np.concatenate((data_train_labels, y_train), axis=0)

    x_val_combined = np.concatenate((data_val, x_val), axis=0)
    y_val_combined = np.concatenate((data_val_labels, y_val), axis=0)

    print(f'EP: {ep + 1}')
    
    indices = np.arange(len(x_train_combined))
    np.random.shuffle(indices)
    x_train_combined = x_train_combined[indices]
    y_train_combined = y_train_combined[indices]
    epoch_loss = 0
    
    for i in range(len(x_train_combined) // Batch):
        AGIprogressBar(i, len(x_train_combined) // Batch, st)

        x_batch = x_train_combined[i*Batch:(i+1)*Batch]
        y_batch = y_train_combined[i*Batch:(i+1)*Batch]

        resultSt = model.train_on_batch(x_batch, y_batch)
        batch_loss = resultSt[0]
        epoch_loss += batch_loss
    
    epoch_loss /= (len(x_train_combined) // Batch)
    losses.append(epoch_loss)
    print(f'Loss = {epoch_loss}')

    # Predict on validation data
    pre = model.predict(x_val_combined, verbose=0)

    total_predictions = len(y_val_combined) * y_val_combined.shape[1]

    correct_predictions = np.sum(np.argmax(pre, axis=2) == np.argmax(y_val_combined, axis=2))

    acc = correct_predictions / total_predictions
    rec.append(acc)
    print(f'ACC = {acc}')

    model.save(f'saved_model_stage1_LSTM/LSTM_epoch_{ep + 1}_stage1.h5')
    print(f'Model saved at epoch {ep + 1}')



