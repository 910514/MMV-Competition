import numpy as np
import tensorflow as tf
import time
import tkinter as tk
from tkinter import Label
import threading
from sklearn.preprocessing import LabelEncoder
from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import FeatureMapReceiver

# 損失函數
def l2_loss(y_true, y_pred):
    diff = y_true - y_pred
    loss = tf.reduce_mean(tf.square(diff))
    return loss

classes = ['BathBackground', 'falldown', 'nono', 'patpat', 'slowUp', 'swipeDown', 'swipeLeft', 'swipeRight', 'swipeUp', 'BathBackground', 'BathBackground']
classes_Text = [
    'BackGround',       # BG_Static
    'Warning',          # falldown
    'open/close water', # nono
    'Safe',             # patpat
    'open/close music', # slowUp
    'watertempdDown',   # swipeDown
    'prev music',       # swipeLeft
    'next music',       # swipeRight
    'watertempdUp',     # swipeUp
    'BackGround',       # BG_MV_809
    'BackGround'
]

CNNmodel = tf.keras.models.load_model('CNN_epoch_200_stage2_without_dense.h5', custom_objects={'l2_loss': l2_loss})
LSTMmodel = tf.keras.models.load_model('LSTM_epoch_10_stage2_slice.h5', compile=False)

label_encoder = LabelEncoder()
label_encoder.fit(classes)

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition")
        self.label = Label(root, text="Predicted Gesture: ", font=("Arial", 24))
        self.label.pack(padx=20, pady=20)

    def update_label(self, text, text2):
        self.root.after(0, self.label.config, {'text': f"Predicted Gesture: {text}, {text2}"})

def segment_data(data, segment_length=35):
    gesture_data = []

    total_frames = data.shape[0]
    if total_frames < segment_length:
        print("Not enough frames to form a segment.")
        return np.array(gesture_data)

    for i in range(0, total_frames - segment_length + 1):
        segment = data[i:i + segment_length]
        gesture_data.append(segment)

    gesture_data = np.array(gesture_data)

    if gesture_data.size == 0:
        print("No valid segments found.")

    return gesture_data

def connect():
    connect = ConnectDevice()
    connect.startUp()
    reset = ResetDevice()
    reset.startUp()

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_60cm")
    ksp = SettingProc()
    ksp.startUp(SettingConfigs)

def startLoop(app):
    R = FeatureMapReceiver(chirps=32)
    R.trigger(chirps=32)
    time.sleep(0.5)
    print('# ======== Start getting gesture ===========')

    mainBuffer = []

    while True:
        RDI_data = R.getResults()
        if RDI_data is None:
            continue

        data_array1, data_array2 = RDI_data
        print('RDI data received:', data_array1.shape, data_array2.shape)
        
        rdi_data_combined = np.stack([RDI_data[0], RDI_data[1]], axis=-1).astype(np.float32)
        mainBuffer.append(rdi_data_combined)

        if len(mainBuffer) < 40:
            print("Accumulating more frames.")
            continue

        rec = np.zeros(len(classes))

        for i in range(len(mainBuffer)):
            if i + 35 > len(mainBuffer):
                break
            
            subBuffer = mainBuffer[i:i + 35]
            predictdata = np.array(subBuffer)
            CNNout = (CNNmodel.predict(predictdata)).reshape([-1, 35, 128])
            LSTMout = LSTMmodel.predict(CNNout)
            
            predicted_index = np.argmax(LSTMout, axis=1)[0]
            rec[predicted_index] += 1

            # Debugging prints
            print(f"LSTM Output: {LSTMout}")
            print(f"Predicted Index: {predicted_index}, Class: {classes[predicted_index]}, Text: {classes_Text[predicted_index]}")

        print("Vote Count:", rec)

        app.update_label(classes[np.argmax(rec, axis=0)], classes_Text[np.argmax(rec, axis=0)])

        mainBuffer = []
        rec = []
        time.sleep(5)
        print("====start====")

def main():
    root = tk.Tk()
    app = GestureRecognitionApp(root)

    kgl.setLib()
    connect()
    startSetting()

    threading.Thread(target=lambda: startLoop(app), daemon=True).start()

    root.mainloop()

if __name__ == '__main__':
    main()
