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
    #print(f"Difference (diff): {diff.numpy()}")
    #print(f"y_true (label): {y_true}")
    #print(f"y_pred (label_pred): {y_pred}")
    loss = tf.reduce_mean(tf.square(diff))
    #print(f"loss: {loss}")
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
# classes=['fall_down', 'patpat', 'quickDown', 'quickUp', 'rollin', 'rollout', 'swipLeft', 'swipRight', 'None']
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
        '''        
        # if segment.shape == (segment_length, input_height, input_width, num_channels):
        #     gesture_data.append(segment)
        # else:
        #     print(f"Skipping segment with shape {segment.shape} due to size mismatch.")
        '''

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
    rec=[]
    timesleeper=0
    '''
    gesture_flag = False
    lower_thd = 0.2
    upper_thd = 0.5
    current_gest = 0
    '''

    while True:
        RDI_data = R.getResults()
        if RDI_data is None:
            continue

        data_array1, data_array2 = RDI_data

        print('RDI data received:', data_array1.shape, data_array2.shape)
        rdi_data_combined = np.stack([RDI_data[0], RDI_data[1]], axis=-1).astype(np.float32)
        mainBuffer.append(rdi_data_combined)

        if len(mainBuffer) < 35:
            print("Accumulating more frames.")
            continue
        elif len(mainBuffer) > 40:
            mainBuffer=mainBuffer[1:40]
        
        '''
        predictdata=np.array(mainBuffer[len(mainBuffer)-35:len(mainBuffer)])
        CNNout=(CNNmodel.predict(predictdata)).reshape([-1, 35, 128])
        LSTMout=LSTMmodel.predict(CNNout)
        print(LSTMout)

        
        if(gesture_flag and LSTMout[0][current_gest] < lower_thd):
            gesture_flag = False
        
        if((gesture_flag==False) and np.max(LSTMout) >= upper_thd):
            current_gest = np.argmax(LSTMout, axis=1)[0]
            app.update_label(classes[current_gest], classes_Text[current_gest])
            gesture_flag = True
        elif gesture_flag==False:
            app.update_label(classes[0], classes_Text[0])
        '''
        
        predictdata=np.array(mainBuffer[len(mainBuffer)-35:len(mainBuffer)])
        CNNout=(CNNmodel.predict(predictdata)).reshape([-1, 35, 128])
        LSTMout=LSTMmodel.predict(CNNout)
        print(LSTMout)
        LSTMout_truth=LSTMout>0.5
        back_truth=np.sum(predictdata[34]) > 50000
        print(back_truth)
        if np.max(LSTMout) > 0.5 and len(LSTMout[LSTMout_truth==True]) == 1:
            rec.append(np.argmax(LSTMout, axis=1)[0])
        else:
            rec.append(0)

        if len(rec) > 6:
            rec=rec[1:len(rec)]
        print(rec)
        
        if rec.count(max(rec, key=rec.count)) > 3 and max(rec, key=rec.count)!=0 and max(rec, key=rec.count)==np.argmax(LSTMout, axis=1)[0] and back_truth and timesleeper==0:
            app.update_label(classes[max(rec, key=rec.count)], classes_Text[max(rec, key=rec.count)])
        elif timesleeper==0:
            app.update_label('None', 'background')

        if timesleeper < 6:
            timesleeper+=1
        else:
            timesleeper=0

        '''
        # subBuffer = mainBuffer.reshape([-1, 32, 32, 2])

        # if subBuffer.size == 0:
        #     print("No valid segments found.")
        #     mainBuffer = []
        #     continue

        # print(f"Segmented data shape: {subBuffer.shape}")

        # reshaped_data = subBuffer.reshape(-1, 32, 32, 2)
        
        CNN_test_predictions = CNNmodel.predict(reshaped_data)
        test_predictions = LSTMmodel.predict(CNN_test_predictions.reshape([-1, 35, 128]))
        # Get the index of the highest probability class for each prediction
        predicted_indices = np.argmax(test_predictions, axis=1)

        if len(predicted_indices) > 0:
            predicted_classes = label_encoder.inverse_transform(predicted_indices)
            predicted_text = predicted_classes[0]
        else:
            predicted_text = "Unknown"

        app.update_label(predicted_text)

        print(test_predictions)
        mainBuffer = []
        time.sleep(2)
        print("====start====")
        '''

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