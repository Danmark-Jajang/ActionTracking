import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from CollectionData import actions
import numpy as np
import os
import queue

DATA_PATH = os.path.join('MP_Data')
# 30개의 프레임
no_sequences = 30
# 한 비디오당 탐지할 길이
sequence_length = 30

# Data load
label_map = {label:num for num, label in enumerate(actions)}

def load_data():
    sequences, label = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_number in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), '{}.npy'.format(frame_number)))
                window.append(res)
            sequences.append(window) # 30*n 개의 다른 데이터가 로드됨, 30개의 frame, 1662개의 각 landmark
            label.append(label_map[action])  
    x = np.array(sequences)
    y = keras.utils.to_categorical(label).astype(int)     
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)
    return x_train, x_test, y_train, y_test
            
#Learning Model build
def model_fn():
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(64, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(actions.shape[0], activation='softmax'))
    return model

def start_learning(x_train, y_train):
    model = model_fn()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    log_dir = os.path.join('Logs')
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(x_train, y_train, epochs=300, callbacks=[tb_callback])
    model.save('faction.h5')
    return model

if __name__=="__main__":
    x_train, x_test, y_train, y_test = load_data()
    print(x_train.shape, x_test.shape)
    start_learning(x_train, y_train)