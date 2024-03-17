from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from app.utilities.setting import Env
from keras.layers import Masking
import os


def model_constructure():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(4, activation="softmax"))

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # 印出模型架構
    model.summary()
    return model
