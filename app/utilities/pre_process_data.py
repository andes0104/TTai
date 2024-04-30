from app.utilities.setting import Env
from app.utilities.common import Common
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

RANDOM_STATE = Env().RANDOM_STATE


def pre_process(npy_path: str):
    # 從.npy取出特徵並建立標籤
    labels, sequences = Common().creat_labels(npy_path)

    # 有調整訓練模型架構, 可接受不同長度的序列來訓練, 因此不用調整sequences
    X = pad_sequences(sequences)

    # 將 labels 轉換成 NumPy 陣列，並使用 to_categorical 函數將標籤轉換成 one-hot 編碼
    y = to_categorical(labels).astype(int)

    # 切割測試資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
