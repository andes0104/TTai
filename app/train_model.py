from app.utilities.model_constructure_init import model_constructure
from app.utilities.setting import Env
import numpy as np
import os
from tensorflow import keras


def train_model(data):
    model = model_constructure()
    # train log
    log_dir = os.path.join("Logs")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    # 訓練模型model.summary()
    model.fit(data["X_train"], data["y_train"], epochs=200, callbacks=[tb_callback])

    # 儲存模型
    model.save(Env().MODEL_FILE)