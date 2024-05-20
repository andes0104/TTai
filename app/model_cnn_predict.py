import os
import cv2
import numpy as np
from keras.preprocessing import image
from app.utilities.load_cnn_model import load_cnn_model
from app.utilities.setting import Env

# 載入模型
model = load_cnn_model()

def predict_image(image_path):
    # 載入圖像並調整大小
    img = image.load_img(image_path, target_size=(640, 360))
    # 將圖像轉換為數組
    x = image.img_to_array(img)
    # 擴展數組的維度以匹配模型的輸入
    x = np.expand_dims(x, axis=0)
    # 進行預測
    predictions = model.predict(x)
    return predictions

def get_prediction(image_path):
    predictions = predict_image(image_path)
    # 將預測結果顯示在畫面上
    image = cv2.imread(image_path)
    if np.argmax(predictions) == 0:
        cv2.putText(image, "Backhand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif np.argmax(predictions) == 1:
        cv2.putText(image, "Backhand Loop", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif np.argmax(predictions) == 2:
        cv2.putText(image, "Forehand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif np.argmax(predictions) == 3:
        cv2.putText(image, "Forehand Loop", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    print(predictions)