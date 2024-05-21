import os
from app.utilities.setting import Env
from keras.preprocessing.image import ImageDataGenerator
from app.utilities.cnn_constructure import cnn_constructure

def train_model():
            
    # 指定訓練數據集和驗證數據集的路徑
    IMAGE_ROOT_PATH = f"{os.getcwd()}/{Env().IMAGE_ROOT_PATH}"
    train_data_dir = f"{IMAGE_ROOT_PATH}/train"
    validation_data_dir = f"{IMAGE_ROOT_PATH}/test"

    target_size = (640, 360)  # 1920*1080的1/3

    # 定義圖像增強器，可根據需要調整參數
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # 圖像像素值縮放到0-1之間
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # 使用ImageDataGenerator從文件夾中加載圖像數據
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=target_size,  # 輸入圖像的大小
        batch_size=32,
        class_mode='categorical')  # 多類別分類

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical')

    # 建立模型
    model = cnn_constructure()

    # 訓練模型
    model.fit(
        train_generator,
        steps_per_epoch=45,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)

    # 儲存模型
    model.save(Env().CNN_MODEL_FILE)

    # 在訓練模型後，返回模型
    return model