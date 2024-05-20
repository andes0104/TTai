import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def cnn_constructure():
    # 指定使用 GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 指定使用第一個 GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # 發生錯誤時打印錯誤訊息
            print(e)

    # 建立一個Sequential模型
    model = Sequential()
    
    # 添加第一層卷積層+池化層
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(640, 360, 3)))  # 32個3x3的卷積核
    model.add(MaxPooling2D((2, 2)))
    
    # 添加第二層卷積層+池化層
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # 添加第三層卷積層+池化層
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # 將3D的特徵圖展平為1D的特徵向量
    model.add(Flatten())
    
    # 添加全連接層
    model.add(Dense(128, activation='relu'))
    # 添加輸出層，4個類別
    model.add(Dense(4, activation='softmax'))
    
    # 編譯模型
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # 顯示模型架構
    model.summary()
    return model