import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_cnn_model():
    model = tf.keras.Sequential()
    
    # 新增第一層卷積層
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    # 新增第二層卷積層
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # 將卷積層輸出平坦化
    model.add(Flatten())
    
    # 新增全連接層
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model

# 建立模型
cnn_model = create_cnn_model()

# 編譯模型
cnn_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# 顯示模型架構
cnn_model.summary()

# 輸出模型架構
tf.keras.utils.plot_model(cnn_model, to_file='cnn_model.png', show_shapes=True)

# 儲存模型架構
cnn_model.save('cnn_model.keras')