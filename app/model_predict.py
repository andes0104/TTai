from app.utilities.pre_process_data import pre_process
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from app.utilities.load_model import load_model


def get_accuracy_score(npy_path: str):
    model = load_model()
    data = pre_process(npy_path=npy_path)

    y_true = np.argmax(data["y_test"], axis=1).tolist()
    y_pred = []

    for x_sample in data["X_test"]:
        # 将样本转换为 NumPy 数组，并添加一个维度，以匹配模型的输入形状
        x_sample = np.array([x_sample])

        # 预测单个样本
        y_hat = model.predict(x_sample)

        # 获取预测的类别
        predicted_class = np.argmax(y_hat, axis=1)[0]
        y_pred.append(predicted_class)

    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print("Confusion Matrices: ", confusion_matrices)
    print("Accuracy: ", accuracy)
