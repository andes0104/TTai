from app.create_npy import create_npy
from app.utilities.pre_process_data import pre_process
from app.train_model import train_model
from app.model_predict import get_accuracy_score
from app.video_predict_model import video_predict_model
import os
from app.utilities.setting import Env
from app.utilities.common import Common
from app.utilities.image_capture import video_split

if __name__ == "__main__":
    video_split(video_path="./test_video/forehand/forehand12.mov", save_path="./test_pre_train_image/forehand12")

    # NPY_PATH = f"{os.getcwd()}/{Env().NPY_ROOT_PATH}"
    # # 剪片、製作每個動作的npy
    # create_npy(npy_path=NPY_PATH)

    # # 把訓練好的npy切割成測試、訓練集
    # data = pre_process(npy_path=NPY_PATH)

    # # 訓練模型
    # train_model(data=data)

    # 預測模型
    # get_accuracy_score(npy_path=NPY_PATH)

    # video_predict_model()