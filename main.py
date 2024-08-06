from app.utilities.setting import Env
from app.utilities.common import Common
from app.utilities.create_data import split_dataset
# from app.utilities.image_capture import video_split
# from app.utilities.make_excel import create_excel_with_coordinates
from app.utilities.image_to_skeleton import image_to_skeleton
from app.image_to_blackbackground import image_pre_process
from app.train_cnn_model import train_model
from app.model_cnn_predict import get_prediction
from app.video_predict_CNN import video_predict_model
import os
from app.utilities.video_to_jpg import extract_frames

if __name__ == "__main__":
    # 影片切割成每個動作的frame
    # video_split()

    # 抓取關鍵動作frame中的鼻子、右腕、右肘、右肩的坐標，並存成excel檔
    # create_excel_with_coordinates()

    # 關鍵動作frame轉換為背景為黑的人體骨架圖
    # image_pre_process()

    # 關鍵動作frame轉換為人體骨架圖
    # image_to_skeleton()

    # 將黑背景的人體骨架圖片，分割成訓練集與測試集
    # split_dataset()

    # 訓練模型
    # train_model()

    # 預測模型
    # get_prediction("1.jpg")

    # video_predict_model('backhand loop20.mov')

    # 影片轉換成圖片
    video_path = "output.mp4"
    output_dir = "result_image/backhand loop"
    extract_frames(video_path, output_dir)