from app.utilities.common import Common
import cv2
import os
import numpy as np

def save_frame_to_npy(
    action_type: str, npy_path: str, video: str, action_frame_list: list):
    common = Common()
    holistic = common.mp_holistic
    one_full_action = 1
    with holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as ho_model:
        for action_frame in action_frame_list:
            # 每個動作的開始與結束幀
            start_frame = action_frame["start"]
            end_frame = action_frame["end"]
            # 讀取影片並設定當前幀為開始幀
            cap = cv2.VideoCapture(video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_num = 1
            # 將該動作的每30幀存成一個npy
            while frame_num <= (end_frame - start_frame + 1):
                _, frame = cap.read()
                if frame_num % 30 == 0:
                    image, results = common.mediapipe_detection(frame, ho_model)
                    keypoints = Common.extract_keypoints(results)
                    action_path = f"{npy_path}/{action_type}/{str(one_full_action)}"
                    if not os.path.exists(action_path):
                        os.makedirs(action_path)
                    np.save(f"{action_path}/{str(frame_num)}", keypoints)
                frame_num += 1

            one_full_action += 1

def create_npy(npy_path: str):
    PRE_TRAIN_VIDEO = f"{os.getcwd()}/{Env().PRE_TRAIN_VIDEO}"
    action_type_list = os.listdir(PRE_TRAIN_VIDEO)

    for action_type in action_type_list:
        video_list = os.listdir(f"{PRE_TRAIN_VIDEO}/{action_type}")
        one_full_action = 1
        for video in video_list:
            video = f"{PRE_TRAIN_VIDEO}/{action_type}/{video}"
            # 將每個完整動作的frame存成npy
            common = Common()
            holistic = common.mp_holistic
            with holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as ho_model:
                # 讀取影片
                cap = cv2.VideoCapture(video)
                frame_num = 1
                while cap.isOpened():
                    _, frame = cap.read()
                    if frame is None:
                        break
                    image, results = common.mediapipe_detection(frame, ho_model)
                    keypoints = Common.extract_keypoints(results)
                    action_path = f"{npy_path}/{action_type}/{str(one_full_action)}"
                    if not os.path.exists(action_path):
                        os.makedirs(action_path)
                    np.save(f"{action_path}/{str(frame_num)}", keypoints)
                    frame_num += 1
                    if frame_num > 30:  # 檢查是否已經存滿30幀
                        one_full_action += 1  # 換到下一個資料夾
                        frame_num = 1  # 重設幀數