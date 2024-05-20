from app.utilities.common import Common
from app.utilities.setting import Env
import os
import cv2
import numpy as np

def image_pre_process():
    PRE_TRAIN_IMAGE = f"{os.getcwd()}/{Env().PRE_TRAIN_IMAGE}"
    BLACK_BACKGROUND = f"{os.getcwd()}/{Env().BLACK_BACKGROUND}"
    action_type_list = os.listdir(PRE_TRAIN_IMAGE)

    for action_type in action_type_list:
        action_subfolders = os.listdir(f"{PRE_TRAIN_IMAGE}/{action_type}")
        for subfolder in action_subfolders:
            image_list = os.listdir(f"{PRE_TRAIN_IMAGE}/{action_type}/{subfolder}")
            one_full_action = 1
            for image in image_list:
                image_path = f"{PRE_TRAIN_IMAGE}/{action_type}/{subfolder}/{image}"
                # 檢查文件是否存在
                if not os.path.isfile(image_path):
                    print(f"File does not exist: {image_path}")
                    continue
                # 讀取圖片
                frame = cv2.imread(image_path)
                # 將每個關鍵動作的frame，利用mediapipe轉換成背景為黑的人體骨架圖
                common = Common()
                holistic = common.mp_holistic # 取用Holistic模型
                with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as ho_model:
                    image, results = common.mediapipe_detection(frame, ho_model) # 進行人體姿勢檢測
                    keypoints = Common.extract_keypoints(results)

                    # 在全黑的背景上繪製特徵點
                    black_background = np.zeros(frame.shape, dtype=np.uint8)
                    draw_styled_landmarks = common.draw_styled_landmarks(black_background, results)

                    # 檢查是否有人體骨架
                    if keypoints is None:
                        print(f"No keypoints detected: {image_path}")
                        continue

                    # 檢查是否偵測到鼻子、右肩、右肘、右腕
                    if results.pose_landmarks is None or len(results.pose_landmarks.landmark) < 16:
                        print(f"Right shoulder, elbow, or wrist not detected: {image_path}")
                        continue

                    # 顯示人體骨架圖片
                    # cv2.imshow("Image", image) # 顯示原始圖片
                    # cv2.imshow("Skeleton", black_background) # 顯示人體骨架圖片
                    # cv2.waitKey(500) # 等待0.5秒
                    # cv2.destroyAllWindows() # 關閉所有窗口          

                    action_path = f"{BLACK_BACKGROUND}/{action_type}/{subfolder}"
                    if not os.path.exists(action_path):
                        os.makedirs(action_path)
                    # 將人體骨架圖片保存到目標資料夾
                    if black_background is not None:
                        cv2.imwrite(f"{action_path}/{str(one_full_action)}.jpg", black_background)
                    else:
                        print(f"Image not processed correctly: {image_path}")
                    one_full_action += 1