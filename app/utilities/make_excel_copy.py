import cv2
import csv
import os
from app.utilities.common import Common
from app.utilities.setting import Env

def create_excel_with_coordinates():
    # 指定圖片路徑
    PRE_TRAIN_IMAGE = f"{os.getcwd()}/{Env().PRE_TRAIN_IMAGE}"
    action_type_list = os.listdir(PRE_TRAIN_IMAGE)
    
    save = []

    # 讀取圖片
    for action_type in action_type_list:
        action_subfolders = os.listdir(f"{PRE_TRAIN_IMAGE}/{action_type}")
        for subfolder in action_subfolders:
            image_list = os.listdir(f"{PRE_TRAIN_IMAGE}/{action_type}/{subfolder}")
            for image in image_list:
                image_path = f"{PRE_TRAIN_IMAGE}/{action_type}/{subfolder}/{image}"
                # 檢查文件是否存在
                if not os.path.isfile(image_path):
                    print(f"File does not exist: {image_path}")
                    continue
                # 讀取圖片
                frame = cv2.imread(image_path)

                # 將指定的坐標保存到Excel文件中
                common = Common() # 初始化mediapipe.holistic
                holistic = common.mp_holistic

                # 開始捕捉坐標
                with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as ho_model:
                    image, results = common.mediapipe_detection(frame, ho_model)

                    keypoints = Common.extract_keypoints(results)
                    frame = cv2.imread(image_path)

                    # 檢查是否有人體骨架
                    if keypoints is None:
                        print(f"No keypoints detected: {image_path}")
                        continue

                    # 檢查是否偵測到右腕、右肘、右肩、鼻子
                    if results.pose_landmarks is None or len(results.pose_landmarks.landmark) < 16:
                        print(f"Right shoulder, elbow, or wrist not detected: {image_path}")
                        continue

                    coordinates = []

                    for _ in range(1):
                        image, results = common.mediapipe_detection(frame, ho_model)

                        if results.pose_landmarks is not None:
                            # 從results中獲取鼻子的坐標
                            nose_x = results.pose_landmarks.landmark[holistic.PoseLandmark.NOSE].x
                            nose_y = results.pose_landmarks.landmark[holistic.PoseLandmark.NOSE].y

                            # 從results中獲取右腕的坐標
                            right_wrist_x = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_WRIST].x
                            right_wrist_y = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_WRIST].y

                            # 從results中獲取右肘的坐標
                            right_elbow_x = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ELBOW].x
                            right_elbow_y = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ELBOW].y

                            # 從results中獲取右肩的坐標
                            right_shoulder_x = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_SHOULDER].x
                            right_shoulder_y = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_SHOULDER].y

                            # 運算出前臂的斜率
                            slope = (right_wrist_y - right_elbow_y) / (right_wrist_x - right_elbow_x)

                        coordinates.append([nose_x, nose_y, right_wrist_x, right_wrist_y, right_elbow_x, right_elbow_y, right_shoulder_x, right_shoulder_y, slope])
                    
                    save.append([image_path, coordinates])
                    print(f"""Path: {image_path}, coordinates: {coordinates}""")

    # 將坐標寫入Excel文件
    with open("coordinates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Path", "Nose X", "Nose Y", "Right Wrist X", "Right Wrist Y", "Right Elbow X", "Right Elbow Y", "Right Shoulder X", "Right Shoulder Y", "Slope"])
        for data in save:
            writer.writerow([data[0]] + data[1][0])