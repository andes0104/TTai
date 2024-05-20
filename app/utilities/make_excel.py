import cv2
import pandas as pd
import os
from app.utilities.common import Common
from app.utilities.setting import Env

def create_excel_with_coordinates():
    # 指定圖片路徑
    PRE_TRAIN_IMAGE = f"{os.getcwd()}/{Env().PRE_TRAIN_IMAGE}"
    action_type_list = os.listdir(PRE_TRAIN_IMAGE)
    
    # 創建一個ExcelWriter物件
    with pd.ExcelWriter('coord.xlsx') as writer:
        
        # 假設你有一個迴圈來處理不同的類別
        for action_type in action_type_list:
            action_subfolders = os.listdir(f"{PRE_TRAIN_IMAGE}/{action_type}")

            # 創建一個空的DataFrame來存儲坐標數據
            df = pd.DataFrame(columns=['nose_x', 'nose_y', 'right_wrist_x', 'right_wrist_y', 'right_elbow_x', 'right_elbow_y', 'right_shoulder_x', 'right_shoulder_y'], dtype=float)

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
                    holistic = common.mp_holistic # 取用Holistic模型

                    # 抓取1個坐標
                    num_coordinates = 1

                    # 開始捕捉坐標
                    with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as ho_model:
                        image, results = common.mediapipe_detection(frame, ho_model) # 進行人體姿勢檢測
                        keypoints = Common.extract_keypoints(results)
                        frame = cv2.imread(image_path)

                        # 檢查是否有人體骨架
                        if keypoints is None:
                            print(f"No keypoints detected: {image_path}")
                            continue

                        for _ in range(num_coordinates):
                            image, results = common.mediapipe_detection(frame, ho_model) # 進行人體姿勢檢測

                            # 檢查是否偵測到右腕、右肘、右肩、鼻子
                            if results.pose_landmarks is None or len(results.pose_landmarks.landmark) < 16:
                                print(f"Right shoulder, elbow, or wrist not detected: {image_path}")
                                continue

                            # 從results中獲取鼻子的坐標
                            nose_x = results.pose_landmarks.landmark[holistic.PoseLandmark.NOSE].x
                            nose_y = results.pose_landmarks.landmark[holistic.PoseLandmark.NOSE].y
                            
                            # 從results中獲取右手腕的坐標
                            right_wrist_x = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_WRIST].x
                            right_wrist_y = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_WRIST].y

                            # 從results中獲取右手肘的坐標
                            right_elbow_x = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ELBOW].x
                            right_elbow_y = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ELBOW].y

                            # 從results中獲取右肩的坐標
                            right_shoulder_x = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_SHOULDER].x
                            right_shoulder_y = results.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_SHOULDER].y

                            # 印出圖片的路徑和坐標
                            print(f"""Path: {image_path},
                                   Nose: ({nose_x}, {nose_y}),
                                   Right Wrist: ({right_wrist_x}, {right_wrist_y}), 
                                   Right Elbow: ({right_elbow_x}, {right_elbow_y}), 
                                   Right Shoulder: ({right_shoulder_x}, {right_shoulder_y})""")

                            # 創建一個新的DataFrame來存儲坐標
                            new_df = pd.DataFrame({
                                'nose_x': [nose_x], 'nose_y': [nose_y], 
                                'right_wrist_x': [right_wrist_x], 'right_wrist_y': [right_wrist_y], 
                                'right_elbow_x': [right_elbow_x], 'right_elbow_y': [right_elbow_y], 
                                'right_shoulder_x': [right_shoulder_x], 'right_shoulder_y': [right_shoulder_y]
                            }, dtype=float)

                            # 檢查原來的DataFrame是否是空的或者所有的條目都是NA
                            if df.empty or df.isna().all().all():
                                df = new_df
                            else:
                                df = pd.concat([df, new_df], ignore_index=True)
                            
            # 將DataFrame保存為Excel文件
            df.to_excel(writer, sheet_name=action_type, index=False)
