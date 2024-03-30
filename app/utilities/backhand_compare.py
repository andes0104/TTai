import math
import cv2
import mediapipe as mp

def compare_backhand_elbow_angle(video_path):
    # 初始化 MediaPipe Holistic 模型
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    # 開啟影片檔案
    cap = cv2.VideoCapture(video_path)

    frame_count = 0  # 新增幀數計數器
    backhand_elbow_angles = []

    # 創建一個字典來儲存右手腕角度狀態
    elbow_angle_status = {"angle": None, "status": None}

    while cap.isOpened():
        # 讀取一幀影片
        ret, frame = cap.read()
        if not ret:
            break

        # 轉換影格顏色空間 BGR 到 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 分析影格
        results = holistic.process(image_rgb)

        # 提取右肩、右肘、右手腕的坐標
        right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

        # 在影格上繪製特徵點和連接線
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if frame_count % 30 == 0:  # 只有當幀數是30的倍數時，才進行分析
            # 計算右肘角度
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # 將右手肘角度添加到列表中
            backhand_elbow_angles.append(angle)

            # 判斷右手肘角度狀態，並更新字典
            elbow_angle_status["angle"] = angle
            if angle > 110 and angle < 125:
                elbow_angle_status["status"] = "backhand"
            elif angle > 115 and angle < 145:
                elbow_angle_status["status"] = "backhand loop"
            else:
                elbow_angle_status["status"] = "angle not in the range of backhand or backhand loop"

        # 在影格上顯示右手肘角度和狀態
        if angle is not None:  # 只有當角度不為空時，才顯示
            cv2.putText(frame, f'R-elbow Angle: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'R-elbow Status: {elbow_angle_status["status"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # 顯示影格
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # 更新幀數計數器

    # 釋放資源
    cap.release()

    # 顯示所有右手肘角度
    for angle in backhand_elbow_angles:
        print(angle)

def calculate_angle(p1, p2, p3):
    # 計算右手肘角度
    angle = math.degrees(
        math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    )

    return angle

# 設定影片路徑
video_path = 'backhand loop16.mov'

# 叫叫反拍手肘角度比較 function
compare_backhand_elbow_angle(video_path)