import cv2
import mediapipe as mp

def forehand_compare(video_path):
    # 初始化 MediaPipe Holistic 模型
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    # 開啟影片檔案
    cap = cv2.VideoCapture(video_path)

    frame_count = 0  # 新增幀數計數器

    # 創建一個字典來儲存右手腕和鼻子的相對位置
    relative_position = {"x_diff": None, "y_diff": None, "position": None}

    while cap.isOpened():
        # 讀取一幀影片
        ret, frame = cap.read()
        if not ret:
            break

        # 轉換影格顏色空間 BGR 到 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 分析影格
        results = holistic.process(image_rgb)

        # 提取鼻子和右手腕的坐標
        nose_landmarks = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        R_wrist_landmarks = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

        # 獲取影格的高度和寬度
        h, w, c = frame.shape

        # 在影格上繪製特徵點和連接線
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if nose_landmarks and R_wrist_landmarks and frame_count % 30 == 0:  # 只有當幀數是30的倍數時，才進行分析
            # 將鼻子和右手腕的坐標轉換為像素坐標
            nose_x, nose_y = int(nose_landmarks.x * w), int(nose_landmarks.y * h)
            Rwrist_x, Rwrist_y = int(R_wrist_landmarks.x * w), int(R_wrist_landmarks.y * h)

            # 計算坐標差
            x_diff = Rwrist_x - nose_x
            y_diff = Rwrist_y - nose_y

            # 更新字典
            relative_position["x_diff"] = x_diff
            relative_position["y_diff"] = y_diff

            # 比較右手腕和鼻子的相對位置
            if x_diff > 0 and y_diff > 0:
                relative_position["position"] = "forehand"
            elif x_diff > 0 and y_diff < 0:
                relative_position["position"] = "forehand loop"

            # 在影格上繪製鼻子和右手腕的坐標
            cv2.circle(frame, (nose_x, nose_y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (Rwrist_x, Rwrist_y), 8, (255, 0, 0), -1)

        # 在影格中顯示正手狀態
        cv2.putText(frame, f'forehand Status: {relative_position["position"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # 顯示處理後的影格
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # 更新幀數計數器

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

    # 返回右手腕和鼻子的相對位置
    return relative_position

# 設定影片路徑
video_path = "forehand loop20.mov"

# 呼叫正手狀態比較 function
forehand_status = forehand_compare(video_path)
print(forehand_status)
