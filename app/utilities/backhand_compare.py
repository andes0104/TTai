import math
import cv2
import mediapipe as mp

def backhand_compare(frame):
    # 初始化 MediaPipe Holistic 模型
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    frame_count = 0  # 新增幀數計數器
    backhand_elbow_angles = [] # 創建一個陣列來存儲每次分析的右手肘角度
    forearm_slopes = []  # 創建一個陣列來存儲每次分析的前臂斜率

    angle = None # 初始化右手肘角度

    # 創建一個字典來儲存右手腕角度狀態
    elbow_angle_status = {"angle": None, "status": None}

    # 轉換影格顏色空間 BGR 到 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 分析影格
    results = holistic.process(image_rgb)

    if results.pose_landmarks is None:  # 如果沒有偵測到人體關鍵點，則返回
        print("No pose landmarks detected in the frame.")
        return
    
    # 提取右肩、右肘、右手腕的坐標
    right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
    right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

    # 在影格上繪製特徵點和連接線
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # 框選手臂擺動區域
    cv2.rectangle(frame, (200, 350),(1200, 800), (255, 255, 0), 2)
    cv2.putText(frame, "Arm swing finished location", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 獲取影格的高度和寬度
    h, w, c = frame.shape

    if frame_count % 30 == 15:  # 只有當幀數是每秒的第15幀時，才進行分析
        # 計算右肘角度
        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # 計算前臂的斜率
        forearm_slope = (right_wrist.y - right_elbow.y) / (right_wrist.x - right_elbow.x)

        # 將右手肘角度添加到列表中
        backhand_elbow_angles.append(angle)

        # 將前臂斜率添加到列表中
        forearm_slopes.append(forearm_slope)

        # 判斷右手肘角度、前臂斜率狀態，並更新字典
        elbow_angle_status["angle"] = angle
        elbow_angle_status["forearm_slope"] = forearm_slope
        # if angle > 35 and angle < 70:
        #     elbow_angle_status["status"] = "backhand"
        # elif angle > 70 and angle < 130:
        #     elbow_angle_status["status"] = "backhand loop"
        
        if forearm_slope > -0.6 and forearm_slope < 0.0063:
            elbow_angle_status["status"] = "backhand"
        elif forearm_slope > 0.0063 and forearm_slope < 1.2:
            elbow_angle_status["status"] = "backhand loop"

    # 將右肩、右肘、右腕的坐標轉換為像素坐標
    right_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))
    right_elbow = (int(right_elbow.x * w), int(right_elbow.y * h))
    right_wrist = (int(right_wrist.x * w), int(right_wrist.y * h))

    # 在影格上繪製出關節點
    cv2.circle(frame, right_shoulder, 8, (0, 255, 0), -1)
    cv2.circle(frame, right_elbow, 8, (0, 255, 0), -1)
    cv2.circle(frame, right_wrist, 8, (0, 255, 0), -1)

    # 在影格上繪製出關節連線
    cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 2)
    cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 2)    

    # 在影格上顯示右手肘角度和狀態
    if angle is not None:  # 只有當角度不為空時，才顯示
        cv2.putText(frame, f'R-elbow angle: {angle:.5f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'R-forearm slope: {forearm_slope:.5f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'R-elbow Status: {elbow_angle_status["status"]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # 顯示影格
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 釋放資源
        cv2.destroyAllWindows()

    # 顯示所有右手肘角度
    for angle in backhand_elbow_angles: 
        print(angle)

    # 顯示每次分析的前臂斜率
    for slope in forearm_slopes:
        print(slope)
    
    # 返回右手肘角度狀態
    return elbow_angle_status

def calculate_angle(p1, p2, p3):
    # 計算右手肘角度
    angle = math.degrees(
        math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    )

    return angle