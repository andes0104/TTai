import cv2
import numpy as np
from app.utilities.load_model import load_model
from collections import deque
from app.utilities.common import Common

def video_predict_model():
    # 載入你的模型
    model = load_model()

    common = Common()
    holistic = common.mp_holistic
    queue = deque(maxlen=30)

    with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as ho_model:
        # 打開摄影機
        cap = cv2.VideoCapture('forehand loop20.mov')
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            # # 按下 a 開始選取
            # keyName = cv2.waitKey(0)
            # if keyName == ord('a'):
            # # 選取區域
            #     area = cv2.selectROI('armROI', frame, showCrosshair=False, fromCenter=False)
            #     print(area)
            #     x, y, w, h = area
            #     # 裁剪區域
            #     armROI = frame[y:y+h, x:x+w]

            #     cv2.imshow('armROI', armROI)

            if not ret:
                break

            image, results = common.mediapipe_detection(frame, ho_model)
            keypoints = Common.extract_keypoints(results)
            queue.append(keypoints)
            frame_num += 1

            # 在畫面中繪製特徵點
            draw_styled_landmarks = common.draw_styled_landmarks(frame, results)
            # 框選手臂擺動區域
            cv2.rectangle(frame, (200, 350),(1200, 800), (255, 255, 0), 2)
            cv2.putText(frame, "Arm swing finished location", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if frame_num >= 30:
                sequence = np.array(queue)  # 將隊列轉換為 numpy 陣列
                sequence = np.expand_dims(sequence, axis=0)  # 添加一個新的維度來表示批次大小

                # 使用模型進行預測
                prediction = model.predict(sequence)
                predicted_class = np.argmax(prediction)

                # 將預測結果添加到幀上
                cv2.putText(frame, f"Predicted class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                frame_num = 0  # 重設幀數
                queue.clear()  # 清空隊列

            # 顯示幀
            cv2.imshow('Frame', frame)

            # 如果按下 'q' 鍵，則退出循環
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()