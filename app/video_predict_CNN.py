import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from app.utilities.forehand_compare import forehand_compare
from app.utilities.backhand_compare import backhand_compare
from app.utilities.load_cnn_model import load_cnn_model
from app.model_cnn_predict import predict_image
from app.utilities.common import Common

def process_frame(frame, common, ho_model):
    # 使用 common 類別進行人體姿勢檢測
    image, results = common.mediapipe_detection(frame, ho_model)
    keypoints = Common.extract_keypoints(results)

    # 在全黑的背景上繪製特徵點
    black_background = np.zeros(frame.shape, dtype=np.uint8)
    common.draw_styled_landmarks(black_background, results)

    # 檢查是否有人體骨架
    if keypoints is None:
        print("No keypoints detected")
        return None
    if results.pose_landmarks is None or len(results.pose_landmarks.landmark) < 16:
        print("Nose, right shoulder, elbow, or wrist not detected")
        return None

    return black_background

def process_and_predict(frame):
    common = Common()
    holistic = common.mp_holistic
    with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as ho_model:
        # 縮小影像解析度
        frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        black_background = process_frame(frame, common, ho_model)
        # 檢查是否有人體骨架
        if black_background is not None:
            cv2.imwrite('frame.jpg', black_background)
            predictions = predict_image('frame.jpg')
            return predictions, frame

def video_predict_model(video_path):
    model = load_cnn_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_num += 1
            if not ret:
                break
            
            if frame_num % int(fps) in [15, 30]:
                future = executor.submit(process_and_predict, frame)
                predictions, frame_copy = future.result()

                # 將預測結果顯示在畫面上
                if np.argmax(predictions) == 0:
                    cv2.putText(frame_copy, "Backhand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif np.argmax(predictions) == 1:
                    cv2.putText(frame_copy, "Backhand Loop", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif np.argmax(predictions) == 2:
                    cv2.putText(frame_copy, "Forehand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif np.argmax(predictions) == 3:
                    cv2.putText(frame_copy, "Forehand Loop", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow("Prediction", frame_copy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()