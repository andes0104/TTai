import cv2
import mediapipe as mp
import math
from app.utilities.common import Common
from app.utilities.setting import Env
import numpy as np
import os


class poseDetector:
    def __init__(
        self,
        mode=False,
        upBody=False,
        smooth=True,
        enableSeg=False,
        smoothSeg=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.upBody,
            self.smooth,
            self.enableSeg,
            self.smoothSeg,
            self.detectionCon,
            self.trackCon,
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                # if draw:
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(
                img,
                str(int(angle)),
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        return angle


def save_frame_to_npy(
    action_type: str, npy_path: str, video: str, action_frame_list: list
):
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
            # 將該動作的每一幀存成一個npy
            for frame in range(start_frame, end_frame + 1):
                _, frame = cap.read()
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
                    if frame_num > 30:  # 檢查當前幀數是否超過30
                        one_full_action += 1  # 換到下一個資料夾
                        frame_num = 1  # 重設幀數