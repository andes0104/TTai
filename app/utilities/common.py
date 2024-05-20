import cv2
import mediapipe as mp
import numpy as np
from app.utilities.setting import Env
import os

PRE_TRAIN_VIDEO = Env().PRE_TRAIN_VIDEO
PRE_TRAIN_IMAGE = Env().PRE_TRAIN_IMAGE
NPY_ROOT_PATH = Env().NPY_ROOT_PATH
IMAGE_ROOT_PATH = Env().IMAGE_ROOT_PATH


class Common:
    def __init__(self) -> None:
        self.mp_holistic = mp.solutions.holistic  # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    def mediapipe_detection(self, image, model):
        """
        Make detections
        Args:
            image (_type_): _description_
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        if image is None:
            print("Image not loaded correctly")
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(self, image, results):
        """
        繪製特徵點

        Args:
            image (_type_): _description_
            results (_type_): _description_
        """
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION
        )  # Draw face connections
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
        )  # Draw pose connections
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )  # Draw left hand connections
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )  # Draw right hand connections

    def draw_styled_landmarks(self, image, results):
        """
        繪製特徵點

        Args:
            image (_type_): _description_
            results (_type_): _description_
        """
        # Draw face connections
        # self.mp_drawing.draw_landmarks(
        #     image,
        #     results.face_landmarks,
        #     self.mp_holistic.FACEMESH_TESSELATION,
        #     self.mp_drawing.DrawingSpec(
        #         color=(80, 110, 10), thickness=1, circle_radius=1
        #     ),
        #     self.mp_drawing.DrawingSpec(
        #         color=(80, 256, 121), thickness=1, circle_radius=1
        #     ),
        # )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(80, 22, 10), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(80, 44, 121), thickness=2, circle_radius=2
            ),
        )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(121, 22, 76), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(121, 44, 250), thickness=2, circle_radius=2
            ),
        )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            ),
        )

    @staticmethod
    def extract_keypoints(results):
        if results is None:
            print("Results object is None")
            return None
        
        pose = (
            np.array(
                [
                    [res.x, res.y, res.z, res.visibility]
                    for res in results.pose_landmarks.landmark
                ]
            ).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 4)
        )
        # face = (
        #     np.array(
        #         [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        #     ).flatten()
        #     if results.face_landmarks
        #     else np.zeros(468 * 3)
        # )
        lh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3)
        )
        rh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3)
        )
        return np.concatenate([pose, lh, rh])

    @staticmethod
    def creat_labels(npy_path) -> list:
        sequences, labels = [], []
        actions = os.listdir(npy_path)
        label_map = {label: num for num, label in enumerate(actions)}

        for action in actions:
            for sequence in np.array(os.listdir(os.path.join(npy_path, action))):
                window = []
                for npy_file in os.listdir(os.path.join(npy_path, action, sequence)):
                    res = np.load(
                        os.path.join(
                            npy_path,
                            action,
                            str(sequence),
                            npy_file,
                        )
                    )
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
        return labels, sequences