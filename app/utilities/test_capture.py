import cv2
from common import Common
from matplotlib import pyplot as plt


def testCapture():
    common = Common()
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with common.mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            # Read feed
            _, frame = cap.read()

            # Make detections
            image, results = common.mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            common.draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow("OpenCV Feed", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    common.draw_landmarks(frame, results)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    testCapture()
