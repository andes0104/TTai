import cv2
import os
import time

def video_split(video_path, save_path):
    vc = cv2.VideoCapture(video_path)
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
        cv2.imshow('frame', frame)
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        if c %30 == 0:
            cv2.imwrite(f"save_path/{str(c//30)}.jpg", frame)
            cv2.waitKey(1)
            print(f"save_path/{str(c//30)}.jpg")
        c += 1
    vc.release()

DATA_DIR = "./test_video/forehand"
SAVE_DIR = "./test_pre_train_image"

start_time = time.time()

for video in os.listdir (DATA_DIR):
    video_path = f"{DATA_DIR}/{video}"
    video_name = os.path.splitext(video)[0]  # 獲取不含副檔名的影片名稱
    save_path = f"{SAVE_DIR}/{video_name}"  # 使用影片名稱建立儲存路徑
    os.makedirs(save_path, exist_ok=True)
    video_split(video_path, save_path)

print(f"--- {time.time() - start_time} seconds ---")
