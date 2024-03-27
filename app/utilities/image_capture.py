import cv2
import os
import time

def video_split(video_path, save_path):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while cap.isOpened():
        rval, frame = cap.read()
        if not rval:
            break
        if frame_num % 30 == 0: # 每隔30幀擷取一張frame
            cv2.imwrite(f"{save_path}/{str(frame_num//30)}.jpg", frame) 
            cv2.imshow('Frame', frame)
            cv2.waitKey(100)
            print(f"{save_path}/{str(frame_num//30)}.jpg") # 顯示儲存檔案名稱
        frame_num += 1
    cap.release() # 停止讀取影片
    cv2.destroyAllWindows()
    end_time = time.time()
    print(f"Time taken for {video_path}: {end_time - start_time} seconds")

DATA_DIR = "./test_video/backhand loop"
SAVE_DIR = "./test_pre_train_image"

for video in os.listdir(DATA_DIR):
    video_path = f"{DATA_DIR}/{video}"
    video_name = os.path.splitext(video)[0]  # 獲取不含副檔名的影片名稱
    save_path = f"{SAVE_DIR}/backhand loop/{video_name}"  # 使用影片名稱建立儲存路徑
    os.makedirs(save_path, exist_ok=True)
    video_split(video_path, save_path)