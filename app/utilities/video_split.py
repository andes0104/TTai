import cv2
import time
import os

DATA_DIR = "./test_video/forehand loop"
SAVE_DIR = "./test_pre_train_image"


def video_split(video_name, time_F):
    video_images = []
    cap = cv2.VideoCapture(video_name)
    frame_num = 0

    while cap.isOpened():  # 使用 while 迴圈持續檢查影片是否開啟
        rval, video_frame = cap.read()  # 使用 read() 讀取影片的每一幀
        if rval:
            if frame_num % time_F == 0:
                video_images.append(video_frame)
            frame_num += 1
        else:
            break

    cap.release()

    return video_images


start_time = time.time()

time_F = 30  # 每隔30幀擷取一次
video_name = os.path.join(DATA_DIR, "forehand loop19.mov")  # 影片名稱
video_images = video_split(video_name, time_F)  # 讀取影片並轉成圖片

save_subdir = os.path.join(SAVE_DIR, "./forehand loop/forehand loop20")
os.makedirs(save_subdir, exist_ok=True)  # 創建目錄

for i in range(0, len(video_images)):  # 顯示出所有擷取之圖片
    cv2.imshow("windows", video_images[i])
    cv2.imwrite(os.path.join(save_subdir, f"{i}.jpg"), video_images[i])
    print(f"save {save_subdir}/{i}.jpg")
    cv2.waitKey(100)

cv2.destroyAllWindows()

print(f"--- {time.time() - start_time} seconds ---")