from app.utilities.setting import Env
import os
import random
import shutil

# 設定圖片資料庫來源路徑
BLACK_BACKGROUND = Env().BLACK_BACKGROUND

# 設定圖片資料庫的儲存路徑
DATASET_PATH = Env().IMAGE_ROOT_PATH
# 設定訓練集與測試集的比例
TRAIN_TEST_SPLIT_RATIO = 0.8

def split_dataset():
    # 建立訓練集與測試集的資料夾
    train_dir = os.path.join(DATASET_PATH, "train")
    test_dir = os.path.join(DATASET_PATH, "test")

    # 獲取所有類別
    categories = os.listdir(BLACK_BACKGROUND)

    # 為每個類別創建訓練集和測試集的資料夾
    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # 獲取每個類別下的所有類別資料夾
        action_dirs = os.listdir(os.path.join(BLACK_BACKGROUND, category))

        # 為每個黑背景動作資料夾的圖片分配到訓練集或測試集
        for action_dir in action_dirs:
            image_files = os.listdir(os.path.join(BLACK_BACKGROUND, category, action_dir))

            for image_file in image_files:
                # 檢查是否為圖片檔
                if image_file.endswith(".jpg") or image_file.endswith(".png"):
                    # 產生隨機數字
                    random_num = random.random()

                    # 決定圖片應該添加到訓練還是測試數據集
                    if random_num < TRAIN_TEST_SPLIT_RATIO:
                        dest_dir = os.path.join(train_dir, category)
                    else:
                        dest_dir = os.path.join(test_dir, category)

                    # 複製圖片至目標資料夾
                    src_path = os.path.join(BLACK_BACKGROUND, category, action_dir, image_file)
                    shutil.copy(src_path, dest_dir)