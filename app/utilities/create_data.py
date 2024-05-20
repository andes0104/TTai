import os
import shutil
from app.utilities.setting import Env
from sklearn.model_selection import train_test_split

def get_category_image_count(data_dir):
    category_counts = {}
    categories = os.listdir(data_dir)
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            total_images = sum(len(files) for _, _, files in os.walk(category_dir))
            category_counts[category] = total_images
    return category_counts

def split_dataset():
    # 定義資料夾路徑
    data_dir = Env().BLACK_BACKGROUND
    # 定義儲存路徑
    dataset_path = Env().IMAGE_ROOT_PATH

    # 獲取所有類別資料夾
    categories = os.listdir(data_dir)

    # 創建訓練集和測試集的資料夾
    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 獲取每個類別的圖片總數
    category_counts = get_category_image_count(data_dir)

    # 定義圖片編號起始值
    start_index = 1

    # 迭代每個類別資料夾
    for category in categories:
        # 構建類別資料夾的完整路徑
        category_dir = os.path.join(data_dir, category)
        
        # 檢查是否為目錄
        if os.path.isdir(category_dir):
            # 獲取類別資料夾中的所有子資料夾
            subfolders = os.listdir(category_dir)
            
            # 迭代每個子資料夾
            for subfolder in subfolders:
                # 構建子資料夾的完整路徑
                subfolder_dir = os.path.join(category_dir, subfolder)
                
                # 檢查是否為目錄
                if os.path.isdir(subfolder_dir):
                    # 獲取子資料夾中的所有圖片檔
                    images = os.listdir(subfolder_dir)
                    
                    # 將圖片檔路徑和標籤分配到訓練集和測試集中
                    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
                    num_train_images = len(train_images)
                    num_test_images = len(test_images)

                    # 複製訓練集的圖片到訓練集資料夾中
                    for img in train_images:
                        src_path = os.path.join(subfolder_dir, img)
                        dest_dir = os.path.join(train_dir, category)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_file_name = f"{start_index}.jpg"  # 使用重新編號組成新的檔案名稱
                        shutil.copy(src_path, os.path.join(dest_dir, dest_file_name))
                        start_index += 1

                    # 複製測試集的圖片到測試集資料夾中
                    for img in test_images:
                        src_path = os.path.join(subfolder_dir, img)
                        dest_dir = os.path.join(test_dir, category)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_file_name = f"{start_index}.jpg"  # 使用重新編號組成新的檔案名稱
                        shutil.copy(src_path, os.path.join(dest_dir, dest_file_name))
                        start_index += 1

    # 確認訓練集和測試集的大小
    train_size = sum(len(files) for _, _, files in os.walk(train_dir))
    test_size = sum(len(files) for _, _, files in os.walk(test_dir))
    print("Number of training samples:", train_size)
    print("Number of testing samples:", test_size)
