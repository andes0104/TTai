from dotenv import load_dotenv
import os

class Env:
    # 環境變數
    def __init__(self) -> None:
        load_dotenv()

        self.NPY_ROOT_PATH = os.getenv("NPY_ROOT_PATH")
        self.IMAGE_ROOT_PATH = os.getenv("IMAGE_ROOT_PATH")

        self.LSTM_MODEL_FILE = os.getenv("LSTM_MODEL_FILE")
        self.CNN_MODEL_FILE = os.getenv("CNN_MODEL_FILE")
        
        self.RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
        
        self.PRE_TRAIN_VIDEO = os.getenv("PRE_TRAIN_VIDEO")
        self.PRE_TRAIN_IMAGE = os.getenv("PRE_TRAIN_IMAGE")
        self.BLACK_BACKGROUND = os.getenv("BLACK_BACKGROUND")
        self.SKELETON = os.getenv("SKELETON")
