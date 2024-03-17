from dotenv import load_dotenv
import os
import json
import numpy as np


class Env:
    # 環境變數
    def __init__(self) -> None:
        load_dotenv()

        self.NPY_ROOT_PATH = os.getenv("NPY_ROOT_PATH")
        self.MODEL_FILE = os.getenv("MODEL_FILE")
        self.RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
        self.PRE_TRAIN_VIDEO = os.getenv("PRE_TRAIN_VIDEO")
