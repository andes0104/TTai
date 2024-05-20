from app.utilities.setting import Env
from app.utilities.cnn_constructure import cnn_constructure

CNN_MODEL_FILE = Env().CNN_MODEL_FILE

def load_cnn_model():
    model = cnn_constructure()
    model.load_weights(CNN_MODEL_FILE)
    return model
