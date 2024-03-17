from app.utilities.setting import Env
from app.utilities.model_constructure_init import model_constructure

MODEL_FILE = Env().MODEL_FILE


def load_model():
    model = model_constructure()
    model.load_weights(MODEL_FILE)
    return model
