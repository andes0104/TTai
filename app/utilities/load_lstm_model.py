from app.utilities.setting import Env
from app.utilities.lstm_constructure import model_constructure

LSTM_MODEL_FILE = Env().LSTM_MODEL_FILE

def load_lstm_model():
    model = model_constructure()
    model.load_weights(LSTM_MODEL_FILE)
    return model
