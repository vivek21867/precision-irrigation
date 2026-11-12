"""Model registry."""
from src.models.bilstm import BiLSTMRegressor
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor

MODEL_REGISTRY = {
    "lstm": LSTMRegressor,
    "bilstm": BiLSTMRegressor,
    "transformer": TransformerRegressor,
}

MODEL_NAMES = {"lstm": "LSTM", "bilstm": "BiLSTM", "transformer": "Transformer"}


def build_model(name: str, n_features: int, params: dict):
    """Instantiate a model by registry name with hyperparameters from config."""
    return MODEL_REGISTRY[name](n_features=n_features, **params)
