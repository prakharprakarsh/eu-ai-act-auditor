import pytest

from models.synthetic_data import generate_hiring_dataset
from models.hiring_classifier import train_hiring_model


@pytest.fixture(scope="session")
def hiring_data():
    X, y = generate_hiring_dataset(n=500, seed=0, bias_strength=0.25)
    return X, y


@pytest.fixture(scope="session")
def trained_hiring_model(hiring_data):
    X, y = hiring_data
    return train_hiring_model(X, y)


@pytest.fixture(scope="session")
def protected_attrs(hiring_data):
    X, _ = hiring_data
    return {
        "gender": X["gender"],
        "age_band": X["age_band"],
        "ethnicity_proxy": X["ethnicity_proxy"],
    }
