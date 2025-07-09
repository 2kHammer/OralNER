from dataclasses import asdict

from app.model.data_provider.data_registry import data_registry

def get_training_data():
    training_data = data_registry.list_training_data()
    return [asdict(td) for td in training_data]

def add_training_data(dataset_name, filename, file):
    if data_registry.add_training_data(file):
        return True
    else:
        return False