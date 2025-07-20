from dataclasses import asdict

from app.model.data_provider.data_registry import data_registry

# --------------------------------------
# public functions
# --------------------------------------
def get_training_data():
    """
    Returns the training data metadata

    Returns:
    (dict): in the format `of training data`
    """
    training_data = data_registry.list_training_data()
    return [asdict(td) for td in training_data]

def add_training_data(dataset_name, filename, file):
    """
    Adds a training dataset

    Parameters:
    dataset_name (str): name of the dataset to add
    filename (str): name of the file to add
    file (file): the file to add

    Returns:
    (bool): True if successful
    """
    if data_registry.add_training_data(dataset_name, filename, file):
        return True
    else:
        return False