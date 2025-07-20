from app.controller import data_manager

# --------------------------------------
# unit tests
# --------------------------------------
def test_get_training_data():
    training_data = data_manager.get_training_data()
    assert isinstance(training_data, list)
    if len(training_data) >0:
        assert isinstance(training_data[0], dict)
        keys = training_data[0].keys()
        assert "name" in keys
        assert "upload_date" in keys
        assert "path" in keys
