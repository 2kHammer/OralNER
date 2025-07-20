import io
import time

import pytest

from app.model.data_provider.data_registry import data_registry
from app.controller import ner_manager

# --------------------------------------
# unit tests
# --------------------------------------

def test_start_ner_false_file():
    assert ner_manager.start_ner(["Ein beliebiger Eingabetext"],True, True) == "-1"
    fake_file = io.StringIO("Virtuelle; csv; \n Virtuelle2; csv2")
    assert ner_manager.start_ner(fake_file, True, True) == "-1"

def test_get_ner_results_false_job_id():
    with pytest.raises(KeyError):
        ner_manager.get_ner_results("-1")

# --------------------------------------
# integration tests
# --------------------------------------
def test_start_ner(dataset_id = 2):
        path = data_registry._datasets[dataset_id].path
        path = path.replace(".json",".csv").replace("/Converted","")
        with open(path,"r") as f:
            file = f.read().splitlines()
            job_id = ner_manager.start_ner(file, True, True)
            assert job_id in ner_manager.ner_jobs

        results = None
        while not results:
            results = ner_manager.get_ner_results(job_id)
            time.sleep(5)
            print(results)

        print(results)
        assert len(results) == 3
        assert isinstance(results[0], list)