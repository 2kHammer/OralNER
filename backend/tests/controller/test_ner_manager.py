import time

from app.model.data_provider.data_registry import data_registry
from app.controller import ner_manager

def test_start_ner(dataset_id = 2):
        path = data_registry._datasets[dataset_id].path
        path = path.replace(".json",".csv").replace("/Converted","")
        with open(path,"r") as f:
            job_id = ner_manager.start_ner(f, True, True)
            assert job_id in ner_manager.ner_jobs

        results = None
        while not results:
            results = ner_manager.get_ner_results(job_id)
            time.sleep(5)
            print(results)

        assert len(results) == 3