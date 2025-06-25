from dataclasses import dataclass
from ..framework_provider.framework import FrameworkNames
from enum import Enum

class ModelState(Enum):
    BASE = 1
    MODIFIED =2
    IN_TRAINING = 3

@dataclass
class TrainingResults:
    f1: float
    precision : float
    recall: float
    duration : float
    accuracy : float

class NERModel:
    def __init__(self, state:int,name: str, framework_name: FrameworkNames, base_model_name: str, storage_path: str):
        self.id = None
        self.state = ModelState(state)
        self.name = name
        self.framework_name = framework_name
        self.base_model_name = base_model_name
        self.storage_path = storage_path
        self.trainings = []
    
    def append_training(self, date:int, dataset_name:str, dataset_id: int,results: TrainingResults,trainings_args: dict):
        self.trainings.append({
            "date": date,
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "trainings_args":trainings_args,
            "metrics": results,
        })

    def set_id(self, id:int):
        self.id = id

    def set_state(self, state:int):
        if state > 0 and state < 4:
            self.state = ModelState(state)
        else:
            raise ValueError("Invalid state")


    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state.name,
            "name": self.name,
            "framework_name": self.framework_name.name,
            "base_model_name": self.base_model_name,
            "storage_path": self.storage_path,
            "trainings": [
                {
                    "date": t["date"],
                    "dataset_name": t["dataset_name"],
                    "dataset_id": t["dataset_id"],
                    "trainings_args": t["trainings_args"],
                    "metrics": t["metrics"].__dict__
                }
                for t in self.trainings
            ]
        }
    @classmethod
    def from_dict(cls, d: dict):
        model = cls(
            state=ModelState[d["state"]],
            name=d["name"],
            framework_name=FrameworkNames[d["framework_name"]],
            base_model_name=d["base_model_name"],
            storage_path=d["storage_path"]
        )

        for training in d.get("trainings", []):
            metrics = TrainingResults.from_dict(training["metrics"])
            trainings_args = training.get("trainings_args")
            model.append_training(
                date=training["date"],
                dataset_name=training["dataset_name"],
                dataset_id=training["dataset_id"],
                results=metrics,
                trainings_args=trainings_args
            )
        model.id = d["id"]

        return model
    
