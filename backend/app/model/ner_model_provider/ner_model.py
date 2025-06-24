from dataclasses import dataclass

@dataclass
class TrainingResults:
    date : int
    f1: float
    prec : float
    recall: float
    duration : float

class NERModel:
    def __init__(self, base: bool, name: str, framework_name: str, base_model_name: str, storage_path: str):
        self.id = None
        self.base = base
        self.name = name
        self.framework_name = framework_name
        self.base_model_name = base_model_name
        self.storage_path = storage_path
        self.trainings = []
    
    def append_training(self, date:int, dataset_name:str, dataset_id: int,results: TrainingResults,parameters):
        self.trainings.append({
            "date": date,
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "parameter":parameters,
            "metrics": results,
            "parameters": parameters
        })

    def set_id(self, id:int):
        self.id = id

    def to_dict(self):
        return {
            "id": self.id,
            "base": self.base,
            "name": self.name,
            "framework_name": self.framework_name,
            "base_model_name": self.base_model_name,
            "storage_path": self.storage_path,
            "trainings": [
                {
                    "date": t["date"],
                    "dataset_name": t["dataset_name"],
                    "dataset_id": t["dataset_id"],
                    "parameters": t["parameters"],
                    "metrics": t["metrics"].__dict__
                }
                for t in self.trainings
            ]
        }
    @classmethod
    def from_dict(cls, d: dict):
        model = cls(
            id=d["id"],
            base=d["base"],
            name=d["name"],
            framework_name=d["framework_name"],
            base_model_name=d["base_model_name"],
            storage_path=d["storage_path"]
        )

        for training in d.get("trainings", []):
            metrics = TrainingResults.from_dict(training["metrics"])
            parameters = training.get("parameters")
            model.append_training(
                date=training["date"],
                dataset_name=training["dataset_name"],
                dataset_id=training["dataset_id"],
                results=metrics,
                parameters=parameters
            )

        return model
    
