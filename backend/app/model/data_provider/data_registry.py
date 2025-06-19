import csv

class DataRegistry:
    _instance = None
    
    # Singleton
    def __new__(cls, *args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def loadTrainingData(self,path):
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                print(row)
                if i == 10:
                    break
        
    def _prepareADG():
        pass