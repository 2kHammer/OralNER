import csv
from dataclasses import dataclass, asdict
from typing import List, Tuple
import spacy
import re
import json

from datasets import load_dataset

@dataclass
class ADGRow:
    idx: int
    unextracted: str
    num: int
    timestamp: str
    person: str
    text: str
    tokens: List[str]
    labels: List[str]
    entities: List[dict]
    other: List[Tuple[str, str]]

entities = {
    "PER":1,
    "ROLE":2,
    "ORG":3,
    "LOC":4,
    "WORK_OF_ART":5,
    "NORP":6,
    "EVENT":7,
    "DATE":8
}


class DataRegistry:
    _instance = None
    
    # Singleton
    def __new__(cls, *args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def loadTrainingData(self,path):
        rows = self._load_rows_as_json(path)
        return rows

    # works mit Path for now, has to be changed to file later
    def saveTrainingData(self,path):
        nlp = spacy.load("app/store/NER-Models/base/NLP/de_core_news_sm")
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile,delimiter=';')
            rows = []
            idx = 1
            for row in reader:
                if len(row) > 0:
                    rows.append(self._extract_ADG_row(row, nlp,idx))
                    idx += 1
            write_path = path.replace("Trainingsdata/","Trainingsdata/Converted/").replace(".csv",".json")
            self._save_rows_as_json(rows,write_path)


    # check the format of the data und try to make it better - if I have time
    def _extract_ADG_row(self,row, nlp,idx):
        """Return the ADGRow from a row
        If a annotated entity doesn't match to words or its only the type -> ist in else
        Arguments:
            -row: string of one ADG ROW
            -tokenizer: spacy tokenizer
        """

        # extract general infos from text
        full_row = "".join(row)
        first_column = row[0].split("\t")
        first_column.extend(row[1:])
        number = first_column.pop(0)
        ts = first_column.pop(0)
        speaker = first_column.pop(0)
        text = first_column.pop(0)
        other = []

        #extract entities from text
        entities = []
        pattern = "(.*)\[(PER|ROLE|ORG|LOC|WORK_OT_ART|NORP|EVENT|DATE)\]"
        for rest in first_column:
            if rest != '':
                match = re.match(pattern, rest)
                if match:
                    text_description_optional = match.group(1).split("[")
                    entities.append((text_description_optional[0].strip(),match.group(2)))
        #map entities to text
        entities_with_positions = []
        for index,entity in enumerate(entities):
            start_end_entities = []
            entity_text = entity[0]
            if len(entity_text) > 0:
                matches = re.finditer(re.escape(entity_text),text)
                for match in matches:
                    start_end_entities.append((match.span()[0],match.span()[1]))

                if len(start_end_entities) == 0:
                    other.append(entity)
                else:
                    entities_with_positions.append({
                        "entity_text": entity_text,
                        "typ": entity[1],
                        "indexes": start_end_entities
                    })
            # if len(start_end_entities) == 0,
            # doesn't found entity text -> but entity must be in text
             # think about handling later


            else:
                other.append(entity)

        # generate lists with tokens and entities
        tokens = []
        startindex_tokens = []
        content_tokens = nlp.tokenizer(text)
        for token in content_tokens:
            tokens.append(token.text)
            startindex_tokens.append(token.idx)

        labels = ["O"] * len(tokens)
        if len(entities_with_positions) > 0:
            for ent in entities_with_positions:
                # over all occurences through the indexes
                for occurance in ent["indexes"]:
                    try:
                        startindex = occurance[0]
                        # check if startindex of the entity is a complete token
                        if startindex in startindex_tokens:
                            index_labels = startindex_tokens.index(startindex)
                            labels[index_labels] = "B-"+ent["typ"]
                            entity_tokens = nlp.tokenizer(ent["entity_text"])
                            len_entity = len(entity_tokens)
                            for i in range(1,len_entity):
                                index_labels += 1
                                labels[index_labels] = "I-"+ent["typ"]
                    except:
                        print("incostency in " +str(idx)+": "+full_row)
                        return None

        return ADGRow(idx,full_row,number,ts,speaker,text,tokens,labels,entities_with_positions,other)

    def _save_rows_as_json(self, rows,path):
        rows_dicts = []
        for row in rows:
            rows_dicts.append(asdict(row))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows_dicts, f, indent=4, ensure_ascii=False)

    def _load_rows_as_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            rows_dicts = json.load(f)
        return [ADGRow(**row_dict) for row_dict in rows_dicts]
