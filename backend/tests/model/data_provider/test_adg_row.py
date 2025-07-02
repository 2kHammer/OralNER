import csv
import re
import spacy

from app.model.data_provider.adg_row import extract_ADG_row
from app.utils.config import TRAININGSDATA_PATH, DEFAULT_TOKENIZER_PATH

test_file = []
path = TRAININGSDATA_PATH+ "/adg1220.csv"

with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')  # Trennzeichen anpassen, falls nötig
    for row in reader:
        if len(row)>0:
            test_file.append(row)
            
test_file2 = []
path2 = TRAININGSDATA_PATH + "/adg0063.csv"
with open(path2, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if len(row)>0:
            test_file2.append(row)
        
nlp = spacy.load(DEFAULT_TOKENIZER_PATH)

def test_example_normal_row():
    row = test_file[6]
    adg_row = extract_ADG_row(row,nlp, 0)
    # check amount entities
    assert len(adg_row.entities) == 3

    #check entities in labels
    assert 1 == adg_row.labels.count("B-ORG")
    assert 2 == adg_row.labels.count("B-ROLE")

    # check if other entities are in list
    filtered_list = [label for label in adg_row.labels if label not in ["B-ORG", "B-ROLE"]]
    assert len(list(set(filtered_list))) == 1

    # check if the lists all have the same length
    assert len(adg_row.tokens) == len(adg_row.labels)
    assert len(adg_row.labels) == len(adg_row.indexes)

# check if the entity is added if it's part of another
def test_example_entity_in_another_token():
    row = test_file[6]
    wort_after_insert = "runtergeleitet"
    index = row[0].find(wort_after_insert) + len(wort_after_insert)
    row[0] = row[0][:index] + "Bürgermeister" + row[0][index:]
    adg_row = extract_ADG_row(row,nlp, 1)
    # no third entry is added
    assert adg_row.labels.count("B-ROLE") == 2

def test_inner_labes(row=test_file[3]):
    adg_row = extract_ADG_row(row,nlp, 2)
    for entity in adg_row.entities:
        doc = nlp(entity["entity_text"])
        tokens = [token.text for token in doc]
        # check entities that are bigger than 1 word
        if len(tokens)>1:
            start_index = entity["indexes"][0]
            index_in_indexes = adg_row.indexes.index(start_index[0])
            type = entity["typ"]
            assert adg_row.labels[index_in_indexes] == ("B-"+type)
            for i in range(1, len(tokens)):
                assert adg_row.labels[index_in_indexes+i] == ("I-"+type)

# check if not found entity texts are in other
def test_not_found_entity(row=test_file2[87]):
    print("test")
    extracted_row = extract_ADG_row(row,nlp, 3)
    first_column = row[0].split("\t")
    first_column.extend(row[1:])
    entities = []
    pattern = "(.*)\[(PER|ROLE|ORG|LOC|WORK_OT_ART|NORP|EVENT|DATE)\]"
    for rest in first_column:
        if rest != '':
            match = re.match(pattern, rest)
            if match:
                text_description_optional = match.group(1).split("[")
                entities.append((text_description_optional[0].strip(), match.group(2)))

    texts_in_other = [other[0] for other in extracted_row.other]
    for entity in entities:
        matches = re.finditer(re.escape(entity[0]), extracted_row.text)
        if len(list(matches)) == 0:
            assert entity[0] in texts_in_other

