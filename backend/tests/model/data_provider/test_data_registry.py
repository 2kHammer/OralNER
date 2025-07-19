import csv
import os
import shutil
from io import StringIO

from app.model.data_provider.adg_row import ADGRow
from app.model.data_provider.data_registry import data_registry, DataRegistry, TrainingData
from app.utils.config import STORE_TEMP_PATH, TRAININGSDATA_PATH

"""
    Notes:
        most test are only running in the specific test environment, especially all integration tests
        they are using the datasets and models which are created in the model and data registry
        this allows the interaction of different functions to be tested
"""

# -------------------------------------
# init & helpers
# -------------------------------------
DataRegistry._reset_instance()
# test metadata path
test_path = STORE_TEMP_PATH +  "/metadata.json"
def delete_test_dir():
    if os.path.exists(test_path):
        print("Deleting test directory")
        for filename in os.listdir(STORE_TEMP_PATH):
            file_path = os.path.join(STORE_TEMP_PATH, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

test_file = None
path = TRAININGSDATA_PATH+ "/adg1220.csv"
with open(path, newline='',encoding="latin1") as csvfile:
    test_file = csvfile.read().splitlines()

def create_add_data_registry():
    dr = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    # tests simple add and save trainingsdata
    dr.add_training_data("test1","adg1220.csv",test_file)
    return dr

# -------------------------------------
# unit tests
# -------------------------------------

def test_read_convert_adg_file():
    csv_str = "1	00:00:24.15	INT_MZ	Mich interessiert zunächst mal: Sind Ihre Eltern hier in Hochlarmark groß geworden oder von anderswo hergekommen und ...	 Eltern [ROLE]; Hochlarmark [LOC];\n\n1	00:00:24.15	INT_MZ	Mich interessiert zunächst mal: Sind Ihre Eltern hier in Hochlarmark groß geworden oder von anderswo hergekommen und ...	 Eltern [ROLE]; Hochlarmark [LOC];"
    fake_file = StringIO(csv_str)
    dr = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    rows = dr._read_convert_adg_file(fake_file)
    assert isinstance(rows[0], ADGRow)
    #check if empty lines are ignored
    assert len(rows) == 2
    csv_error = "Fehlerhafter CSV-String"
    fake_file2 = StringIO(csv_error)
    rows = dr._read_convert_adg_file(fake_file2)
    assert rows is None

def test_prepare_data_without_labels():
    test = "Ich bin ein Student. Nebenbei mache ich gern Sport. Zum Essen gibt es hoffentlich Pizza"
    dr = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    sents = dr.prepare_data_without_labels(test)
    assert len(sents) == 3
    nothing = dr.prepare_data_without_labels("")
    assert len(nothing) == 0

def test_check_convert_adg_file():
    dr = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    csv_str = "1	00:00:24.15	INT_MZ	Mich interessiert zunächst mal: Sind Ihre Eltern hier in Hochlarmark groß geworden oder von anderswo hergekommen und ...	 Eltern [ROLE]; Hochlarmark [LOC];\n\n 1	00:00:24.15	INT_MZ	Mich interessiert zunächst mal: Sind Ihre Eltern hier in Hochlarmark groß geworden oder von anderswo hergekommen und ...	 Eltern [ROLE]; Hochlarmark [LOC];"
    file = StringIO(csv_str)
    #check correct lines
    assert dr.check_convert_adg_file(file, steps=1)
    #check false if no line is checked
    assert not dr.check_convert_adg_file(file, steps=5)
    #checke false file
    fake_csv = "1 00:00:24.15 Fehler; 1 00:00:24.15 Fehler;"
    file2 = StringIO(fake_csv)
    assert not dr.check_convert_adg_file(file2, steps=1)

def test_get_next_id():
    dr = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    dr._datasets=[TrainingData(0,"test", "dummypath","01.01.1999"), TrainingData(1,"test", "dummypath","01.01.1999")]
    assert dr._get_next_id() == 2
    dr._datasets.append(TrainingData(3, "test", "dummypath", "01.01.1999"))
    assert dr._get_next_id() == 2

def test_simple_tokenizer():
    dr = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    test = "Ich bin ein Student. Nebenbei mache ich gern Sport. Zum Essen gibt es hoffentlich Pizza"
    tokens, indexes =dr._simple_tokenizer(test)
    for index,token in enumerate(tokens):
        len_token = len(token)
        assert token == test[indexes[index]:indexes[index]+len_token]

def test_simple_splite_sentences():
    dr = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    test = "Ich bin ein Student. Nebenbei mache ich gern Sport. Zum Essen gibt es hoffentlich Pizza"
    tokens, indexes_tokens = dr._simple_tokenizer(test)
    sentences, indexes = dr._simple_split_sentences(test)
    for i,index in enumerate(indexes):
        start_token = tokens[index[0]]
        end_token = tokens[index[1]-1]
        assert start_token in sentences[i]
        assert end_token in sentences[i]

# add maybe a unit test for add_training_data
    # but tested this already in the integration test

# -------------------------------------
# integration tests
# -------------------------------------
# data registry is for the management of the datasets and metadata
# most functionality includes file io -> is tested with an integration test

def test_data_registry():
    delete_test_dir()
    dr = create_add_data_registry()
    #check singleton
    dr2 = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    assert dr == dr2
    #check if file with the same filename will be overriden
    dr.add_training_data("test1","adg1220.csv",test_file)
    assert len(dr._datasets) == 1
    #check if file with other name is appended
    dr.add_training_data("test1","adg1220_.csv",test_file)
    assert len(dr._datasets) == 2
    # check false csv
    csv_str = "01:24:52.17	IP_EF	Die Kameradschaft und so weiter;\n	 Kameradschaft;\n1	01:24:55.05"
    fake_file = StringIO(csv_str)
    res =dr.add_training_data("test_fake","fake.csv",fake_file)
    res_check = dr.check_convert_adg_file(fake_file,1)
    assert res is None
    assert res_check is False
    assert len(dr._datasets) == 2
    # test load
    rows = dr.load_training_data(0)
    assert isinstance(rows, list) and isinstance(rows[0], ADGRow)
    # test load false id
    assert dr.load_training_data(100) is None

    # delete data_registry
    dr._reset_instance()
    del dr
    del dr2

    #load data new
    dr3 = DataRegistry(test_path, path_store=STORE_TEMP_PATH)
    assert len(dr3.list_training_data()) == 2

#test with adg1220
def test_split_sentences():
    data_registry = create_add_data_registry()
    rows = data_registry.load_training_data(0)
    sentence_data =data_registry.split_training_data_sentences(rows)
    #check if amount tokens and labels is the same
    for index, sentence in enumerate(sentence_data):
        assert len(sentence.tokens) == len(sentence.labels)

    # check if tokens are the same without '"' and ' '
    for row in rows:
        sen_rows = [sen for sen in sentence_data if sen.row_index == row.idx]
        tokens_sen = []
        for sen in sen_rows:
            tokens_sen += [token for token in sen.tokens if (token != ' ' and token != '"')]

        tokens_row = [token for token in row.tokens if (token != ' ' and token != '"')]
        assert tokens_sen == tokens_row

    for sen in sentence_data:
        # check if the amount of indexes is the same as the amount of tokens
        assert len(sen.token_indexes) == len(sen.tokens)
        for ind, token in enumerate(sen.tokens):
            len_token = len(token)
            startindex_token = sen.token_indexes[ind]
            # check if index corresponds to the token
            assert sen.text[startindex_token:startindex_token+len_token] == token