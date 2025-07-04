from app.model.data_provider.data_registry import data_registry

# should reset data_registry for the real tests, like with data model

def check_split_sentences(dataset_id=3):
    rows = data_registry.load_training_data(dataset_id)
    sentence_tokens, sentence_labels =data_registry.split_training_data_sentences(rows)
    #check if amount tokens and labels is the same
    for index, sentence in enumerate(sentence_tokens):
        assert len(sentence) == len(sentence_labels[index])

    sum_tokens = sum(len(row.tokens) for row in rows)
    sum_tokens_sentences = sum(len(sentence) for sentence in sentence_tokens)
    # check if the amount of tokens is the same
    assert sum_tokens == sum_tokens_sentences

def test_split_sentences():
    for i in range(0,4):
        check_split_sentences(i)