from dataclasses import dataclass, asdict
from typing import List, Tuple
import re

@dataclass
class ADGRow:
    idx: int
    unextracted: str
    num: int
    timestamp: str
    person: str
    text: str
    tokens: List[str]
    indexes: List[int]
    labels: List[str]
    entities: List[dict]
    other: List[Tuple[str, str]]
    

# check the format of the data und try to make it better - if I have time
def extract_ADG_row(row, nlp,idx):
    """
    Return the ADGRow from a row.
    If an annotated entity doesn't match to words or it contains only the type -> saved in other

    Parameters:
        -row: string of one ADG ROW
        -tokenizer: Spacy tokenizer

    Returns:
        - ADGRow
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

    #map entities to text with start and endindex
    entities_with_positions = []
    for index,entity in enumerate(entities):
        start_end_entities = []
        entity_text = entity[0]
        #entity without text -> save in other
        if len(entity_text) > 0:
            matches = re.finditer(re.escape(entity_text),text)
            for match in matches:
                start_end_entities.append((match.span()[0],match.span()[1]))

            # entity which is not found in text -> save in other
            if len(start_end_entities) == 0:
                other.append(entity)
            else:
                entities_with_positions.append({
                    "entity_text": entity_text,
                    "typ": entity[1],
                    "indexes": start_end_entities
                })
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
        # iterate over all startindexes of an entity
        for ent in entities_with_positions:
            for occurance in ent["indexes"]:
                try:
                    startindex = occurance[0]
                    # check if startindex of the entity is a complete token (maybe the entity is part of another token -> don't include)
                    if startindex in startindex_tokens:
                        index_labels = startindex_tokens.index(startindex)
                        labels[index_labels] = "B-"+ent["typ"]
                        entity_tokens = nlp.tokenizer(ent["entity_text"])
                        len_entity = len(entity_tokens)
                        # if the entity is longer than 1 token -> label the other tokens
                        for i in range(1,len_entity):
                            index_labels += 1
                            labels[index_labels] = "I-"+ent["typ"]
                except:
                    print("incostency in " +str(idx)+": "+full_row)
                    return None

    return ADGRow(idx,full_row,number,ts,speaker,text,tokens,startindex_tokens,labels,entities_with_positions,other)



