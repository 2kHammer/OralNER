from app.model.data_provider.adg_row import ADGRow
from app.model.ner_model_provider.ner_model import NERModel


def type_check_process_ner_pipeline(model, ner_content, framework_name):
    """
    Checks the type for the processing of the NER-pipeline

    Parameters:
    model (NERModel): the NER model which should be used for ner
    ner_content (List[ADGRows]|List[str]): the content on which NER should be applied
    framework_name (FrameworkNames): the framework name of the framework
    """
    if not isinstance(ner_content, list) or len(ner_content) == 0:
        raise TypeError("Excepts a list with content")
    if not (isinstance(ner_content[0], str) or isinstance(ner_content[0], ADGRow)):
        raise TypeError("Expects a list of strings or ADGRows")
    if not isinstance(model, NERModel):
        raise TypeError("Expects an object of type NERModel")
    if model.framework_name != framework_name:
        raise ValueError("Expects an model for Flair")