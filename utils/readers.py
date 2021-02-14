from transformers import AutoTokenizer
from transformers import BertForQuestionAnswering
from transformers import ElectraForQuestionAnswering

def lan_type(data_name):
    if data_name in ['korquad_v2']:
        return 'kor'
    elif data_name in ['squad_v2']:
        return 'en'
    else:
        raise KeyError(data_name)

def get_text_reader(reader_name, data_name):
    if reader_name == "bert":
        if lan_type(data_name) == "en":
            model_name = "bert-base-uncased"
        else:
            model_name = "bert-base-multilingual-cased"
        text_reader = BertForQuestionAnswering.from_pretrained(model_name)

    elif reader_name == "kobert":
        if lan_type(data_name) == "kor":
            model_name = "monologg/kobert"
            text_reader = BertForQuestionAnswering.from_pretrained(model_name)
        else:
            raise KeyError(data_name)

    elif reader_name == "koelectra":
        if lan_type(data_name) == "kor":
            model_name = "monologg/koelectra-base-discriminator"
            text_reader = ElectraForQuestionAnswering.from_pretrained(model_name)
        else:
            raise KeyError(data_name)

    else:
        raise KeyError(reader_name)

    return text_reader

def get_tokenizer(reader_name, data_name):
    if reader_name == "bert":
        if lan_type(data_name) == "en":
            model_name = "bert-base-uncased"
        else:
            model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif reader_name == "kobert":
        if lan_type(data_name) == "kor":
            from utils.tokenization_kobert import KoBertTokenizer
            model_name = "monologg/kobert"
            tokenizer = KoBertTokenizer.from_pretrained(model_name)
        else:
            raise KeyError(data_name)

    elif reader_name == "koelectra":
        if lan_type(data_name) == "kor":
            from transformers import ElectraTokenizer
            model_name = "monologg/koelectra-base-discriminator"
            tokenizer = ElectraTokenizer.from_pretrained(model_name)
        else:
            raise KeyError(data_name)

    else:
        raise KeyError(reader_name)

    return tokenizer
