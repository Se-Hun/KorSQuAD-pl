from transformers import (
    BertForQuestionAnswering,
    BertTokenizer,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    ElectraForQuestionAnswering,
    ElectraTokenizer
)

# from utils.tokenization_kobert import KoBertTokenizer
from kobert_transformers import get_tokenizer as get_kobert_tokenizer

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "kobert": (BertForQuestionAnswering, get_kobert_tokenizer),
    "distilbert": (DistilBertForQuestionAnswering, DistilBertTokenizer),
    "distilkobert": (DistilBertForQuestionAnswering, get_kobert_tokenizer),
    "koelectra": (ElectraForQuestionAnswering, ElectraTokenizer),
    "albert": (AlbertForQuestionAnswering, AlbertTokenizer),
    "xlnet": (XLNetForQuestionAnswering, XLNetTokenizer),
    "electra": (ElectraForQuestionAnswering, ElectraTokenizer)
}


def get_model(model_type, model_name_or_path):
    model_class, _ = MODEL_CLASSES[model_type]

    model = model_class.from_pretrained(model_name_or_path)
    return model

def get_tokenizer(model_type, model_name_or_path, do_lower_case):
    _, tokenizer_class = MODEL_CLASSES[model_type]

    if model_type == "kobert" or model_type == "distilkobert":
        tokenizer = tokenizer_class()
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                    do_lower_case=do_lower_case)

    return tokenizer

