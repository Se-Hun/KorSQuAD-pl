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

from utils.tokenization_kobert import KoBertTokenizer


MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "kobert": (BertForQuestionAnswering, KoBertTokenizer),
    "distilbert": (DistilBertForQuestionAnswering, DistilBertTokenizer),
    "distilkobert": (DistilBertForQuestionAnswering, KoBertTokenizer),
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

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case)

    return tokenizer

