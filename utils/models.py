from transformers import (
    BertForQuestionAnswering,
    BertTokenizer,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    XLMForQuestionAnswering,
    XLMTokenizer,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    AlbertForQuestionAnswering,
    AlbertTokenizer
)

from utils.tokenization_kobert import KoBertTokenizer

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertForQuestionAnswering, AlbertTokenizer),
    "kobert": (BertForQuestionAnswering, KoBertTokenizer),
    "distilkobert": (DistilBertForQuestionAnswering, KoBertTokenizer),
}


def get_model(model_type, model_name_or_path):
    model_class, _ = MODEL_CLASSES[model_type]

    model = model_class.from_pretrained(model_name_or_path)
    return model

def get_tokenizer(model_type, model_name_or_path, do_lower_case):
    _, tokenizer_class = MODEL_CLASSES[model_type]

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case)  # (코딩중)ISoft-Electra에도 작동하는지 확인!

    return tokenizer

