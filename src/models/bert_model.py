from transformers import AutoModelForSequenceClassification


def build_bert_classifier(pretrained_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
    )
