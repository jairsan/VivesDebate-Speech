
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


if __name__ == "__main__":
    model_name = "softcatala/fullstop-catalan-punctuation-prediction"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    pipeline("Els investigadors suggereixen que tot i que es tracta de la cua d'un dinosaure jove la mostra revela un plomatge adult i no pas plomissol")
