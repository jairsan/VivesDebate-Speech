
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline


if __name__ == "__main__":
    model_name = "softcatala/fullstop-catalan-punctuation-prediction"

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

    print(pipeline("Els investigadors suggereixen que tot i que es tracta de la cua d'un dinosaure jove la mostra revela un plomatge adult i no pas plomissol", aggregation_strategy="first"))
