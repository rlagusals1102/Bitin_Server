from transformers import TFBertForSequenceClassification, BertTokenizer

model_name = "klue/bert-base"


model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.save_pretrained("app/model")

tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("app/model")
