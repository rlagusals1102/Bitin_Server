from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

class SentimentModel:
    def __init__(self):
        self.model = TFBertForSequenceClassification.from_pretrained("app/model", num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained("app/model")
        self.max_seq_len = 128

    def predict_sentiment(self, sentence):
        input_ids = self.tokenizer.encode(sentence, max_length=self.max_seq_len, pad_to_max_length=True, truncation=True)
        attention_mask = [1] * (self.max_seq_len - input_ids.count(self.tokenizer.pad_token_id)) + [0] * input_ids.count(self.tokenizer.pad_token_id)
        token_type_ids = [0] * self.max_seq_len

        input_ids = np.array([input_ids], dtype=int)
        attention_mask = np.array([attention_mask], dtype=int)
        token_type_ids = np.array([token_type_ids], dtype=int)

        prediction = self.model.predict([input_ids, attention_mask, token_type_ids])
        predicted_label = np.argmax(prediction.logits[0], axis=-1)
        return predicted_label

sentiment_model = SentimentModel()
