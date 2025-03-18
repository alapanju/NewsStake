import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("/home/alapan/llm/test_trainer/checkpoint-210")
model = AutoModelForSequenceClassification.from_pretrained("/home/alapan/llm/test_trainer/checkpoint-210")

sentence = 'Punjab national bank is one of the oldest bank in India. The employees there are working over time for handleing the present scenario'
hypothesis = 'The entity Punjab national bank belongs to the stakeholder group of opposition'

inputs = tokenizer(sentence, hypothesis, truncation='longest_first', return_tensors="pt")


with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(predicted_class_id)