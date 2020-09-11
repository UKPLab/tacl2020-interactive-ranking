from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
import torch
import numpy as np

# You can try various pretrained models to generate embeddings.
# See https://huggingface.co/transformers/model_summary.html
# NOTE: if you change the model, also change the tokenizer line below.

# This one is a smaller model that uses less memory but may not perform so well (alternatively, try ALBERT):
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# This is the standard in many recent NLP papers:
#bert = BertModel.from_pretrained("bert-base-uncased")

# This one often performs better in more recent work but requires more memory:
# bert = RobertaModel.from_pretrained("roberta-large")

embedding_size = bert.config.hidden_size

documents = ['University starts in the fall',
             'If you fall, I will catch you.']

focus_word_idxs = [4, 2]  # indexes of the focus words in each document. In both of our examples, "fall" is the focus.

max_len = 512
input_ids = torch.empty((len(documents), max_len), dtype=torch.long)
attention_masks = torch.empty((len(documents), max_len), dtype=torch.short)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

for d, doc in enumerate(documents):
    encoding1 = tokenizer.encode_plus(
        doc,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids[d] = encoding1['input_ids'].flatten()
    attention_masks[d] = encoding1['attention_mask'].flatten()

input_ids = torch.from_numpy(np.array(input_ids))
attention_masks = torch.from_numpy(np.array(attention_masks))

sequence_emb = bert(
    input_ids=input_ids,
    attention_mask=attention_masks
)[0]  # size: batch_size x  sequence length x hidden size

# obtains a matrix of batch_size x hidden_size containing a contextualized embedding for the focus word in each document
sequence_emb = sequence_emb[range(sequence_emb.shape[0]), focus_word_idxs, :].detach().numpy()

print(sequence_emb)
print(sequence_emb.shape)
