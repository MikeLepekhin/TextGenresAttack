from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
raw_model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=11)
raw_model.load_state_dict(torch.load('xlm_roberta_en/state_dict_val_1.pt', map_location='cuda:0'))
model = HuggingFaceModelWrapper(raw_model, tokenizer)