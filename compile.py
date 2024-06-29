import torch
from sentence_transformers import SentenceTransformer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

inputs = model.tokenizer(sentences, padding=True, return_tensors="pt")

# Wrap the model's forward method for ONNX export
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model({'input_ids': input_ids, 'attention_mask': attention_mask})['sentence_embedding']

wrapped_model = WrappedModel(model)

wrapped_model.eval()

torch.onnx.export(
    wrapped_model,
    (inputs['input_ids'], inputs['attention_mask']),
    f"{model_id}.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['sentence_embedding'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'sentence_embedding': {0: 'batch_size'}
    },
    opset_version=17
)
