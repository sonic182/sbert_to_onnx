import torch
import onnxruntime
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "all-MiniLM-L6-v2"
onnx_model_path = f"{model_id}.onnx"


providers = []
if torch.cuda.is_available():
    # you may also need libcudnn installed to make this provider work
    providers = ['CUDAExecutionProvider']
session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/" + model_id)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

inputs = tokenizer(sentences, padding=True, return_tensors="np")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

onnx_inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
}

outputs = session.run(None, onnx_inputs)

sentence_embeddings = torch.from_numpy(outputs[0])

# Print the output
print("Sentence Embeddings:")
print(sentence_embeddings)

normal_model = SentenceTransformer(model_id)

pt_sentence_embeddings = normal_model.encode(sentences, convert_to_tensor=True).to("cpu")

# Print the output
print("normal Sentence Embeddings:")
print(pt_sentence_embeddings)

print("cos sim between onnx and pt models is")
print(util.pairwise_cos_sim(pt_sentence_embeddings, sentence_embeddings))
# you can check that embeddings are almost the same
