"""冒烟测试：加载本地 bert-base-chinese，抽取一句汉语的 13 层隐状态。"""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "models/bert-base-chinese"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()

sent = "看似简单的二选一，最后决定的还是自己。"
inputs = tok(sent, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

hs = out.hidden_states
print("句子:", sent)
print("分词:", tok.convert_ids_to_tokens(inputs["input_ids"][0]))
print("层数:", len(hs), "| 每层形状:", tuple(hs[0].shape))
print("SMOKE TEST OK")
