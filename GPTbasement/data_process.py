import torch
import json
from itertools import islice
from tqdm import tqdm

### 处理总体数据集的功能函数
def update_tokenizer(tokenizer):
    tokenizer.pad_token = "<|pad|>"
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|user|>", "<|assistant|>"],
        'pad_token': "<|pad|>", # 151667
        'eos_token': "<|eos|>"  # 151668
    })
    return tokenizer

# text = ["<|user|>苹果<|assistant|>香蕉<|eos|>","<|user|>苹果<|assistant|>橘子<|eos|>"]
def dataloader(text,tokenizer):
    tokenizer_output = tokenizer(text,padding=True,return_tensors="pt",return_attention_mask=True)
    input_ids,attention_mask = tokenizer_output["input_ids"],tokenizer_output["attention_mask"]
    user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")             # 151665
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")   # 151666
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")               # 151667
    labels = input_ids.clone()
    #下面这部分是ChatGPT写的
    for b in range(input_ids.shape[0]):
        seq = input_ids[b]
        mask = torch.zeros_like(seq, dtype=torch.bool)
        user_positions = (seq == user_token_id).nonzero(as_tuple=True)[0]
        assistant_positions = (seq == assistant_token_id).nonzero(as_tuple=True)[0]
        for user_pos, assistant_pos in zip(user_positions, assistant_positions):
            if assistant_pos > user_pos:  # 保证配对顺序正确
                mask[user_pos: assistant_pos + 1] = True
        mask = mask | (seq == pad_token_id)
        labels[b][mask] = -100
    return input_ids, labels, attention_mask