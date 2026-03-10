import re
from .config import GlobalConfig
from .display import token_position_table
import torch

def format_conversation(text):
    #text = text.replace("<|user|>", "\nUser:")
    #text = text.replace("<|assistant|>", "\nAssistant:")
    text = re.sub(r'<\|user\|>.*?<\|assistant\|>', '', text, flags=re.DOTALL)
    #text = "gpt:" + text
    text = re.sub(r'<\|eos\|>.*', '', text, flags=re.DOTALL)
    text = text.replace("<|pad|>", '')
    return text

def chat(model,tokenizer,context=None,multi_turns=False,config=None):
    decoded_text = ""
    if context != None:
        decoded_texts = inference(model, tokenizer, context, config)
        #decoded_string = [f"第{i+1}条回复: {output}" for i, output in enumerate(decoded_texts)]
        decoded_string = [f"miniGPT:\n {output}" for i, output in enumerate(decoded_texts)]
        decoded_string = '\n'.join(decoded_string)
        print(decoded_string)
        return decoded_texts
    else:
        while(True):
            context_input = input("User:")
            if context_input == "break":
                break
            context_input = "<|user|> " + context_input + " <|assistant|>"
            if multi_turns:
                context_input = decoded_text + context_input
            else:
                context_input = context_input
            outputs = inference(model,tokenizer,context_input,config)
            decoded_text = outputs[0]
            print("miniGPT:"+outputs[0])
        

def inference(model,tokenizer,context=None,config=None,extra_return_required=False):
    tokenizer_output = tokenizer(context, padding=True, return_tensors="pt", return_attention_mask=True,padding_side="left") # 推理时候使用左padding
    input_ids = tokenizer_output["input_ids"].to(GlobalConfig.device)
    attention_mask = tokenizer_output["attention_mask"].to(GlobalConfig.device)

    with torch.no_grad():
        token_index_outputs,logprobs,entropies,mask = model.generate(input_ids,attention_mask,tokenizer=tokenizer,config=config,extra_return_required=True)

    decoded_tokens = [
        tokenizer.decode(sample.tolist(), skip_special_tokens=False)
        for sample in token_index_outputs  # 遍历 batch 维度
    ]

    if GlobalConfig.attn_scores_plots == True:  # 把生成的token按表格形式打印
        token_position_table(tokenizer, token_index_outputs[0]) # 此时的输出 batch_size 应该为 1

    decoded_texts = [
        format_conversation(decoded_token)
        for decoded_token in decoded_tokens
    ]

    entropy = torch.sum(entropies) / entropies.shape[1] # 除以T,总token长度。因为总token长度不一样
    
    if extra_return_required:return decoded_texts,token_index_outputs,logprobs,entropy,mask
    else:return decoded_texts