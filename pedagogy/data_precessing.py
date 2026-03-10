from transformers import AutoTokenizer
import torch

def data_preprocess_superior(text,tokenizer):
    tokenizer.pad_token = "<|pad|>"
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|user|>", "<|assistant|>"],
        'pad_token': "<|pad|>",
        'eos_token': "<|eos|>"
    })
    tokenizer_output = tokenizer(text,padding=True,return_tensors="pt",return_attention_mask=True)
    input_ids,attention_mask = tokenizer_output["input_ids"],tokenizer_output["attention_mask"]
    #print("input_ids",input_ids)
    #print("attention_mask",attention_mask)

    user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")             # 151665
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")   # 151666
    eos_token_id = tokenizer.convert_tokens_to_ids("<|eos|>")               # 151668
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")               # 151667
    print(f"""<|user|>       : {user_token_id}\n<|assistant|>  : {assistant_token_id}\n<|eos|>        : {eos_token_id}\n<|pad|>        : {pad_token_id}""")
    input_ids_original = input_ids.clone()
    for input_ids_single in input_ids:
        flag_is_user = False
        for i in range(len(input_ids_single)):
            if input_ids_single[i] == user_token_id:
                input_ids_single[i] = -100
                flag_is_user = True
            if input_ids_single[i] == assistant_token_id:
                input_ids_single[i] = -100
                flag_is_user = False
            if flag_is_user == True:
                input_ids_single[i] = -100
            if input_ids_single[i] == pad_token_id:
                input_ids_single[i] = -100
    labels = input_ids
    input_ids = input_ids_original

    return input_ids, labels, attention_mask


def data_preprocess(text,tokenizer):
    tokenizer.pad_token = "<|pad|>"
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|user|>", "<|assistant|>"],
        'pad_token': "<|pad|>",
        'eos_token': "<|eos|>"
    })
    tokenizer_output = tokenizer(text,padding=True,return_tensors="pt",return_attention_mask=True)
    input_ids,attention_mask = tokenizer_output["input_ids"],tokenizer_output["attention_mask"]
    #print("input_ids",input_ids)
    #print("attention_mask",attention_mask)

    user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")             # 151665
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")   # 151666
    eos_token_id = tokenizer.convert_tokens_to_ids("<|eos|>")               # 151668
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")               # 151667
    print(f"""<|user|>       : {user_token_id}\n<|assistant|>  : {assistant_token_id}\n<|eos|>        : {eos_token_id}\n<|pad|>        : {pad_token_id}""")
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



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer", trust_remote_code=True, local_files_only=True)
    Vocabulary_Size = len(tokenizer)
    print("Vocabulary_Size",Vocabulary_Size)

    #  token: "苹果"  "香蕉"  "橘"  "子"  （对于Qwen的分词器）
    text = [
        "<|user|>苹果<|assistant|>香蕉<|eos|>",
        "<|user|>苹果<|assistant|>橘子<|eos|>"
    ]
    text_tokenized = [
        ["<|user|>","苹果","<|assistant|>","香蕉","<|eos|>","<|pad|>"],
        ["<|user|>","苹果","<|assistant|>","橘"  ,"子"     ,"<|eos|>"],
    ]
    input_ids = [
        [151665, 104167, 151666, 112622, 151668, 151667],
        [151665, 104167, 151666, 114088,  44729, 151668]
    ]
    attention_mask = [
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1]
    ]
    # 最后处理得到的标签       "<|user|>","苹果","<|assistant|>","<|pad|>" 完成掩码
    # label 最后进行 cross_entropy 计算出loss; -100的部分会自动忽略,不参与得出loss
    label = [
        [  -100,   -100,   -100, 112622, 151668,   -100],
        [  -100,   -100,   -100, 114088,  44729, 151668]
    ]


    input_ids, labels, attention_mask = data_preprocess_superior(text,tokenizer)

    print("input_ids",input_ids)
    print("labels", labels)
    print("attention_mask",attention_mask)
    print("\n")

    input_ids, labels, attention_mask = data_preprocess(text,tokenizer)

    print("input_ids",input_ids)
    print("labels", labels)
    print("attention_mask",attention_mask)