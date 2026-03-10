import json
from itertools import islice
from tqdm import tqdm

###  处理Belle数据集的功能函数
def read_data_Belle(file_path,num_data):
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in islice(f, num_data):
            string = json.loads(line)
            string = adaptor_for_Belle(string)
            if string != '':
                data.append(string)
    return data

def adaptor_for_Belle(conversation_data):
    transformed_lines = []

    '''
    for msg in conversation_data['conversations']:
        if msg['from'] == 'human':
            transformed_lines.append(f"<|user|> {msg['value']}")
        elif msg['from'] == 'assistant':
            transformed_lines.append(f"<|assistant|> {msg['value']} <|eos|>")
    '''

    for i in range(0, len(conversation_data['conversations']), 2):
        msg_user = conversation_data['conversations'][i]
        msg_assistant = conversation_data['conversations'][i+1]
        assert msg_user['from'] == 'human'
        assert msg_assistant['from'] == 'assistant'

        if  msg_assistant['value'] == "":
            continue

        transformed_lines.append(f"<|user|> {msg_user['value']}")
        transformed_lines.append(f"<|assistant|> {msg_assistant['value']} <|eos|>")


    return ' '.join(transformed_lines)

def select_data_1(data,tokenizer,save_path="Belle.json"):
    data_records = []
    for datum in tqdm(data):
        tokenized_datum = tokenizer(datum, return_tensors="pt")
        num_tokens = len(tokenized_datum['input_ids'][0])
        record = {
            "content": datum,
            "tokens": num_tokens
        }
        data_records.append(record)
    sorted_records = sorted(data_records, key=lambda x: x["tokens"])
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sorted_records, f, ensure_ascii=False, indent=2)


def select_data_2(num_tokens_limit,save_path="Belle.json"):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = [item["content"] for item in data if item.get("tokens", 0) <= num_tokens_limit]
    return dataset