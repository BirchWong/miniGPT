from GPTbasement.data_process import update_tokenizer
from tools import read_data_Belle,select_data_1,select_data_2
from transformers import AutoTokenizer
import time
import os
import json

start_time = time.time()

# 3606402
file_path = r'F:\Data\train_3.5M_CN\train_3.5M_CN.json'  # 这个可以自己去网上下载 Belle 中文数据库

dataset = read_data_Belle(file_path, num_data=5000000)  #  这个数据集里面有空行
print("Belle数据集总量",len(dataset))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer", trust_remote_code=True)
tokenizer = update_tokenizer(tokenizer)

temp_dataset_path = "Belle.json"
if not os.path.exists(temp_dataset_path):
    select_data_1(dataset,tokenizer,save_path=temp_dataset_path)

num_tokens_limit = 200
dataset = select_data_2(num_tokens_limit=num_tokens_limit,save_path=temp_dataset_path)
print("dataset",len(dataset))

Belle_path = 'Belle_'+str(num_tokens_limit)+'.json'
# 保存
with open(Belle_path, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Dataset saved to {Belle_path}")