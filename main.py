from GPTbasement.data_process import update_tokenizer
from GPTbasement.chat import chat
from GPTbasement.config import GlobalConfig,GenerationConfig
from GPTbasement.model import GPT
from GPTbasement.LoRA import lora_init
from GPTbasement.train import train, TrainConfig
from GPTbasement.display import display_checkpoint_config
from transformers import AutoTokenizer
import time
import torch
import json
import random
import os

os.environ["HF_HUB_URL"] = "https://hf-mirror.com"

# 设置种子，使得每次运行结果相同。
#torch.manual_seed(45)

start_time = time.time()

with open('Belle_100.json', 'rb') as f:
    dataset = json.load(f)

random.shuffle(dataset)

print("数据总量",len(dataset))

os.environ["HF_HUB_URL"] = "https://hf-mirror.com"
# C:\Users\用户名\.cache\huggingface\hub  分词器的缓存windows一般在这个路径下，没多大，会自动下载
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer", trust_remote_code=True)
tokenizer = update_tokenizer(tokenizer)
Vocabulary_Size = len(tokenizer)
print("Vocabulary_Size",Vocabulary_Size)

model = GPT(Vocabulary_Size=Vocabulary_Size).to(GlobalConfig.device)

TrainConfig.save_optimizer = True
TrainConfig.save_scheduler = True

model = train(model,dataset,tokenizer,save_path="./model_v1.pth",batch_size=4,epochs=10,lr=3e-4,checkpoint_path=None)

checkpoint = torch.load('./model_v1.pth')
display_checkpoint_config(checkpoint)
model.load_state_dict(checkpoint['model_state_dict'],strict=False)

hours = (time.time() - start_time) / 3600
print(f"耗时: {hours:.2f} 小时(hour)")

model.eval()
generate_config = GenerationConfig(
    do_sample=True,          # 启用采样（非贪婪解码） ; 当为False时，temperature top_k top_p 无效
    temperature=0.8,         # 温度越低随机性越低
    top_k=50,                # 限制高概率候选词范围
    top_p=0.9,               # 覆盖90%概率的动态候选 ; 1为无效
    max_length=100,          # 生成最大长度
    repetition_penalty=1.2   # 温和抑制重复
)

#GlobalConfig.attn_scores_plots_folder = "./attn_scores_plots"
#GlobalConfig.attn_scores_plots = True

context = ["<|user|> 写一段散文，描述一场雨。 <|assistant|>"] * 4

chat(model,tokenizer,context=None,config=generate_config)


