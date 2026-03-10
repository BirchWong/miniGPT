import json

with open('Belle_100.json', 'rb') as f:
    loaded_dataset = json.load(f)

print("数据条数: ",len(loaded_dataset))