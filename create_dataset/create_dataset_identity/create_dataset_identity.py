import json
import re

def read_data_identity(json_path, save_path=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for item in data:
        user_text = item.get("instruction", "")
        if item.get("input"):
            user_text += " " + item["input"]
        assistant_text = item.get("output", "")
        lines.append(f"<|user|>{user_text}<|assistant|>{assistant_text}<|eos|>")

    new_lines = []
    for s in lines:
        s = re.sub(r"\s*{{name}}\s*", "提拉米苏", s)
        s = re.sub(r"\s*{{author}}\s*", "王老板", s)
        new_lines.append(s)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(new_lines, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
   read_data_identity("identity_original.json", save_path="identity.json")
