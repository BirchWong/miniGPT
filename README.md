# miniGPT

This project trains a chatbot from scratch using the provided Chinese conversation datasets.

All core algorithm implementations are located in the **`GPTbasement`** folder for easy reading and learning.

On an RTX 5090 Laptop GPU, setting the batch size to 24 results in approximately 30 minutes per epoch. Running for 10 epochs can yield good performance.

## Explanation of Basic Principles

YouTube: https://youtu.be/haRzBb9Jv0A?si=BrSci2tG63KCLh5j  

bilibili: https://www.bilibili.com/video/BV1EQwWzXEDe/?share_source=copy_web&vd_source=2ce5fc590d6295912fe2a2acc28f2f04

## Dependencies

• torch (GPU version)  
• json  
• glob  
• itertools  
• re  
• tqdm  
• matplotlib  
• PIL  
• transformers  

## quick start

Set the batch_size according to your GPU memory.

```
model = train(model,dataset,tokenizer,save_path="./model_v1.pth",batch_size=4,epochs=10,lr=3e-4,checkpoint_path=None)
```

Then run the code directly.

```
python main.py
```

For more details, please read the .docx document.

---
