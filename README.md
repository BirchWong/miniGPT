# miniGPT

This project trains a chatbot from scratch using the provided Chinese conversation datasets.

All core algorithm implementations are located in the **`GPTbasement`** folder for easy reading and learning.

On an RTX 5090 Laptop GPU, setting the batch size to 24 results in approximately 30 minutes per epoch. Running for 10 epochs can yield good performance.

## Explanation of Basic Principles (About 30 mins)

YouTube (English): https://youtu.be/haRzBb9Jv0A?si=BrSci2tG63KCLh5j  

bilibili (Chinese): https://www.bilibili.com/video/BV1EQwWzXEDe/?share_source=copy_web&vd_source=2ce5fc590d6295912fe2a2acc28f2f04

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

## Quick Start

Set the batch_size according to your GPU memory.

```
model = train(model,dataset,tokenizer,save_path="./model_v1.pth",batch_size=4,epochs=10,lr=3e-4,checkpoint_path=None)
```

Then run the code directly.

```
python main.py
```

For more details, please read the .docx document.

## Results

![miniGPT training results](https://cdn.jsdelivr.net/gh/BirchWong/images@master/miniGPT-result.png)

---

### English Translation:

**Prompt:** Please write a prose paragraph describing a rain.

**miniGPT:**  
Today's rain is the awakening season of nature. The sky is overcast and damp, yet the sunlight is bright. The gentle rain taps against the window, forming patches of green grass blades. It seems to be cheering for this beautiful day. In this season, everything becomes quiet and serene, making one feel relaxed and happy. Today is a place full of vitality and hope, so let us fully enjoy this tranquility and beauty.

**miniGPT:**  
The sound of rain taps against the window, with raindrops dripping onto the windowsill. The world outside is gently brushed by the rain, as if enveloped in this beauty and the memories of hope it brings. From dawn to dusk, one can see beautiful scenery, tranquil mountain peaks, and the azure sky—all of which evoke a sense of peace and comfort.

**miniGPT:**  
The rain taps against the window, creating sounds as it strikes the glass. The raindrops gently patter against the ground, forming dense streaks. Raindrops from the eaves fall onto the damp paths, resembling shimmering beacons of light. The rain falls softly, making the entire world moist and fresh. This rain is so beautiful, evoking a sense of tranquility and comfort.

**miniGPT:**  
Rain is a grand feast of nature, its endless scenes unfolding like a stunning painting. The fine raindrops gently tap the earth, accompanied by the soft whisper of the wind, refreshing the mind and spirit. The grass and trees thrive with vitality; the rippling water exudes a serene tranquility. In this quiet moment, people immerse themselves in this peaceful and mysterious world.

---
