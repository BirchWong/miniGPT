from GPTbasement.data_process import dataloader
from GPTbasement.config import GlobalConfig
from GPTbasement.display import display_checkpoint_config
from tqdm import tqdm
import torch
import os
import time
from transformers import get_cosine_schedule_with_warmup

class TrainConfig:
    learning_rate = 3e-4 # 默认推荐值
    batch_size = None # 在main里面设置吧，这里也设置容易乱
    epochs = None # 在main里面设置吧，这里也设置容易乱
    warmup_ratio = 0.1  # 10% 的 warmup
    save_optimizer = False
    save_scheduler = False
    key_checkpoints = [] # 需要额外保存epoch的时机


def train(model,dataset,tokenizer,save_path=None,batch_size=TrainConfig.batch_size,epochs=TrainConfig.epochs,lr=TrainConfig.learning_rate,checkpoint_path=None,total_epochs=None):
    if save_path.endswith('.pth'): save_path = save_path[:-4]
    #optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.learning_rate)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    total_steps = (len(dataset) // batch_size) * epochs
    warmup_steps = int(TrainConfig.warmup_ratio * total_steps)

    # scheduler 默认必须使用。需要提前知道total_epochs，不写的话，默认就是epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    if total_epochs == None: total_epochs = epochs

    start_epoch = 0
    if checkpoint_path is not None:
        if checkpoint_path.endswith('.pth'): checkpoint_path = checkpoint_path[:-4]
        checkpoint = torch.load(checkpoint_path+".pth")
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        if not GlobalConfig.lora_mode:
            if os.path.exists(checkpoint_path+"_optimizer.pth"):
                optimizer.load_state_dict(torch.load(checkpoint_path+"_optimizer.pth"))
                print("Loaded optimizer.")
            if os.path.exists(checkpoint_path+"_scheduler.pth"):
                scheduler.load_state_dict(torch.load(checkpoint_path+"_scheduler.pth"))
                print("Loaded scheduler.")
                total_epochs = checkpoint['total_epochs']
            start_epoch = checkpoint['epoch']
        display_checkpoint_config(checkpoint)
        print("训练继续(continue)")

    # 训练的核心部分
    for epoch in range(1,epochs+1):
        loss_epoch = 0
        batches = range(0, len(dataset), batch_size)
        progress_bar = tqdm(batches, desc="Training")
        for i in progress_bar:
            input_ids, labels, attention_mask = dataloader(dataset[i:i+batch_size],tokenizer)
            input_ids = input_ids.to(GlobalConfig.device)
            labels = labels.to(GlobalConfig.device)
            attention_mask = attention_mask.to(GlobalConfig.device)
            logits, loss = model(input_ids,labels,attention_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_epoch+=loss.item()
            progress_bar.set_postfix(loss=loss.item(),epoch=start_epoch+epoch)

        avg_loss_epoch = loss_epoch / len(batches)
        print(f"Epoch {start_epoch+epoch} average loss: {avg_loss_epoch:.4f}")
        time.sleep(0.1) # tqdm进度条和print容易冲突，加这个等待，能好不少。

        if GlobalConfig.lora_mode:  # 如果不是LoRA微调，则只保存adaptor
            sd = model.state_dict()
            lora_sd = {k: v.cpu() for k, v in sd.items() if k.endswith(".A.weight") or k.endswith(".B.weight")}
            torch.save(lora_sd, save_path+".pth")
        else: # 如果不是LoRA微调，则正常保存完整模型
            # 每轮保存模型
            checkpoint = {
                'model_state_dict': model.state_dict(),  # 保存模型权重
                'D': model.D,
                'h': model.h,
                'H': model.D // model.h,
                'num_blocks': model.num_blocks,
                'tokenizer_category': tokenizer.name_or_path,

                # 额外记录的信息
                'loss': avg_loss_epoch,
                'epoch': start_epoch+epoch,
                'total_epochs': total_epochs,
            }
            torch.save(checkpoint, save_path+".pth")

            if epoch in TrainConfig.key_checkpoints:
                torch.save(checkpoint, save_path +"_epoch_"+ str(epoch) +".pth")

            if TrainConfig.save_optimizer:
                torch.save(optimizer.state_dict(), save_path+"_optimizer.pth")
            if TrainConfig.save_scheduler:
                torch.save(scheduler.state_dict(), save_path+"_scheduler.pth")

    return model