from PIL import Image
import glob
from .config import GlobalConfig
import os
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def display_checkpoint_config(checkpoint):
    try:
        print(
            f"\n已加载模型配置如下(Model configuration loaded):\n"
            f"D: {checkpoint['D']}\n"
            f"h: {checkpoint['h']}\n"
            f"H: {checkpoint['H']}\n"
            f"num_blocks: {checkpoint['num_blocks']}\n"
            f"tokenizer_category: {checkpoint['tokenizer_category']}\n"
            f"loss: {checkpoint['loss']}\n"
            f"epochs_done: {checkpoint['epoch']}\n"
            f"epochs_total: {checkpoint['total_epochs']}\n"
        )
    except:
        pass

def attn_scores_plots(attn_scores,count, folder=GlobalConfig.attn_scores_plots_folder):
    if not hasattr(attn_scores_plots, "has_run"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        # 第一次调用该函数
        attn_scores_plots.has_run = True

    B, h, T, T = attn_scores.shape
    title = "Time_" + str(T) + " Block_" + str(count)
    print("count", count, "attn_scores", attn_scores.shape)
    attn_scores = attn_scores.squeeze(0).cpu().detach().numpy()
    for i in range(h):
        fig, ax = plt.subplots(figsize=(h, h))
        ax.imshow(attn_scores[i], cmap='viridis')
        ax.set_title(f"{title} Head_{i + 1}")
        ax.set_xticks(range(T))  # 设置x轴刻度位置
        ax.set_yticks(range(T))  # 设置y轴刻度位置
        ax.set_xticklabels(range(T))
        ax.set_yticklabels(range(T))
        plt.savefig(os.path.join(folder,f"{title} Head_{i + 1}.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)

def token_position_table(tokenizer,result,folder=GlobalConfig.attn_scores_plots_folder):
    data = []
    for i, token_id in enumerate(result):
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        data.append([i, token])
    fig, ax = plt.subplots(figsize=(8, len(result) * 0.3 + 1))
    ax.axis('off')
    table = ax.table(cellText=[[f"{row[0]}", f"'{row[1]}'"] for row in data], colLabels=["Position", "Token"],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.savefig(os.path.join(folder,"token_position_table.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 2))
    text = tokenizer.decode(result, skip_special_tokens=False)
    plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(os.path.join(folder,"text.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)

    # 把图片和表格图例放一块，方便观看。也可以不要下面这一行。
    image_merge(folder)


def image_merge(folder):
    attn_img_paths = glob.glob(os.path.join(folder, "Time_*.png"))
    table = Image.open(os.path.join(folder,"token_position_table.png"))
    for img_path in attn_img_paths:
        img = Image.open(img_path)
        new_width = img.width + table.width
        new_height = max(img.height, table.height)

        new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))  # 白色背景

        new_image.paste(img, (0, 0))
        new_image.paste(table, (img.width, 0))

        new_image.save(img_path)







