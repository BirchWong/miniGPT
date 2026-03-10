import torch
import torch.nn as nn

### 可选择如下两种位置编码形式
### 一个是deepseek写的，另一个是huggingface transformers库的底层代码里面抄的 ; 两者作用的位置不一样，所有输入输出都不一样

### 正余弦位置编码 组件
def build_sinusoidal_pos_emb(Maximum_Sequence_Length, D):
    position = torch.arange(Maximum_Sequence_Length).unsqueeze(1)  # [Maximum_Sequence_Length, 1]
    exponent = - torch.log(torch.tensor(10000.0)) / D  # - ( ln(10000) / D )
    div_term = torch.exp(torch.arange(0, D, 2) * exponent)  # [D//2,]      exp(-2*i*ln10000/d) == 1 / (10000^(2*i/d))
    pe = torch.zeros(Maximum_Sequence_Length, D)  # [Maximum_Sequence_Length, D]
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数 sin(pos * (1/10000^{2i/d}))
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数 cos(pos * (1/10000^{2i/d}))
    return pe  # [Maximum_Sequence_Length, D]
### 正余弦位置编码 组件


### RoPE 位置编码 组件 
rope_type = 'linear'
class build_RoPE(nn.Module):
    def __init__(self,H):
        super().__init__()
        self.dim = H  # head_size
        self.base = 10000
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))
        self.factor = 1  # 推理的时候可以改这个拓展位置编码，暂时没用
        self.attention_factor = 1
        if rope_type == 'linear':
            self.inv_freq /= self.factor

    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None,:,None].expand(position_ids.shape[0],-1,1).to(x.device)
        position_ids_expanded = position_ids[:,None,:].to(x.device)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2) # [bs, Maximum_Sequence_Length, head_size//2]

        emb = torch.cat((freqs, freqs), dim=-1) # [bs, Maximum_Sequence_Length, head_size]
        cos = emb.cos().to(dtype=x.dtype).to(x.device) * self.attention_factor
        sin = emb.sin().to(dtype=x.dtype).to(x.device) * self.attention_factor

        return cos,sin

def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed,k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
### RoPE 位置编码 组件