from torch.nn import functional as F
from .positional_embedding import * # 如果 PE_type = "Learnable" 则没用 (但里面有import torch等)
from .config import GlobalConfig
from .display import attn_scores_plots

'''  缩写变量的指代意义  '''
# B   batch size
# T   time step length = sequence length
# D   token embedding dimension
# H   head size = token embedding dimension (per head)  通常不小于64
# h   head amount

class ModelConfig:
    dropout = 0.2
    PE_type = ["Learnable", "Sinusoidal", "RoPE"][2]
    h = 8           # n_heads
    H = 64          # d_head
    D = H * h       # d_model
    num_block = 8   # n_layers

class MultiHeadAttention(nn.Module):
    _count = 0
    # input  (B, T, D)
    # output (B, T, D)
    def __init__(self, D, h):
        super().__init__()
        MultiHeadAttention._count += 1
        self.count = MultiHeadAttention._count
        self.h = h
        self.D = D
        self.H = D // h
        self.key = nn.Linear(D, D, bias=False)
        self.query = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(GlobalConfig.Maximum_Sequence_Length, GlobalConfig.Maximum_Sequence_Length)))  # 获得self.tril，作为缓存，不进行学习更新
        self.dropout = nn.Dropout(ModelConfig.dropout)
        self.proj = nn.Linear(self.H * self.h, self.D)
        if ModelConfig.PE_type == "RoPE":
            self.rotary_emb = build_RoPE(H=self.H)
        self.attn_scores_plots = attn_scores_plots
        self.gate = nn.Linear(D, self.h, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,mask):
        B, T, D = x.shape
        k = self.key(x).view(B,T,self.h,self.H).transpose(1,2)    # (B,h,T,H)
        q = self.query(x).view(B,T,self.h,self.H).transpose(1,2)  # (B,h,T,H)
        v = self.value(x).view(B,T,self.h,self.H).transpose(1,2)  # (B,h,T,H)
        # 使用 RoPE
        if ModelConfig.PE_type == "RoPE":
            #position_ids = torch.arange(T, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(B,1)  # (B,T)
            position_ids = (mask.cumsum(dim=-1) - 1).to(x.dtype) # (B,T)
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # (B,H,T,hs)
        # compute attention scores   T = T_Q = T_K
        attn_scores = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B,h,T_Q,H) @ (B,h,H,T_K) -> (B,h,T_Q,T_K)  内积自动发生在H维度上
        # 原本的tril提供因果掩码，mask提供padding掩码
        tril = mask.float().view(B, 1, 1, T) * self.tril[:T, :T].unsqueeze(0).unsqueeze(0)  # [B,1,1,T_K] * [1,1,T_Q,T_K] -> [B,1,T_Q,T_K]

        mask_value = -torch.finfo(attn_scores.dtype).max  # 根据数据类型自动选择最小值
        attn_scores = attn_scores.masked_fill(tril == 0, mask_value)
        # attn_scores = attn_scores.masked_fill(tril == 0, -1e9)  # (B,h,T_Q,T_K)
        # 如果使用 float('-inf') 则需要特殊处理，防止出现nan问题

        attn_scores = F.softmax(attn_scores, dim=-1)  # (B,h,T_Q,T_K)  在k的T维度上进行softmax
        if GlobalConfig.attn_scores_plots == True:
            if B == 1:
                self.attn_scores_plots(attn_scores, self.count)
            else: print("打印attn_scores图时，只能一次输入一个prompt")

        attn_scores = self.dropout(attn_scores)
        out = attn_scores @ v # (B,h,T_Q,T_K) @ (B,h,T_K,H) -> (B,h,T_Q,H)

        # gated attenion 如果不想用了，就把下面三行和self.gate一起删掉。
        gate = self.sigmoid(self.gate(x)) # (B, T, D) @ (D,h) -> (B, T, h)
        gate = gate.permute(0, 2, 1).unsqueeze(-1) # (B, T, h) -> # (B, h, T, 1)
        out = gate * out # (B,h,T_Q,H) * (B, h, T, 1) -> (B,h,T_Q,H)  Broadcasting

        out = out.transpose(1, 2).reshape(B, T, D)  # (B,h,T,H) -> (B,T,h,H) -> (B,T,h*H)
        out = self.proj(out)
        return self.dropout(out)   # (B,T,D)

class FeedFoward(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),  # 现有的模型，我都用这个激活函数了
            #nn.ReLU(),
            nn.Linear(4 * D, D),
            nn.Dropout(ModelConfig.dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.MHA = MultiHeadAttention(D, h)
        self.FFwd = FeedFoward(D)
        self.LN = nn.LayerNorm(D)

    def forward(self, x,mask):
        x = x + self.MHA(self.LN(x),mask)
        x = x + self.FFwd(self.LN(x))
        return x

class GPT(nn.Module):
    def __init__(self, D=ModelConfig.D, h=ModelConfig.h, num_blocks=ModelConfig.num_block, Vocabulary_Size=None):
        super().__init__()
        self.token_embedding_table = nn.Embedding(Vocabulary_Size, D)
        self.position_embedding_table = nn.Embedding(GlobalConfig.Maximum_Sequence_Length, D)
        self.blocks = nn.ModuleList([Block(D=D,h=h) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, Vocabulary_Size),
        )

        # 用于后续保存模型配置
        self.D = D
        self.h = h
        self.num_blocks = num_blocks

        # important
        self.apply(self._init_weights)

        # 使用 正余弦位置编码
        if ModelConfig.PE_type == "Sinusoidal":
            with torch.no_grad():
                self.position_embedding_table.weight.copy_(build_sinusoidal_pos_emb(GlobalConfig.Maximum_Sequence_Length, D))
            self.position_embedding_table.weight.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None,mask=None):
        B,T = x.shape
        tok_emb = self.token_embedding_table(x.long()) # (B,T,D)
        if ModelConfig.PE_type in ["Sinusoidal", "Learnable"]:
            pos_emb = self.position_embedding_table(torch.arange(T, device=GlobalConfig.device)) # (T,D)
            x = tok_emb + pos_emb # (B,T,D)
        else: x = tok_emb # (B,T,D)

        # 模型核心区域
        for block in self.blocks:
            x = block(x, mask)  # (B,T,D)

        logits = self.tail(x) # (B,T,Vocabulary_Size)

        if y is None:
            loss = None
        else:
            logits = logits[:, :-1, :]  #  去掉最后一个时间步T
            y = y[:, 1:]  # 去掉第一个时间步 ; 为了和logits信息对其 ; logits应该预测下一个token
            B, T, D = logits.shape
            logits = logits.reshape(B*T, D)
            y = y.reshape(B*T)
            loss = F.cross_entropy(logits, y) # 一个是概率分布，一个是索引, 默认reduction='mean'

        return logits, loss

    def generate(self, x,mask,tokenizer,config,extra_return_required=False):
        repetition_penalty = config.repetition_penalty
        temperature = config.temperature
        top_k = config.top_k
        top_p = config.top_p
        do_sample = config.do_sample
        max_length = config.max_length

        eos_token_id = tokenizer.convert_tokens_to_ids("<|eos|>")  # 151667
        # 保存所有token的logprobs和entropies，暂时没用。方便扩展，如GRPO等算法需要。
        all_logprobs = []
        all_entropies = []

        B, original_T  = x.shape
        eos_flags = B * [False] # 为批处理prompt功能提供，用来标记一个batch的问题，是否都达到eos。
        # x is (B, T)
        for _ in range(max_length):
            B, T = x.shape
            logits, loss = self(x=x,mask=mask)
            #mask_add = torch.ones(B, 1, dtype=mask.dtype,device=mask.device)
            mask_add = 1 - torch.tensor(eos_flags, dtype=mask.dtype,device=mask.device).unsqueeze(-1) # tensor形状同上,方便用mask得知何时出现<|eos|>，给GRPO等算法使用。单纯输出功能和上一行相同。
            mask = torch.cat([mask,mask_add], dim=1)

            # last time step
            logits = logits[:, -1, :] # (B, vocab_size)

            logprobs = F.log_softmax(logits, dim=-1)  # (B, vocab_size)

            # 第1步：重复惩罚
            logits = self.use_repetition_penalty(x,logits,repetition_penalty,B,original_T)
            # 第2步：温度调节
            logits = self.use_temperature(logits,temperature)

            probs = F.softmax(logits, dim=-1) # (B, vocab_size)

            # 第3步：Top-K过滤
            probs = self.use_top_k(probs,top_k)
            # 第4步：Top-p过滤
            probs = self.use_top_p(probs,top_p)

            if do_sample:
                index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            else:
                index_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # (B)
            all_entropies.append(entropy)

            selected_logprobs = logprobs.gather(dim=-1, index=index_next).squeeze(-1)  # (B,)
            all_logprobs.append(selected_logprobs)
            
            x = torch.cat((x, index_next), dim=1) # (B, T+1)

            for i in range(B):
                if index_next[i,0] == eos_token_id:
                    eos_flags[i] = True
            if all(eos_flags):
                break

        all_logprobs = torch.stack(all_logprobs, dim=1)
        all_entropies = torch.stack(all_entropies, dim=1)
        if extra_return_required:return x,all_logprobs,all_entropies,mask
        else:return x


    ''' 这下面的staticmethod都是deepseek写的，辅助generate的功能。也可以不要，注销generate的对应行 '''
    @staticmethod
    def use_repetition_penalty(x,logits,repetition_penalty,B,T):
        # 只在推理时候加入; 防止模型结巴，一直重复
        if repetition_penalty != 1.0:
            generated_tokens = x[:, T:]
            if generated_tokens is not None:
                for b in range(B):
                    seen_tokens = generated_tokens[b].unique()
                    for token in seen_tokens:
                        if logits[b, token] > 0:
                            logits[b, token] /= repetition_penalty
                        else:
                            logits[b, token] *= repetition_penalty
        return logits
    @staticmethod
    def use_top_k(probs,top_k):  # 仅保留概率最高的 k 个 token
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)  # 重新归一化
        return probs
    @staticmethod
    def use_top_p(probs,top_p):  # 从累积概率超过 p 的最小 token 集合中采样 ; 1是不启用
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs <= top_p
            sorted_mask[..., 0] = True
            sorted_probs[~sorted_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(sorted_probs).scatter_(
                dim=-1, index=sorted_indices, src=sorted_probs
            )
        return probs
    @staticmethod
    def use_temperature(logits,temperature):
        return logits / temperature