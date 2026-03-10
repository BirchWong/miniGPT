from GPTbasement.model import MultiHeadAttention
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=2, alpha=4, dropout_p=0.05):  # LoRA的参数目前只能在这里改。
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = alpha / r

        nn.init.kaiming_uniform_(self.A.weight, a=0)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.B(self.dropout(self.A(x))) * self.scale


def wrap_linear_attr_with_lora(module: nn.Module, attr_name: str):
    """把 module.<attr_name> (必须是 nn.Linear) 替换为 LoRALinear，保持原权重不变"""
    old = getattr(module, attr_name)  # 取出module类中self.是attr_name名称的变量或者对象，作为old
    assert isinstance(old, nn.Linear), f"{attr_name} 不是 nn.Linear"  # 判断属性必须是 linear; 这行也可以不写
    wrapped = LoRALinear(old)
    wrapped.to(device=old.weight.device, dtype=old.weight.dtype)
    setattr(module, attr_name, wrapped)

def apply_lora_to_mha_qv(model: nn.Module):
    """遍历整网，找到MultiHeadAttention，把 query/value 套上 LoRA"""
    for m in model.modules(): # 遍历model类中所有 nn.Module 类型的子模块
        if isinstance(m, MultiHeadAttention):
            wrap_linear_attr_with_lora(m, "query")
            wrap_linear_attr_with_lora(m, "value")


def lora_init(model):
    apply_lora_to_mha_qv(model)

    for n, p in model.named_parameters():
        p.requires_grad = (n.endswith(".A.weight") or n.endswith(".B.weight"))
