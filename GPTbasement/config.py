import torch

class GlobalConfig:
    Maximum_Sequence_Length = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_mode = False
    attn_scores_plots = False
    attn_scores_plots_folder = "./attn_scores_plots"

class GenerationConfig:
    def __init__(
        self,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 1.0,
        max_length: int = 200,
        repetition_penalty: float = 1.0,
        **kwargs  # 支持其他扩展参数
    ):
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length
        self.repetition_penalty = repetition_penalty