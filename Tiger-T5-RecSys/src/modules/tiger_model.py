import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, List, Callable

# (我们不再从这里导入 RQVAE，因为 V2 流程中 T5 模型是独立的)
# from src.modules.rqvae_model import RQVAE 

PAD_TOKEN = 0
BOS_TOKEN = 1 
EOS_TOKEN = 2

class TigerT5(nn.Module):
    def __init__(
        self,
        unified_vocab_size: int,
        d_model: int,
        n_head: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        decoder_start_token_id: int = BOS_TOKEN, # (BOS_TOKEN = 1)
        pad_token_id: int = PAD_TOKEN, # (PAD_TOKEN = 0)
        eos_token_id: int = EOS_TOKEN, # (EOS_TOKEN = 2)
    ):
        super().__init__()
        
        self.config = T5Config(
            vocab_size=unified_vocab_size,
            d_model=d_model,
            num_heads=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            d_ff=dim_feedforward,
            dropout_rate=dropout,
            is_encoder_decoder=True,
            use_cache=True, # 对于 .generate()
            decoder_start_token_id=decoder_start_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )
        
        # (我们使用 T5Config 来初始化，而不是 T5-base)
        self.model = T5ForConditionalGeneration(self.config)
        
    def forward(self,
              encoder_input_ids: torch.Tensor,    # (Batch, QuerySeqLen) [已偏移]
              encoder_attention_mask: torch.Tensor, # (Batch, QuerySeqLen)
              decoder_input_ids: torch.Tensor,    # (Batch, LLMSeqLen) [已偏移, 带 BOS]
              
              # --- 【V4 修正】使 'decoder_labels' 变为可选 ---
              decoder_labels: Optional[torch.Tensor] = None # (Batch, LLMSeqLen) [已偏移, 带 EOS]
             ):
        
        # (V4 修正: 不再调用 self.model.train()，由调用者决定模式)
        # self.model.train()

        labels = None
        # --- 【V4 修正】只有在提供了 'decoder_labels' 时才处理它们 ---
        if decoder_labels is not None:
            labels = decoder_labels.clone()
            labels[labels == self.config.pad_token_id] = -100

        outputs = self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels # 'labels' 要么是 None (predict时), 要么是处理过的 (train时)
        )

        return outputs

    @torch.no_grad()
    def generate(self,
                 
                 # --- ↓↓↓ 【V4 修正】参数名必须与 'forward' 保持一致 ↓↓↓ ---
                 encoder_input_ids: torch.Tensor,    # (Batch, QuerySeqLen)
                 encoder_attention_mask: torch.Tensor, # (Batch, QuerySeqLen)
                 # --- ↑↑↑ 【V4 修正】参数名必须与 'forward' 保持一致 ---
                 
                 logits_processor: Optional[List[Callable]] = None, 
                 max_length: int = 5,    
                 num_beams: int = 1
                ):
        
        self.model.eval()
        
        outputs = self.model.generate(
            
            # --- ↓↓↓ 【V4 修正】传递的变量名也已更改 ↓↓↓ ---
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            # --- ↑↑↑ 【V4 修正】传递的变量名也已更改 ---
            
            max_length=max_length,
            num_beams=num_beams,
            logits_processor=logits_processor,
            # (确保 .generate() 使用正确的 BOS_TOKEN)
            decoder_start_token_id=self.config.decoder_start_token_id 
        )
        
        return outputs