import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, List, Callable

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
        decoder_start_token_id: int = BOS_TOKEN,
        pad_token_id: int = PAD_TOKEN,
        eos_token_id: int = EOS_TOKEN,
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
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id
        )
        self.model = T5ForConditionalGeneration(config=self.config)

    def forward(self, 
              encoder_input_ids: torch.Tensor,
              encoder_attention_mask: torch.Tensor,
              decoder_input_ids: torch.Tensor,
              decoder_labels: torch.Tensor
             ):

        self.model.train()

        labels = decoder_labels.clone()
        labels[labels == self.config.pad_token_id] = -100

        outputs = self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        return outputs

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 
                 logits_processor: Optional[List[Callable]] = None, 
                 max_length: int = 5,    
                 num_beams: int = 1
                ):
        
        self.model.eval()
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            
            max_length=max_length,
            num_beams=num_beams,
            logits_processor=logits_processor,
            decoder_start_token_id=self.config.decoder_start_token_id
        )
        return outputs