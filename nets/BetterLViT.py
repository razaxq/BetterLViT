# -*- coding: utf-8 -*-
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from .LViT import LViT


class BetterLViT(LViT):
    """LViT with a modern medical text encoder (CXR-BERT-specialized) replacing
    the legacy bert-embedding pipeline. The text encoder produces
    [B, seq_len, 768] sequence embeddings that flow into the existing
    text_module4 -> text_module1 channel-reduction stack unchanged.

    seq_len is configurable via text_seq_len (threaded into Vit.CTBN3.in_channels)
    and must match Config.text_max_len so the tokenizer output aligns with the
    Conv1d input. Embedding dim is fixed at 768 by LViT.text_module4.
    """

    def __init__(
        self,
        config,
        n_channels=3,
        n_classes=1,
        img_size=224,
        vis=False,
        text_encoder_name='microsoft/BiomedVLP-CXR-BERT-specialized',
        text_seq_len=32,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__(
            config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            text_seq_len=text_seq_len,
        )

        self.text_encoder = AutoModel.from_pretrained(
            text_encoder_name, trust_remote_code=True
        )

        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=['query', 'value'],
                lora_dropout=lora_dropout,
                bias='none',
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_cfg)
        else:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        return outputs[0]

    def forward(self, x, input_ids, attention_mask):
        text = self.encode_text(input_ids, attention_mask)
        return super().forward(x, text)
