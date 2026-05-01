# -*- coding: utf-8 -*-
"""Frozen CXR-BERT text encoder used to feed token-level features into LViT.

Replaces the legacy mxnet `bert_embedding` package with Microsoft's
`microsoft/BiomedVLP-CXR-BERT-specialized`, a BERT pretrained on chest-X-ray
radiology reports. The hidden size (768) and the downstream sequence length
(10) match the existing LViT text branch (`text_module4` Conv1d in_channels=768
and `Vit.CTBN3` in_channels=10), so no changes to the segmentation network
are required.

Design choices (confirmed with user):
- Encoder is fully frozen (eval + requires_grad=False).
- We use `last_hidden_state` (768-d per-token features), not the 128-d CLS
  projection, so the existing `text_module4..text_module1` cascade stays valid.
- Padding to a fixed `max_length` is done with the tokenizer's native [PAD]
  token rather than the legacy ' EOF XXX' word padding from `utils.read_text`.
- All texts are encoded once in the main process at Dataset construction; the
  resulting CPU tensor dict is then pickled to DataLoader workers.
"""
from __future__ import annotations

import re
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional

_MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"
_HIDDEN_SIZE = 768
_EOF_XXX_SUFFIX = re.compile(r"(\s+EOF\s+XXX)+\s*$")

_RESULT_CACHE: Dict[tuple, Dict[str, torch.Tensor]] = {}


def _strip_legacy_padding(text: str) -> str:
    cleaned = _EOF_XXX_SUFFIX.sub("", text).strip()
    return cleaned if cleaned else " "


class CXRBertEncoder:
    MODEL_NAME = _MODEL_NAME
    HIDDEN_SIZE = _HIDDEN_SIZE

    def __init__(
            self,
            max_length: int = 10,
            device: Optional[str] = None,
            batch_size: int = 64,
    ) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_dict(self, text_dict: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Encode every (key, text) pair to a CPU tensor of shape
        [max_length, 768]. Returned dict shares no GPU storage and is safe to
        pickle into DataLoader workers."""
        keys = list(text_dict.keys())
        cleaned = [_strip_legacy_padding(text_dict[k]) for k in keys]

        cache_key = (self.MODEL_NAME, self.max_length, tuple(zip(keys, cleaned)))
        cached = _RESULT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        for p in model.parameters():
            p.requires_grad = False
        model.eval().to(self.device)

        result: Dict[str, torch.Tensor] = {}
        try:
            for start in range(0, len(keys), self.batch_size):
                batch_keys = keys[start: start + self.batch_size]
                batch_texts = cleaned[start: start + self.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                ).to(self.device)
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state.detach().cpu()
                for i, key in enumerate(batch_keys):
                    result[key] = hidden[i].clone().contiguous()
        finally:
            del model, tokenizer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        _RESULT_CACHE[cache_key] = result
        return result
