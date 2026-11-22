"""
Embeddings contextuais com PubMedBERT, direto do texto cru.

Usa o tokenizer do próprio BERT (WordPiece), que informa de quais caracteres do texto vem
cada subtoken (offsets). Assim os vetores se alinham com as posições do .ann sem passar
pelo tokenizer.py do Projeto 2.

Interface esperada pelo resto do pipeline (qualquer embedder que a siga serve):
    embedder.name                      -> str, usado no nome do cache
    embedder.embed_text(text)          -> (offsets [n, 2] int, embeddings [n, dim] float32)
"""

import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  # PubMedBERT


class BertEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 max_len: int = 512, stride: int = 64, batch_size: int = 8,
                 mask_digits: bool = False):
        """
        max_len/stride: textos longos são cortados em janelas de max_len tokens com
                        sobreposição de `stride` tokens.
        mask_digits:    troca cada dígito por "0" ("1.25 mg" -> "0.00 mg"). Mantém o tamanho
                        do texto, então as posições (offsets) continuam valendo. Serve para
                        testar o efeito de esconder os valores das medidas.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len, self.stride, self.batch_size = max_len, stride, batch_size
        self.mask_digits = mask_digits

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if not self.tok.is_fast:
            raise ValueError("é preciso um tokenizer 'fast' para obter os offsets")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    @property
    def name(self) -> str:
        return self.model_name.split("/")[-1] + ("-digits0" if self.mask_digits else "")

    @torch.no_grad()
    def embed_text(self, text: str):
        """Retorna (offsets, embeddings) de todos os subtokens do texto."""
        if self.mask_digits:
            text = re.sub(r"\d", "0", text)

        enc = self.tok(text, return_offsets_mapping=True, return_overflowing_tokens=True,
                       truncation=True, max_length=self.max_len, stride=self.stride,
                       padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)

        vecs = {}
        for i in range(0, offsets.shape[0], self.batch_size):
            batch = {k: v[i:i + self.batch_size].to(self.device) for k, v in enc.items()}
            hidden = self.model(**batch).last_hidden_state.cpu().numpy()
            for w in range(hidden.shape[0]):
                for t, (a, b) in enumerate(offsets[i + w].tolist()):
                    if a != b:                       # ignora [CLS], [SEP] e padding
                        vecs.setdefault((a, b), hidden[w, t])   # vale a 1ª janela em que aparece

        if not vecs:
            return np.zeros((0, 2), np.int32), np.zeros((0, self.dim), np.float32)
        keys = sorted(vecs)
        return (np.array(keys, dtype=np.int32),
                np.stack([vecs[k] for k in keys]).astype(np.float32))
