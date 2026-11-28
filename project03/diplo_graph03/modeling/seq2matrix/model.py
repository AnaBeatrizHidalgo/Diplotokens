"""
Modelo seq2matrix: texto -> (tipos de nó, spans, matriz de arestas) em uma passada só.

    tokens --[encoder PubMedBERT]--> H [B, L, d]
    queries aprendidas [N, d] --[decoder: self-attention entre queries + cross-attention em H]--> U [B, N, d]
    U --> cabeça de TIPO    (só slots candidatos)  [B, C, 8]    7 tipos + EMPTY
    U, H --> cabeças de POSIÇÃO (início, fim)      [B, C, L]    qual token começa / termina o nó
    U --> cabeça de ARESTAS (todos os pares)       [B, N, N, 11] 10 tipos + NO_EDGE

B = lote, L = tokens, d = tamanho do encoder (768), N = num_slots, C = max_nodes (candidatos).
Com patient_slot, o slot 0 é o paciente: ele passa pelo decoder como os outros e tem linhas e
colunas na matriz de arestas, mas não tem cabeça de tipo nem de posição. Os candidatos são os
slots 1..N-1, e a posição c das saídas de tipo/início/fim corresponde ao slot c + patient_offset.

Aqui só há o forward (logits). O pareamento húngaro, as perdas e a decodificação ficam em outros
arquivos. As saídas são logits crus (sem softmax), na ordem das queries.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from diplo_graph03 import schema
from diplo_graph03.modeling.seq2matrix.config import Seq2MatrixConfig

NEG = -1e4          # logit dos tokens que não podem ser span (cabe em fp16)


class SpanHead(nn.Module):
    """Aponta um token: produto escalar entre a query (projetada) e cada token (projetado)."""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, u: torch.Tensor, h: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """u [B, C, d], h [B, L, d], token_mask [B, L] bool -> logits [B, C, L]."""
        logits = torch.einsum("bcd,bld->bcl", self.q(u), self.k(h)) * self.scale
        return logits.masked_fill(~token_mask[:, None, :], NEG)


class EdgeHead(nn.Module):
    """Classifica a aresta de cada par ordenado (i -> j) a partir dos vetores das duas queries."""

    MULT = {"concat": 2, "concat_diff": 3, "concat_diff_prod": 4}

    def __init__(self, d: int, cfg: Seq2MatrixConfig):
        super().__init__()
        self.mode = cfg.edge_features
        dim = self.MULT[self.mode] * d
        layers = []
        for _ in range(cfg.edge_layers - 1):
            layers += [nn.Linear(dim, cfg.edge_hidden), nn.ReLU(), nn.Dropout(cfg.edge_dropout)]
            dim = cfg.edge_hidden
        layers.append(nn.Linear(dim, schema.N_EDGE_CLASSES))
        self.mlp = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u [B, N, d] -> logits [B, N, N, 11]; [b, i, j] é a aresta do slot i para o slot j."""
        B, N, d = u.shape
        ui = u.unsqueeze(2).expand(B, N, N, d)
        uj = u.unsqueeze(1).expand(B, N, N, d)
        feats = [ui, uj]
        if self.mode in ("concat_diff", "concat_diff_prod"):
            feats.append(ui - uj)                  # a diferença dá a noção de direção
        if self.mode == "concat_diff_prod":
            feats.append(ui * uj)
        return self.mlp(torch.cat(feats, dim=-1))


class Seq2Matrix(nn.Module):
    def __init__(self, cfg: Seq2MatrixConfig, encoder: nn.Module | None = None):
        """`encoder` permite injetar um encoder já carregado (usado nos testes com um BERT minúsculo)."""
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(cfg.encoder_name)
        d = self.encoder.config.hidden_size
        if d % cfg.decoder_heads:
            raise ValueError(f"decoder_heads={cfg.decoder_heads} não divide o tamanho do encoder ({d})")

        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # queries aprendidas: a ENTRADA do decoder (no lugar das palavras já geradas de um decoder de texto)
        self.query_embed = nn.Embedding(cfg.num_slots, d)

        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.decoder_heads, dim_feedforward=cfg.decoder_ffn_dim,
            dropout=cfg.decoder_dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.decoder_layers, norm=nn.LayerNorm(d))

        self.type_head = nn.Linear(d, schema.N_NODE_CLASSES)
        self.start_head = SpanHead(d)
        self.end_head = SpanHead(d)
        self.edge_head = EdgeHead(d, cfg)

    # ------------------------------------------------------------------ encoder

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Estados dos tokens H [B, L, d]. Com o encoder congelado, nem guarda gradientes."""
        if self.cfg.freeze_encoder:
            with torch.no_grad():
                return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()                    # encoder congelado também fica sem dropout
        return self

    # ------------------------------------------------------------------ forward

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_mask: torch.Tensor) -> dict:
        """
        input_ids, attention_mask, token_mask: [B, L] (vindos do collate de data.py).
        Devolve logits: type_logits [B, C, 8], start_logits e end_logits [B, C, L],
        edge_logits [B, N, N, 11] e slot_states [B, N, d].
        """
        h = self.encode(input_ids, attention_mask)
        B = h.size(0)
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        u = self.decoder(tgt, h, memory_key_padding_mask=~attention_mask.bool())   # sem máscara causal

        cand = u[:, self.cfg.patient_offset:, :]    # slots candidatos a nó (sem o paciente)
        return {
            "type_logits": self.type_head(cand),
            "start_logits": self.start_head(cand, h, token_mask),
            "end_logits": self.end_head(cand, h, token_mask),
            "edge_logits": self.edge_head(u),
            "slot_states": u,
        }

    # ------------------------------------------------------------------ utilidades para o treino

    def parameter_groups(self) -> list[dict]:
        """Grupos para o otimizador: decoder e cabeças com `lr`; encoder (se descongelado) com `lr_encoder`."""
        enc_ids = {id(p) for p in self.encoder.parameters()}
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in enc_ids]
        groups = [{"params": other, "lr": self.cfg.lr}]
        if not self.cfg.freeze_encoder:
            groups.append({"params": [p for p in self.encoder.parameters() if p.requires_grad],
                           "lr": self.cfg.lr_encoder})
        return groups

    def count_parameters(self) -> tuple[int, int]:
        """(treináveis, total)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total