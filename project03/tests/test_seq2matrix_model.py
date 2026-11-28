"""Testes do model.py com um BERT minúsculo (sem download). pytest tests/test_seq2matrix_model.py -v"""

import pytest
import torch
from transformers import BertConfig, BertModel

from diplo_graph03 import schema
from diplo_graph03.modeling.seq2matrix.config import Seq2MatrixConfig
from diplo_graph03.modeling.seq2matrix.model import NEG, Seq2Matrix

D, B, L = 48, 2, 10


def _encoder():
    return BertModel(BertConfig(hidden_size=D, num_hidden_layers=1, num_attention_heads=4,
                                intermediate_size=64, vocab_size=100))


def _cfg(**kw):
    base = dict(max_nodes=5, decoder_heads=4, decoder_ffn_dim=64, edge_hidden=32, decoder_layers=1)
    base.update(kw)
    return Seq2MatrixConfig(**base)


def _batch():
    ids = torch.randint(1, 100, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    attn[1, -2:] = 0                                              # a 2ª sequência é mais curta
    tok = torch.zeros(B, L, dtype=torch.bool)
    tok[:, 1:L - 3] = True                                        # [CLS], [SEP] e padding não são span
    return ids, attn, tok


def test_formatos_com_slot_do_paciente():
    m = Seq2Matrix(_cfg(), encoder=_encoder())
    out = m(*_batch())
    assert out["type_logits"].shape == (B, 5, schema.N_NODE_CLASSES)
    assert out["start_logits"].shape == out["end_logits"].shape == (B, 5, L)
    assert out["edge_logits"].shape == (B, 6, 6, schema.N_EDGE_CLASSES)       # 5 candidatos + paciente
    assert out["slot_states"].shape == (B, 6, D)


def test_formatos_sem_slot_do_paciente():
    out = Seq2Matrix(_cfg(patient_slot=False), encoder=_encoder())(*_batch())
    assert out["type_logits"].shape[1] == 5
    assert out["edge_logits"].shape == (B, 5, 5, schema.N_EDGE_CLASSES)


def test_tokens_fora_do_span_ficam_mascarados():
    ids, attn, tok = _batch()
    out = Seq2Matrix(_cfg(), encoder=_encoder())(ids, attn, tok)
    for name in ("start_logits", "end_logits"):
        assert (out[name][:, :, 0] <= NEG + 1).all()                          # [CLS]
        assert (out[name][:, :, L - 1] <= NEG + 1).all()                      # fim/padding
        assert (out[name][:, :, 1:L - 3] > NEG + 1).all()                     # tokens válidos


def test_encoder_congelado_nao_recebe_gradiente():
    m = Seq2Matrix(_cfg(freeze_encoder=True), encoder=_encoder())
    m.train()
    assert not m.encoder.training                                             # fica em eval mesmo em train()
    out = m(*_batch())
    (out["type_logits"].sum() + out["edge_logits"].sum()).backward()
    assert all(p.grad is None for p in m.encoder.parameters())
    assert m.query_embed.weight.grad is not None
    assert len(m.parameter_groups()) == 1


def test_encoder_descongelado_recebe_gradiente():
    m = Seq2Matrix(_cfg(freeze_encoder=False), encoder=_encoder())
    out = m(*_batch())
    out["start_logits"].clamp(min=-100).sum().backward()
    assert any(p.grad is not None for p in m.encoder.parameters())
    groups = m.parameter_groups()
    assert len(groups) == 2 and groups[1]["lr"] == m.cfg.lr_encoder
    trainable, total = m.count_parameters()
    assert trainable == total


@pytest.mark.parametrize("mode", ["concat", "concat_diff", "concat_diff_prod"])
def test_variantes_das_features_de_aresta(mode):
    m = Seq2Matrix(_cfg(edge_features=mode), encoder=_encoder())
    first = m.edge_head.mlp[0]
    assert first.in_features == {"concat": 2, "concat_diff": 3, "concat_diff_prod": 4}[mode] * D
    assert m(*_batch())["edge_logits"].shape == (B, 6, 6, schema.N_EDGE_CLASSES)


def test_cabecas_que_nao_dividem_o_encoder():
    with pytest.raises(ValueError):
        Seq2Matrix(_cfg(decoder_heads=5), encoder=_encoder())                 # 48 não é divisível por 5