"""Testes do seq2matrix (config e dados). Rodar na raiz: pytest tests/test_seq2matrix_data.py -v"""

from types import SimpleNamespace

import pytest

from diplo_graph03 import schema
from diplo_graph03.modeling.seq2matrix.config import Seq2MatrixConfig
from diplo_graph03.modeling.seq2matrix.data import (
    IGNORE,
    WindowDataset,
    build_tokenizer,
    char_to_token_span,
    collate,
)

TEXT = "He has pain in elbow.\nA test was done.\nResult was normal.\n"


# ---------------------------------------------------------------- config


def test_config_padrao_e_derivados():
    cfg = Seq2MatrixConfig()
    assert cfg.num_slots == 17 and cfg.patient_offset == 1
    assert Seq2MatrixConfig(patient_slot=False).num_slots == 16


def test_config_validacao():
    with pytest.raises(ValueError):
        Seq2MatrixConfig(matching="outro")
    with pytest.raises(ValueError):
        Seq2MatrixConfig(max_len=1000)
    with pytest.raises(ValueError):
        Seq2MatrixConfig.from_dict({"chave_que_nao_existe": 1})


def test_config_yaml_ida_e_volta(tmp_path):
    cfg = Seq2MatrixConfig(window_lines=2, max_nodes=24, edge_neg_ratio=5)
    cfg.save_yaml(tmp_path / "c.yaml")
    assert Seq2MatrixConfig.from_yaml(tmp_path / "c.yaml") == cfg
    assert Seq2MatrixConfig.from_dict({"lr": "1e-4"}).lr == 1e-4        # YAML lê 1e-4 como texto


def test_yaml_base_do_repositorio_bate_com_o_padrao():
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "configs" / "seq2matrix_base.yaml"
    if base.exists():
        assert Seq2MatrixConfig.from_yaml(base) == Seq2MatrixConfig()


# ---------------------------------------------------------------- span em tokens


def test_char_to_token_span():
    offsets = [(0, 0), (0, 2), (3, 6), (6, 8), (0, 0)]                  # [CLS] he ... [SEP]
    assert char_to_token_span(offsets, 3, 8) == (2, 3)
    assert char_to_token_span(offsets, 0, 2) == (1, 1)
    assert char_to_token_span(offsets, 10, 12) is None


# ---------------------------------------------------------------- dataset (usa o tokenizer real)


def _ent(eid, label, word):
    s = TEXT.index(word)
    return SimpleNamespace(id=eid, label=label, spans=[(s, s + len(word))], text=word)


def _doc():
    ents = [_ent("T1", "PATIENT", "He"), _ent("T2", "SYMPTOM", "pain"),
            _ent("T3", "ANATOMY", "elbow"), _ent("T4", "EXAM", "test"),
            _ent("T5", "RESULT", "normal")]
    rels = [SimpleNamespace(label="HAS_SYMPTOM", arg1="T1", arg2="T2"),
            SimpleNamespace(label="OCCURS_IN", arg1="T2", arg2="T3"),
            SimpleNamespace(label="UNDERGOES", arg1="T1", arg2="T4"),
            SimpleNamespace(label="REVEALS", arg1="T4", arg2="T5")]
    return SimpleNamespace(id="d1", text=TEXT, entities={e.id: e for e in ents}, relations=rels)


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer(Seq2MatrixConfig().encoder_name)
    except OSError:
        pytest.skip("tokenizer do PubMedBERT indisponível (sem cache nem rede)")


def test_alvos_da_primeira_janela(tok):
    cfg = Seq2MatrixConfig(max_nodes=4, max_len=64)
    ds = WindowDataset([_doc()], cfg, tok)
    assert len(ds) == 3
    ex = ds[0]
    assert ex["node_types"].tolist() == [schema.NODE_TO_ID["SYMPTOM"], schema.NODE_TO_ID["ANATOMY"]]
    s, e = ex["node_starts"][0].item(), ex["node_ends"][0].item()
    assert tok.decode(ex["input_ids"][s:e + 1]) == "pain"               # span em tokens, fim incluído
    et = ex["edge_target"]
    assert et.shape == (5, 5)                                           # 4 candidatos + paciente
    assert et[0, 1] == schema.EDGE_TO_ID["HAS_SYMPTOM"]                 # paciente (slot 0) -> pain (slot 1)
    assert et[1, 2] == schema.EDGE_TO_ID["OCCURS_IN"]                   # pain -> elbow
    assert (et != schema.NO_EDGE_ID).sum() == 2
    assert ds.report["nodes_kept"] == ds.report["nodes_total"] == 4


def test_sem_slot_do_paciente(tok):
    cfg = Seq2MatrixConfig(max_nodes=4, max_len=64, patient_slot=False)
    ds = WindowDataset([_doc()], cfg, tok)
    et = ds[0]["edge_target"]
    assert et.shape == (4, 4)
    assert et[0, 1] == schema.EDGE_TO_ID["OCCURS_IN"]                   # pain = slot 0, elbow = slot 1
    assert (et != schema.NO_EDGE_ID).sum() == 1                         # HAS_SYMPTOM saiu do alvo
    assert ds.report["edges_patient_skipped"] == 2                      # HAS_SYMPTOM e UNDERGOES


def test_lote(tok):
    cfg = Seq2MatrixConfig(max_nodes=4, max_len=64)
    ds = WindowDataset([_doc()], cfg, tok)
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["input_ids"].shape[0] == 3
    assert batch["num_nodes"].tolist() == [2, 1, 1]
    assert batch["node_types"].shape == (3, 2) and batch["node_types"][1, 1] == IGNORE
    assert batch["edge_target"].shape == (3, 5, 5)
    assert not batch["token_mask"][0, 0]                                # [CLS] não é um token de span


def test_contagem_do_que_nao_cabe(tok):
    cfg = Seq2MatrixConfig(max_nodes=1, max_len=64)                     # só 1 candidato por janela
    ds = WindowDataset([_doc()], cfg, tok)
    assert ds.report["nodes_dropped_over_max_nodes"] == 1               # elbow não cabe na janela 1
    assert ds.report["edges_lost_dropped_node"] >= 1                    # OCCURS_IN perdeu um nó
    small = WindowDataset([_doc()], Seq2MatrixConfig(max_nodes=4, max_len=8), tok)
    assert small.report["windows_truncated"] > 0
