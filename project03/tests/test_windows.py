"""Testes do windows.py. Rodar na raiz do projeto: pytest tests/test_windows.py -v"""

from types import SimpleNamespace

from diplo_graph03 import schema
from diplo_graph03.preprocessing import windows as W

TEXT = "He has pain in elbow.\nA test was done.\nResult was normal.\n"


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
            SimpleNamespace(label="REVEALS", arg1="T4", arg2="T5")]      # cruza linhas 2 e 3
    return SimpleNamespace(id="d1", text=TEXT, entities={e.id: e for e in ents}, relations=rels)


def _types(g):
    return [n.type for n in g.nodes]


def test_line_spans_ignora_linhas_vazias_e_crlf():
    t = "a b\r\n\r\n  \r\nc d\n"
    assert [t[s:e] for s, e in W.line_spans(t)] == ["a b", "c d"]


def test_window_spans():
    assert len(W.window_spans(TEXT, 1)) == 3
    assert len(W.window_spans(TEXT, 2)) == 2                 # linhas 1-2 e linha 3
    for k in (0, None):                                      # caso inteiro
        (s, e), = W.window_spans(TEXT, k)
        assert TEXT[s:e].startswith("He") and TEXT[s:e].endswith("normal.")


def test_janelas_de_1_linha():
    w0, w1, w2 = W.make_windows(_doc(), 1)
    assert _types(w0.graph) == ["PATIENT", "SYMPTOM", "ANATOMY"]      # paciente 1º, depois ordem do texto
    assert {e.type for e in w0.graph.edges} == {"HAS_SYMPTOM", "OCCURS_IN"}
    assert _types(w1.graph) == ["PATIENT", "EXAM"]
    assert {e.type for e in w1.graph.edges} == {"UNDERGOES"}
    assert _types(w2.graph) == ["PATIENT", "RESULT"]
    assert w2.graph.edges == []                                       # REVEALS cruza janelas: se perde
    assert all(w.graph.problems() == [] for w in (w0, w1, w2))


def test_posicoes_globais_e_locais():
    w1 = W.make_windows(_doc(), 1)[1]
    s, e = w1.graph.nodes[1].spans[0]
    assert TEXT[s:e] == "test"                                        # global: indexa o texto do caso
    ls, le = w1.local_graph().nodes[1].spans[0]
    assert w1.text[ls:le] == "test"                                   # local: indexa o texto da janela


def test_caso_inteiro_preserva_tudo():
    (w,) = W.make_windows(_doc(), 0)
    assert len(w.graph.nodes) == 5 and len(w.graph.edges) == 4
    assert w.graph.problems() == []


def test_estatisticas():
    s1 = W.window_stats([_doc()], 1)
    assert s1["n_windows"] == 3 and s1["nodes"]["p95"] == 2
    assert s1["semantic_edges"][:2] == (2, 1)                         # OCCURS_IN fica, REVEALS se perde
    assert s1["patient_edges"][:2] == (2, 2)
    s0 = W.window_stats([_doc()], 0)
    assert s0["n_windows"] == 1 and s0["semantic_edges"][:2] == (2, 2)


def test_fusao_das_janelas():
    ws = W.make_windows(_doc(), 1)
    merged = W.merge_window_graphs(ws, [w.local_graph() for w in ws], "d1")
    assert _types(merged).count("PATIENT") == 1 and len(merged.nodes) == 5
    assert {e.type for e in merged.edges} == {"HAS_SYMPTOM", "OCCURS_IN", "UNDERGOES"}
    pain = next(n for n in merged.nodes if n.type == "SYMPTOM")
    s = TEXT.index("pain")
    assert pain.spans == [(s, s + 4)]                                 # voltou para posições globais
    assert merged.problems() == []