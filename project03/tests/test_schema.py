"""Testes do schema.py. Rodar na raiz do projeto: pytest tests/test_schema.py -v"""

from diplo_graph03 import schema
from diplo_graph03.preprocessing import dataset


def test_tamanhos_e_ordem():
    assert schema.N_NODE_CLASSES == 8 and schema.EMPTY_ID == 7
    assert schema.N_EDGE_CLASSES == 11 and schema.NO_EDGE_ID == 10
    assert schema.NODE_CLASSES[-1] == schema.EMPTY
    assert schema.EDGE_CLASSES[-1] == schema.NO_EDGE


def test_dataset_usa_so_tipos_do_esquema():
    """Os rótulos que o dataset.py gera têm que existir no schema.py."""
    nos = set(dataset.NODE_MAP.values()) | {"EFFECT", "PATIENT"}   # EFFECT vem da regra CAUSE
    assert nos <= set(schema.GRAPH_NODE_TYPES)
    arestas = (set(dataset.MODIFY_RULES.values()) | set(dataset.PATIENT_RULES.values())
               | {"FAMILY_HISTORY", "PERSONAL_HISTORY", "HAS_EFFECT"})
    assert arestas == set(schema.EDGE_TYPES)


def test_assinaturas_do_dataset_batem():
    """MODIFY_RULES e PATIENT_RULES do dataset.py concordam com EDGE_SIGNATURE."""
    for (src, dst), edge in dataset.MODIFY_RULES.items():
        assert schema.EDGE_SIGNATURE[edge] == (src, dst)
    for node_type, edge in dataset.PATIENT_RULES.items():
        assert schema.EDGE_SIGNATURE[edge] == ("PATIENT", node_type)


def test_mascara():
    m = schema.edge_mask("SYMPTOM", "ANATOMY")
    assert m[schema.EDGE_TO_ID["OCCURS_IN"]] and m[schema.NO_EDGE_ID]
    assert sum(m) == 2                                    # só OCCURS_IN e NO_EDGE
    assert sum(schema.edge_mask("PATIENT", "DISEASE")) == 4   # 3 históricos + NO_EDGE
    assert sum(schema.edge_mask("DRUG", "ANATOMY")) == 1      # só NO_EDGE
    assert sum(schema.edge_mask(schema.EMPTY, "SYMPTOM")) == 1


def test_tabela_de_mascara():
    t = schema.edge_mask_table()
    n, k = len(schema.MASK_TYPES), schema.N_EDGE_CLASSES
    assert len(t) == n and len(t[0]) == n and len(t[0][0]) == k
    i = schema.MASK_TYPE_TO_ID
    assert t[i["EXAM"]][i["RESULT"]] == schema.edge_mask("EXAM", "RESULT")


def test_graph_ida_e_volta_e_validade():
    g = schema.Graph(
        nodes=[schema.GraphNode("PATIENT", [], ""),
               schema.GraphNode("SYMPTOM", [(10, 14)], "pain"),
               schema.GraphNode("ANATOMY", [(20, 25)], "elbow")],
        edges=[schema.GraphEdge("HAS_SYMPTOM", 0, 1), schema.GraphEdge("OCCURS_IN", 1, 2)],
        doc_id="x",
    )
    assert g.problems() == [] and g.edge_validity() == (2, 2) and g.patient_index() == 0
    assert schema.Graph.from_dict(g.to_dict()) == g

    g.edges.append(schema.GraphEdge("OCCURS_IN", 2, 1))       # ANATOMY -> SYMPTOM: inválida
    assert g.edge_validity() == (2, 3) and len(g.problems()) == 1