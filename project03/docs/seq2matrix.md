# Pipeline v1: texto clínico → grafo (one-shot)

```mermaid
flowchart TD
    classDef dado fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef modelo fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;
    classDef param fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#5d4037;
    classDef treino fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef infer fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;

    subgraph COMUM ["Treino e inferência"]
        T["Janela de texto<br/>string"]:::dado
        ENC["Encoder PubMedBERT<br/>H ∈ ℝ^(L × 768)"]:::modelo

        subgraph DECODER ["Decoder (camadas empilhadas)"]
            Q["Queries aprendidas (entrada do decoder)<br/>Q ∈ ℝ^(N × 768)<br/>slot 0 = paciente<br/>slots 1..N−1 = candidatos a nó"]:::param
            CAM["Em cada camada:<br/>1. self-attention entre as queries<br/>2. cross-attention nos tokens H<br/>saída: U ∈ ℝ^(N × 768)"]:::modelo
            Q --> CAM
        end

        NH["Cabeça de nós (slots 1..N−1)<br/>tipo + VAZIO: ℝ^((N−1) × 8)<br/>início, fim: ℝ^((N−1) × L)"]:::modelo
        EH["Cabeça de arestas (todos os pares)<br/>entrada: u_i ; u_j ; u_i−u_j ; u_i⊙u_j<br/>logits ∈ ℝ^(N × N × 11)"]:::modelo

        T --> ENC
        ENC -->|"H"| CAM
        CAM --> NH
        CAM --> EH
        ENC -.->|"H"| NH
    end

    subgraph TREINO ["Só no treino"]
        G["Gabarito<br/>paciente + M nós (tipo, início, fim)<br/>arestas (i, tipo, j)"]:::dado
        HUN["Húngaro nos slots 1..N−1 vs M nós<br/>(paciente já pareado ao slot 0)<br/>π: slots → nós ou VAZIO"]:::treino
        LOSS["Perda (escalar)<br/>nós: CE tipo (VAZIO com peso baixo) + CE posição<br/>arestas: focal CE em N × N reordenado por π"]:::treino
        G --> HUN --> LOSS
        G --> LOSS
    end

    subgraph INFER ["Só na inferência"]
        DECODE["argmax do tipo; descarta VAZIO<br/>posição: argmax de início e fim<br/>→ n nós: (tipo, texto do span)"]:::infer
        EDGE["Máscara de esquema + argmax<br/>descarta SEM_ARESTA<br/>→ arestas (i, tipo, j)"]:::infer
        GRAF["Grafo da janela → fusão das janelas<br/>→ grafo do caso"]:::infer
        DECODE --> GRAF
        EDGE --> GRAF
    end

    NH --> HUN
    NH --> LOSS
    EH --> LOSS
    LOSS -.->|"backprop"| CAM

    NH --> DECODE
    EH --> EDGE
    DECODE --> EDGE
```

| Símbolo | Significado | Valor sugerido |
|---|---|---|
| L | tokens da janela | ~30 a 80 (uma frase) |
| N | slots por janela (1 paciente + N−1 candidatos) | ~16 |
| M | nós reais no gabarito da janela (sem o paciente) | varia, M ≤ N−1 |
| 8 | 7 tipos de nó (DISEASE, SYMPTOM, DRUG, EXAM, ANATOMY, EFFECT, RESULT) + VAZIO | fixo |
| 11 | 10 tipos de aresta + SEM_ARESTA | fixo |

O paciente é o slot 0: existe sempre, não passa pela cabeça de nós nem pelo Húngaro, e suas arestas
(`DIAGNOSED_WITH`, `PERSONAL_HISTORY`, `FAMILY_HISTORY`, `HAS_SYMPTOM`, `TAKES`, `UNDERGOES`)
saem do mesmo tensor de arestas que as demais (`OCCURS_IN`, `AFFECTS`, `REVEALS`, `HAS_EFFECT`).
A probabilidade de um slot ser um nó é 1 − P(VAZIO); a perda de tipo dá menos peso à classe VAZIO
(no DETR, 0,1), e a perda de posição só vale nos slots pareados com um nó real.