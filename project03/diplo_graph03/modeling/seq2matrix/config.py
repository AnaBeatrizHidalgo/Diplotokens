"""
Configuração do seq2matrix: uma chave por hiperparâmetro da tabela hiperparametros_seq2matrix.md.

Cada experimento é um YAML que muda algumas chaves em relação ao padrão:

    cfg = Seq2MatrixConfig.from_yaml("configs/seq2matrix_base.yaml")
    cfg = Seq2MatrixConfig(window_lines=2, max_nodes=24)       # ou direto no código

Valores marcados "provisório" (max_nodes, max_len, max_span_len) devem ser trocados pelos
percentis das estatísticas das janelas: python -m diplo_graph03.preprocessing.windows
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  # PubMedBERT

CHOICES = {
    "edge_features": ("concat", "concat_diff", "concat_diff_prod"),
    "matching": ("hungarian", "text_order"),
    "edge_loss": ("focal", "ce"),
}


@dataclass
class Seq2MatrixConfig:
    # ---------------------------------------------------------------- dados
    window_lines: int | None = 1        # linhas por janela (0 ou None = caso inteiro)
    max_nodes: int = 20                 # candidatos a nó por janela
    max_len: int = 192                  # tokens por janela, com [CLS] e [SEP]
    max_span_len: int = 16              # tamanho máximo de um span, em subtokens
    mask_digits: bool = False           # troca dígitos por 0 antes do encoder
    seed: int = 0

    # ---------------------------------------------------------------- encoder
    encoder_name: str = DEFAULT_ENCODER
    freeze_encoder: bool = True
    lr_encoder: float = 2e-5            # só vale com freeze_encoder=False

    # ---------------------------------------------------------------- decoder
    decoder_layers: int = 2
    decoder_heads: int = 12
    decoder_ffn_dim: int = 3072
    decoder_dropout: float = 0.1
    patient_slot: bool = True           # slot 0 = paciente

    # ---------------------------------------------------------------- cabeças
    edge_features: str = "concat_diff_prod"
    edge_hidden: int = 512
    edge_layers: int = 2
    edge_dropout: float = 0.1

    # ---------------------------------------------------------------- pareamento e perdas
    matching: str = "hungarian"
    cost_type: float = 1.0
    cost_span: float = 1.0
    empty_weight: float = 0.1           # peso da classe EMPTY na CE de tipo
    loss_w_span: float = 1.0
    loss_w_nodes: float = 1.0           # α
    loss_w_edges: float = 1.0           # β
    edge_loss: str = "focal"
    focal_gamma: float = 2.0
    edge_neg_ratio: int | None = None   # None = usa todos os pares NO_EDGE

    # ---------------------------------------------------------------- treino
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 50
    early_stopping_patience: int = 10
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    fp16: bool = False

    # ---------------------------------------------------------------- decodificação
    schema_mask: bool = True
    node_empty_threshold: float = 0.5   # mantém o nó se P(EMPTY) < limiar
    edge_none_threshold: float = 0.5    # mantém a aresta se P(NO_EDGE) < limiar
    dedupe_spans: bool = True

    def __post_init__(self):
        self.validate()

    # ---------------------------------------------------------------- derivados

    @property
    def patient_offset(self) -> int:
        """Posição do primeiro nó real entre os slots (1 se o slot 0 é o paciente)."""
        return 1 if self.patient_slot else 0

    @property
    def num_slots(self) -> int:
        """N: slots por janela (paciente, se houver, mais os candidatos a nó)."""
        return self.max_nodes + self.patient_offset

    # ---------------------------------------------------------------- validação

    def validate(self) -> None:
        def check(cond, msg):
            if not cond:
                raise ValueError(msg)

        for key, allowed in CHOICES.items():
            check(getattr(self, key) in allowed, f"{key}={getattr(self, key)!r}: use um de {allowed}")
        check(self.window_lines is None or self.window_lines >= 0, "window_lines deve ser >= 0 ou None")
        check(self.max_nodes >= 1, "max_nodes deve ser >= 1")
        check(8 <= self.max_len <= 512, "max_len deve estar entre 8 e 512")
        check(self.max_span_len >= 1, "max_span_len deve ser >= 1")
        check(self.decoder_layers >= 1 and self.edge_layers >= 1, "o número de camadas deve ser >= 1")
        check(self.batch_size >= 1 and self.epochs >= 1, "batch_size e epochs devem ser >= 1")
        check(self.empty_weight > 0, "empty_weight deve ser > 0")
        check(self.edge_neg_ratio is None or self.edge_neg_ratio > 0, "edge_neg_ratio deve ser None ou > 0")
        for name in ("decoder_dropout", "edge_dropout", "warmup_ratio",
                     "node_empty_threshold", "edge_none_threshold"):
            check(0.0 <= getattr(self, name) <= 1.0, f"{name} deve estar entre 0 e 1")

    # ---------------------------------------------------------------- (de)serialização

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Seq2MatrixConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"chaves desconhecidas na configuração: {sorted(unknown)}")
        defaults = cls()
        clean = {}
        for k, v in d.items():
            # o YAML 1.1 lê "1e-4" como texto; converte para float quando o padrão é float
            if isinstance(getattr(defaults, k), float) and isinstance(v, (int, str)) and not isinstance(v, bool):
                v = float(v)
            clean[k] = v
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path) -> "Seq2MatrixConfig":
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    def save_yaml(self, path) -> None:
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
                              encoding="utf-8")
