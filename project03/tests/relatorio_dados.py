from diplo_graph03 import config
from diplo_graph03.preprocessing import dataset
from diplo_graph03.modeling.seq2matrix.config import Seq2MatrixConfig
from diplo_graph03.modeling.seq2matrix.data import WindowDataset, build_tokenizer

cfg = Seq2MatrixConfig()
docs = dataset.load_docs(config.MACCROBAT_GRAFO_DIR)
ds = WindowDataset(docs, cfg, build_tokenizer(cfg.encoder_name))
ds.print_report()

