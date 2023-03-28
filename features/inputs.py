from dataclasses import dataclass
from typing import Optional

@dataclass
class SparseFeat:
    name: str
    category_size: int
    embedding_dim: int
    dtype: str
    feature_idx: [Optional] = None

    def __post_init__(self):
        if self.embedding_dim == "auto":
            self.embedding_dim = 6 * int(pow(self.category_size, 0.25))


@dataclass
class DenseFeat:
    name: str
    dimension: int
    dtype: str
    feature_idx: [Optional] = None


@dataclass
class VarLenSparseFeat:
    name: str
    sparsefeat: SparseFeat
    max_length: int
    feature_idx: [Optional] = None


@dataclass
class VarLenDenseFeat:
    name: str
    densefeat: DenseFeat
    max_length: int
    feature_idx: [Optional] = None
