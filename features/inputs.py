from dataclasses import dataclass


@dataclass
class SparseFeat:
    name: str
    category_size: int
    embedding_dim: int
    dtype: str

    def __post_init__(self):
        if self.embedding_dim == "auto":
            self.embedding_dim = 6 * int(pow(self.category_size, 0.25))


@dataclass
class DenseFeat:
    name: str
    dimension: int
    dtype: str


@dataclass
class VarLenSparseFeat:
    name: str
    sparsefeat: SparseFeat
    max_length: int


@dataclass
class VarLenDenseFeat:
    name: str
    densefeat: DenseFeat
    max_length: int