from dataclasses import dataclass

@dataclass
class SparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int #  embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
    use_hash: bool
    dtype: str
    embedding_name: str
    group_name: str


@dataclass
class VarLenSparseFeat:
    sparsefeat: SparseFeat
    maxlen: int
    combiner: str
    length_name: str


@dataclass
class DenseFeat:
    name: str
    dimension: int
    dtype: str

