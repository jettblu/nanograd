pub enum UnaryOps {
    EXP2,
    Tanh,
}

pub enum BinaryOps {
    ADD,
    SUB,
    MUL,
}

pub enum ReduceOps {
    SUM,
    MAX,
}

pub enum TernaryOps {
    MULACC,
    WHERE,
}

pub enum LoadOps {
    EMPTY,
    RAND,
    CONST,
    FROM,
    CONTIGUOUS,
    CUSTOM,
}

pub enum Ops {
    UnaryOps,
    BinaryOps,
    ReduceOps,
    TernaryOps,
    LoadOps,
}
