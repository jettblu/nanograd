pub enum UnaryOps {
    EXP2,
    NOOP,
    LOG2,
    CAST,
    SIN,
    SQRT,
    RECIP,
}

pub enum BinaryOps {
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MOD,
    CMPLT,
}

pub enum ReduceOps {
    SUM,
    MAX,
}

pub enum TernaryOps {
    MULACC,
    WHERE,
}

pub enum MovementOps {
    RESHAPE,
    PERMUTE,
    EXPAND,
    PAD,
    SHRINK,
    STRIDE,
}

pub enum LoadOps {
    EMPTY,
    RAND,
    CONST,
    FROM,
    CONTIGUOUS,
    CUSTOM,
}
