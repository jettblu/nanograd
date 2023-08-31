enum UnaryOps {
    EXP2,
    NOOP,
    LOG2,
    CAST,
    SIN,
    SQRT,
    RECIP,
}

enum BinaryOps {
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MOD,
    CMPLT,
}

enum ReduceOps {
    SUM,
    MAX,
}

enum TernaryOps {
    MULACC,
    WHERE,
}

enum MovementOps {
    RESHAPE,
    PERMUTE,
    EXPAND,
    PAD,
    SHRINK,
    STRIDE,
}

enum LoadOps {
    EMPTY,
    RAND,
    CONST,
    FROM,
    CONTIGUOUS,
    CUSTOM,
}
