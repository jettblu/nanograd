

// TODO: REMOVE SIGMOID AND SOFTMAX OPS... THEY SHOULD BE COMPOSITIONS OF OTHER OPS
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UnaryOps {
    EXP2,
    Sigmoid,
    Softmax,
    MAX,
    LOG2,
    SUM,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BinaryOps {
    ADD,
    SUB,
    MUL,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ReduceOps {
    SUM,
    MAX,
}


#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TernaryOps {
    MULACC,
    WHERE,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LoadOps {
    EMPTY,
    RAND,
    CONST,
    FROM,
    CONTIGUOUS,
    CUSTOM,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Ops {
    UnaryOps(UnaryOps),
    BinaryOps(BinaryOps),
    ReduceOps(ReduceOps),
    TernaryOps(TernaryOps),
    LoadOps(LoadOps),
    None,
}
