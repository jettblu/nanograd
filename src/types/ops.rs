use std::hash::Hash;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UnaryOps {
    EXP2,
    Sigmoid,
    MAX,
    LOG2,
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

impl Hash for ReduceOps {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ReduceOps::SUM => (1).hash(state),
            ReduceOps::MAX => (2).hash(state),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TernaryOps {
    MULACC,
    WHERE,
}

impl Hash for TernaryOps {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            TernaryOps::MULACC => (3).hash(state),
            TernaryOps::WHERE => (4).hash(state),
        }
    }
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

impl Hash for LoadOps {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            LoadOps::EMPTY => (5).hash(state),
            LoadOps::RAND => (6).hash(state),
            LoadOps::CONST => (7).hash(state),
            LoadOps::FROM => (8).hash(state),
            LoadOps::CONTIGUOUS => (9).hash(state),
            LoadOps::CUSTOM => (10).hash(state),
        }
    }
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
