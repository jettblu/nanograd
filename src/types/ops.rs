use std::hash::Hash;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UnaryOps {
    EXP2,
    Sigmoid,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BinaryOps {
    ADD,
    SUB,
    MUL,
    MAX,
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

impl Hash for UnaryOps {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            UnaryOps::EXP2 => (11).hash(state),
            UnaryOps::Sigmoid => (12).hash(state),
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

impl Hash for BinaryOps {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            BinaryOps::ADD => (13).hash(state),
            BinaryOps::SUB => (14).hash(state),
            BinaryOps::MUL => (15).hash(state),
            BinaryOps::MAX => (16).hash(state),
        }
    }
}

// implement hash trait for operation enum
impl Hash for Ops {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Ops::UnaryOps(op) => op.hash(state),
            Ops::BinaryOps(op) => op.hash(state),
            Ops::ReduceOps(op) => op.hash(state),
            Ops::TernaryOps(op) => op.hash(state),
            Ops::LoadOps(op) => op.hash(state),
            Ops::None => (0).hash(state),
        }
    }
}
