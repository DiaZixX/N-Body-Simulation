//! @file lib.rs
//! @brief Library entry point for n-body simulation

pub mod body;
pub mod geom;
pub mod kdtree;

pub use body::Body;
pub use geom::Vector;
pub use kdtree::{KdCell, KdTree, Node};
