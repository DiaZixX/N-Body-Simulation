//! @file lib.rs
//! @brief Library entry point for n-body simulation

pub mod body;
pub mod geom;
pub mod kdtree;
pub mod simul;

pub use body::Body;
pub use geom::Vector;
pub use kdtree::{KdCell, KdTree, Node};
pub use simul::{generate_gaussian, generate_solar_system, generate_uniform};
