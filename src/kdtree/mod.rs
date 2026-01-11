//! @file mod.rs
//! @brief K-dimensional tree module for spatial partitioning
//!
//! This module implements the Barnes-Hut algorithm for efficient N-body simulation.
//! It provides hierarchical spatial data structures that adapt to dimensionality:
//! - 2D mode (vec2): Quadtree with 4 children per node
//! - 3D mode (vec3): Octree with 8 children per node
//!
//! The tree is used to approximate gravitational forces by treating distant
//! groups of bodies as single point masses, reducing complexity from O(N²) to O(N log N).
//!
//! ## Components:
//! - **KdCell**: Spatial bounding box (square in 2D, cube in 3D)
//! - **Node**: Tree node storing mass, center of mass, and children
//! - **KdTree**: Complete tree structure with insertion and force computation
//!

pub mod kdcell;
pub mod kdtree;
pub mod node;

pub use kdcell::KdCell;
pub use kdtree::KdTree;
pub use node::Node;
