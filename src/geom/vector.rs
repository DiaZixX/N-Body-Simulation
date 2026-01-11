//! @file vector.rs
//! @brief Conditional vector type selection based on compilation features

#[cfg(feature = "vec2")]
pub use crate::geom::vec2::Vec2 as Vector;

#[cfg(feature = "vec3")]
pub use crate::geom::vec3::Vec3 as Vector;
