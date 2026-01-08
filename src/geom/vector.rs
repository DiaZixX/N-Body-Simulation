//! @file vector.rs
//! @brief Conditional vector type selection based on compilation features

#[cfg(feature = "vec2")]
pub use crate::geom::vec2::Vec2 as Vector;

#[cfg(feature = "vec3")]
pub use crate::geom::vec3::Vec3 as Vector;

// Type alias pour la clarté
#[cfg(feature = "vec2")]
pub type VectorType = crate::geom::vec2::Vec2;

#[cfg(feature = "vec3")]
pub type VectorType = crate::geom::vec3::Vec3;
