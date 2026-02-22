//! @file mod.rs
//! @brief Renderer module for GPU-accelerated N-body visualization
//!
//! This module provides WGPU-based rendering capabilities for visualizing
//! N-body simulations with camera controls and real-time updates.

pub mod app;
pub mod camera;
pub mod config;
pub mod instance;
pub mod state;
pub mod vertex;

pub use app::{App, run};
pub use camera::{Camera, CameraController};
pub use config::GuiConfig;
pub use instance::{InstanceRaw, bodies_to_instances};
pub use state::State;
pub use vertex::Vertex;
