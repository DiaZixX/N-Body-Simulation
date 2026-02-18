//! @file instance.rs
//! @brief Instance data for GPU instancing

use crate::body::Body;

/// @brief Instance data sent to GPU for each body
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub scale: f32,
    pub _padding: f32, // Pour alignement 16 bytes
}

impl InstanceRaw {
    /// @brief Returns the vertex buffer layout descriptor for instances
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Scale
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// @brief Converts bodies to instance data
///
/// @param bodies Slice of bodies
/// @param scale Global scale factor
/// @return Vector of instance data
pub fn bodies_to_instances(bodies: &[Body], scale: f32) -> Vec<InstanceRaw> {
    bodies
        .iter()
        .map(|body| {
            #[cfg(feature = "vec2")]
            let position = [body.pos.x / scale, body.pos.y / scale, 0.0];

            #[cfg(feature = "vec3")]
            let position = [body.pos.x / scale, body.pos.y / scale, body.pos.z / scale];

            InstanceRaw {
                position,
                color: [1.0, 1.0, 1.0],
                scale: body.radius,
                _padding: 0.0,
            }
        })
        .collect()
}
