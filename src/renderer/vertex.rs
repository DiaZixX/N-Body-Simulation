//! @file vertex.rs
//! @brief Vertex structures and generation functions

use crate::body::Body;

/// @brief Vertex structure for GPU rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    /// @brief Returns the vertex buffer layout descriptor
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// @brief Generates vertices and indices for a circle (2D)
///
/// @param center_x X coordinate of circle center
/// @param center_y Y coordinate of circle center
/// @param radius Circle radius
/// @param segments Number of segments
/// @param color RGB color
/// @return Tuple of (vertices, indices)
pub fn generate_circle_vertices(
    center_x: f32,
    center_y: f32,
    radius: f32,
    segments: u32,
    color: [f32; 3],
) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    vertices.push(Vertex {
        position: [center_x, center_y, 0.0],
        color,
    });

    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * 2.0 * std::f32::consts::PI;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();

        vertices.push(Vertex {
            position: [x, y, 0.0],
            color,
        });
    }

    for i in 0..segments {
        indices.push(0);
        indices.push((i + 1) as u16);
        indices.push((i + 2) as u16);
    }

    (vertices, indices)
}

/// @brief Generates vertices and indices for a sphere (3D)
///
/// @param center_x X coordinate of sphere center
/// @param center_y Y coordinate of sphere center
/// @param center_z Z coordinate of sphere center
/// @param radius Sphere radius
/// @param stacks Number of vertical divisions
/// @param sectors Number of horizontal divisions
/// @param color RGB color
/// @return Tuple of (vertices, indices)
pub fn generate_sphere_vertices(
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    stacks: u32,
    sectors: u32,
    color: [f32; 3],
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let stack_angle =
            std::f32::consts::PI / 2.0 - (i as f32 * std::f32::consts::PI / stacks as f32);
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();

        for j in 0..=sectors {
            let sector_angle = j as f32 * 2.0 * std::f32::consts::PI / sectors as f32;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            vertices.push(Vertex {
                position: [center_x + x, center_y + y, center_z + z],
                color,
            });
        }
    }

    for i in 0..stacks {
        let k1 = i * (sectors + 1);
        let k2 = k1 + sectors + 1;

        for j in 0..sectors {
            if i != 0 {
                indices.push((k1 + j) as u32);
                indices.push((k2 + j) as u32);
                indices.push((k1 + j + 1) as u32);
            }

            if i != stacks - 1 {
                indices.push((k1 + j + 1) as u32);
                indices.push((k2 + j) as u32);
                indices.push((k2 + j + 1) as u32);
            }
        }
    }

    (vertices, indices)
}

/// @brief Converts bodies to vertices and indices for rendering
///
/// @param bodies Slice of bodies to convert
/// @return Tuple of (vertices, indices)
pub fn bodies_to_vertices_indices(bodies: &[Body]) -> (Vec<Vertex>, Vec<u32>) {
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let scale = 4.5e12;

    for body in bodies {
        let offset = all_vertices.len() as u32;

        #[cfg(feature = "vec2")]
        let (vertices, indices) = generate_sphere_vertices(
            body.pos.x / scale,
            body.pos.y / scale,
            0.0,
            body.radius,
            8,
            16,
            [1.0, 1.0, 1.0],
        );

        #[cfg(feature = "vec3")]
        let (vertices, indices) = generate_sphere_vertices(
            body.pos.x / scale,
            body.pos.y / scale,
            body.pos.z / scale,
            body.radius,
            8,
            16,
            [1.0, 1.0, 1.0],
        );

        all_vertices.extend(vertices);
        all_indices.extend(indices.iter().map(|&i| i + offset));
    }

    (all_vertices, all_indices)
}
