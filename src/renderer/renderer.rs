use std::sync::Arc;
use wgpu::util::DeviceExt;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::body::Body;
use crate::geom::Vec2;
use crate::kdtree::{Quad, Quadtree};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

// If the struct includes types that don't implement Pod and Zeroable, need to implement traits manually.
// Traits don't require to implement any methods, just need to use the following to get the code to work.
// unsafe impl bytemuck::Pod for Vertex {}
// unsafe impl bytemuck::Zeroable for Vertex {}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
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

// Fonction pour générer les vertices d'un cercle
fn generate_circle_vertices(
    center_x: f32,
    center_y: f32,
    radius: f32,
    segments: u32,
    color: [f32; 3],
) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Centre du cercle
    vertices.push(Vertex {
        position: [center_x, center_y, 0.0],
        color,
    });

    // Générer les points sur le périmètre du cercle
    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * 2.0 * std::f32::consts::PI;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();

        vertices.push(Vertex {
            position: [x, y, 0.0],
            color,
        });
    }

    // Générer les indices pour les triangles
    for i in 0..segments {
        indices.push(0); // Centre
        indices.push((i + 1) as u16);
        indices.push((i + 2) as u16);
    }

    (vertices, indices)
}

// Générer plusieurs cercles - Thème espace/étoiles
fn generate_circles() -> (Vec<Vertex>, Vec<u16>) {
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    // Grande étoile blanche brillante au centre
    let (vertices1, indices1) = generate_circle_vertices(0.0, 0.0, 0.15, 32, [1.0, 1.0, 0.95]);
    all_vertices.extend(vertices1);
    all_indices.extend(indices1);

    // Étoile jaune-orange (comme le soleil)
    let offset = all_vertices.len() as u16;
    let (vertices2, indices2) = generate_circle_vertices(0.5, 0.5, 0.12, 32, [1.0, 0.85, 0.3]);
    all_vertices.extend(vertices2);
    all_indices.extend(indices2.iter().map(|&i| i + offset));

    // Étoile bleu-blanc (étoile chaude)
    let offset = all_vertices.len() as u16;
    let (vertices3, indices3) = generate_circle_vertices(-0.6, 0.3, 0.08, 32, [0.7, 0.85, 1.0]);
    all_vertices.extend(vertices3);
    all_indices.extend(indices3.iter().map(|&i| i + offset));

    // Petite étoile blanche
    let offset = all_vertices.len() as u16;
    let (vertices4, indices4) = generate_circle_vertices(0.7, -0.2, 0.05, 24, [0.95, 0.95, 1.0]);
    all_vertices.extend(vertices4);
    all_indices.extend(indices4.iter().map(|&i| i + offset));

    // Étoile rouge (étoile froide/géante rouge)
    let offset = all_vertices.len() as u16;
    let (vertices5, indices5) = generate_circle_vertices(-0.4, -0.5, 0.1, 32, [1.0, 0.4, 0.3]);
    all_vertices.extend(vertices5);
    all_indices.extend(indices5.iter().map(|&i| i + offset));

    // Petites étoiles dispersées
    let offset = all_vertices.len() as u16;
    let (vertices6, indices6) = generate_circle_vertices(0.3, -0.7, 0.04, 16, [0.9, 0.9, 0.95]);
    all_vertices.extend(vertices6);
    all_indices.extend(indices6.iter().map(|&i| i + offset));

    let offset = all_vertices.len() as u16;
    let (vertices7, indices7) = generate_circle_vertices(-0.8, -0.1, 0.06, 20, [0.85, 0.9, 1.0]);
    all_vertices.extend(vertices7);
    all_indices.extend(indices7.iter().map(|&i| i + offset));

    let offset = all_vertices.len() as u16;
    let (vertices8, indices8) = generate_circle_vertices(0.15, 0.75, 0.045, 16, [1.0, 0.95, 0.8]);
    all_vertices.extend(vertices8);
    all_indices.extend(indices8.iter().map(|&i| i + offset));

    (all_vertices, all_indices)
}

// Fonction pour générer les vertices d'une sphère
fn generate_sphere_vertices(
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    stacks: u32,  // Nombre de divisions verticales
    sectors: u32, // Nombre de divisions horizontales
    color: [f32; 3],
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Générer les vertices de la sphère
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

    // Générer les indices pour les triangles
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

// Générer plusieurs sphères - Thème espace/étoiles
fn generate_spheres() -> (Vec<Vertex>, Vec<u32>) {
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    // Grande étoile blanche brillante au centre
    let (vertices1, indices1) =
        generate_sphere_vertices(0.0, 0.0, 0.0, 0.15, 16, 32, [1.0, 1.0, 0.95]);
    all_vertices.extend(vertices1);
    all_indices.extend(indices1);

    // Étoile jaune-orange (comme le soleil)
    let offset = all_vertices.len() as u32;
    let (vertices2, indices2) =
        generate_sphere_vertices(0.5, 0.5, 0.0, 0.12, 16, 32, [1.0, 0.85, 0.3]);
    all_vertices.extend(vertices2);
    all_indices.extend(indices2.iter().map(|&i| i + offset));

    // Étoile bleu-blanc (étoile chaude)
    let offset = all_vertices.len() as u32;
    let (vertices3, indices3) =
        generate_sphere_vertices(-0.6, 0.3, -0.2, 0.08, 12, 24, [0.7, 0.85, 1.0]);
    all_vertices.extend(vertices3);
    all_indices.extend(indices3.iter().map(|&i| i + offset));

    // Petite étoile blanche
    let offset = all_vertices.len() as u32;
    let (vertices4, indices4) =
        generate_sphere_vertices(0.7, -0.2, 0.1, 0.05, 10, 20, [0.95, 0.95, 1.0]);
    all_vertices.extend(vertices4);
    all_indices.extend(indices4.iter().map(|&i| i + offset));

    // Étoile rouge (étoile froide/géante rouge)
    let offset = all_vertices.len() as u32;
    let (vertices5, indices5) =
        generate_sphere_vertices(-0.4, -0.5, 0.3, 0.1, 14, 28, [1.0, 0.4, 0.3]);
    all_vertices.extend(vertices5);
    all_indices.extend(indices5.iter().map(|&i| i + offset));

    // Petites étoiles dispersées
    let offset = all_vertices.len() as u32;
    let (vertices6, indices6) =
        generate_sphere_vertices(0.3, -0.7, -0.1, 0.04, 8, 16, [0.9, 0.9, 0.95]);
    all_vertices.extend(vertices6);
    all_indices.extend(indices6.iter().map(|&i| i + offset));

    let offset = all_vertices.len() as u32;
    let (vertices7, indices7) =
        generate_sphere_vertices(-0.8, -0.1, 0.2, 0.06, 10, 20, [0.85, 0.9, 1.0]);
    all_vertices.extend(vertices7);
    all_indices.extend(indices7.iter().map(|&i| i + offset));

    let offset = all_vertices.len() as u32;
    let (vertices8, indices8) =
        generate_sphere_vertices(0.15, 0.75, -0.3, 0.045, 8, 16, [1.0, 0.95, 0.8]);
    all_vertices.extend(vertices8);
    all_indices.extend(indices8.iter().map(|&i| i + offset));

    (all_vertices, all_indices)
}

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and the eye so
            // that it doesn't change. The eye, therefore, still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

// This store the state of our simulation
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    window: Arc<Window>,
    num_indices: u32,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Structure pour les bodies
    bodies: Vec<Body>,
    dt: f32, // Pas de temps pour la simulation
    // Structure pour kdtree
    quadtree: Quadtree,
    theta: f32, // Paramètre theta pour Barnes-Hut (précision)
    epsilon: f32,
}

fn bodies_to_vertices_indices(bodies: &[Body]) -> (Vec<Vertex>, Vec<u32>) {
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    for body in bodies {
        let offset = all_vertices.len() as u32;

        // Générer un cercle/sphère pour chaque body
        // Position normalisée (bodies sont dans [-1, 1])
        let (vertices, indices) = generate_sphere_vertices(
            body.pos.x,      // center_x
            body.pos.y,      // center_y
            0.0,             // center_z (plan z=0)
            body.radius,     // radius
            8,               // stacks (divisions verticales)
            16,              // sectors (divisions horizontales)
            [1.0, 1.0, 1.0], // couleur blanche
        );

        all_vertices.extend(vertices);
        all_indices.extend(indices.iter().map(|&i| i + offset));
    }

    (all_vertices, all_indices)
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        // Adapt to the GPU (maybe interesting for compilation)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // exp_feat : specify if we want to use unstable feature or not
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        // Config definition for the surface we are using
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code assumes an sRGB surface texture.
        // Using a different one will result in all the colors coming out darker.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shader.wgsl").into()),
        });

        //use crate::generate_gaussian; // Assurez-vous que cette fonction est accessible
        //let bodies = generate_gaussian(
        //    1000,                // nombre de bodies
        //    Vec2::new(0.0, 0.0), // centre
        //    0.3,                 // sigma
        //    1.0,                 // masse
        //    0.02,                // radius
        //);
        use crate::generate_solar_system_varied;
        let bodies = generate_solar_system_varied(
            2, // 10 planètes
            Vec2::new(0.0, 0.0),
            100.0, // masse étoile
            0.05,  // rayon étoile
            0.15,  // rayon orbital min
            0.9,   // rayon orbital max
        );

        // Convertir les bodies en vertices/indices
        let (vertices, indices) = bodies_to_vertices_indices(&bodies);
        // Générer les cercles
        // let (vertices, indices) = generate_spheres();
        let mut quadtree = Quadtree::new();
        // Créer le quad englobant tous les bodies
        let quad = Quad::new_containing(&bodies);
        quadtree.clear(quad);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = indices.len() as u32;

        let camera = Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 0.5, 3.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let camera_controller = CameraController::new(0.2);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            window,
            num_indices,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            bodies,
            dt: 0.001,
            quadtree,
            theta: 0.5, // Valeur standard pour Barnes-Hut (0.5 = bon compromis vitesse/précision)
            epsilon: 0.01,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        let method = false;

        if method {
            use crate::compute_nsquares;
            compute_nsquares(&mut self.bodies);
        } else {
            let quad = Quad::new_containing(&self.bodies);
            self.quadtree.clear(quad);

            for body in &self.bodies {
                self.quadtree.insert(body.pos, body.mass);
            }

            self.quadtree.propagate();
            static mut FRAME_COUNT: u32 = 0;
            unsafe {
                FRAME_COUNT += 1;
                if FRAME_COUNT % 30 == 0 {
                    let nodes_count = self.quadtree.nodes.len();
                    let ratio = nodes_count as f32 / self.bodies.len() as f32;
                    let tmp = FRAME_COUNT;
                    println!(
                        "Frame {}: {} bodies → {} nodes (ratio: {:.2})",
                        tmp,
                        self.bodies.len(),
                        nodes_count,
                        ratio
                    );

                    // RATIO NORMAL: 1.3 à 4.0
                    // RATIO ANORMAL: > 10.0
                    if ratio > 10.0 {
                        println!("⚠️  WARNING: Trop de nodes! Subdivisions excessives détectées!");
                    }
                }
            }

            for body in &mut self.bodies {
                body.reset_acceleration();
                body.acc = self.quadtree.acc(body.pos, self.theta, self.epsilon) * 9.81;
            }
        }

        for body in &mut self.bodies {
            body.update(self.dt);
        }

        // Régénérer les vertices/indices
        let (vertices, indices) = bodies_to_vertices_indices(&self.bodies);

        // === SOLUTION: Recréer les buffers au lieu de write_buffer ===
        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        self.index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        self.num_indices = indices.len() as u32;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        // Can't render unless the surface is configured
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        // Control how the render code interacts with the texture
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Build a command buffer that we can then send to the GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Extra brackets in order to drop variable outside of the scope
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.00,
                                g: 0.00,
                                b: 0.02,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // Submit accept anything that implem IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if code == KeyCode::Escape && is_pressed {
            event_loop.exit();
        } else {
            self.camera_controller.handle_key(code, is_pressed);
        }
    }
}

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
