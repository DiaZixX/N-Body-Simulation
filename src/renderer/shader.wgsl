// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct InstanceInput {
    @location(2) instance_position: vec3<f32>,
    @location(3) instance_color: vec3<f32>,
    @location(4) instance_scale: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Scale the unit sphere and translate to instance position
    let scaled_pos = model.position * instance.instance_scale;
    let world_pos = scaled_pos + instance.instance_position;
    
    out.color = instance.instance_color;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
