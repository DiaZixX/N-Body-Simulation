use std::ffi::c_void;
use std::os::raw::c_int;

#[link(name = "compute_forces_gpu", kind = "static")]
unsafe extern "C" {
    fn compute_forces_gpu(
        x: *mut f32,
        y: *mut f32,
        vx: *mut f32,
        vy: *mut f32,
        mass: *mut f32,
        n: c_int,
        G: f32,
        eps2: f32,
    );
}

pub fn compute_forces(bodies: &mut [crate::body::Body], dt: f32) {
    let n = bodies.len();
    if n == 0 {
        return;
    }
    let mut x: Vec<f32> = bodies.iter().map(|b| b.pos.x).collect();
    let mut y: Vec<f32> = bodies.iter().map(|b| b.pos.y).collect();
    let mut vx: Vec<f32> = bodies.iter().map(|b| b.vel.x).collect();
    let mut vy: Vec<f32> = bodies.iter().map(|b| b.vel.y).collect();
    let mut mass: Vec<f32> = bodies.iter().map(|b| b.mass as f32).collect();

    unsafe {
        compute_forces_gpu(
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            vx.as_mut_ptr(),
            vy.as_mut_ptr(),
            mass.as_mut_ptr(),
            n as i32,
            6.67430e-11_f32,
            1e-9_f32,
        );
    }

    // write back velocities (we update vel by dt externally if needed)
    for i in 0..n {
        bodies[i].vel.x = vx[i];
        bodies[i].vel.y = vy[i];
    }
}
