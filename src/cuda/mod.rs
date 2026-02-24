//! @file mod.rs
//! @brief CUDA FFI bindings and public GPU API

use crate::body::Body;

// ============================================================================
// N² Direct Method
// ============================================================================

#[cfg(feature = "vec2")]
unsafe extern "C" {
    fn cuda_compute_forces_nsquare(
        pos_x: *const f32,
        pos_y: *const f32,
        masses: *const f32,
        acc_x: *mut f32,
        acc_y: *mut f32,
        n: i32,
        epsilon_sq: f32,
        G: f32,
    );

    fn cuda_update_bodies(
        pos_x: *mut f32,
        pos_y: *mut f32,
        vel_x: *mut f32,
        vel_y: *mut f32,
        acc_x: *const f32,
        acc_y: *const f32,
        n: i32,
        dt: f32,
    );
}

#[cfg(feature = "vec3")]
unsafe extern "C" {
    fn cuda_compute_forces_nsquare(
        pos_x: *const f32,
        pos_y: *const f32,
        pos_z: *const f32,
        masses: *const f32,
        acc_x: *mut f32,
        acc_y: *mut f32,
        acc_z: *mut f32,
        n: i32,
        epsilon_sq: f32,
        G: f32,
    );

    fn cuda_update_bodies(
        pos_x: *mut f32,
        pos_y: *mut f32,
        pos_z: *mut f32,
        vel_x: *mut f32,
        vel_y: *mut f32,
        vel_z: *mut f32,
        acc_x: *const f32,
        acc_y: *const f32,
        acc_z: *const f32,
        n: i32,
        dt: f32,
    );
}

// ============================================================================
// Barnes-Hut Method
// ============================================================================

#[cfg(feature = "vec2")]
unsafe extern "C" {
    fn cuda_barnes_hut_forces(
        pos_x: *const f32,
        pos_y: *const f32,
        masses: *const f32,
        acc_x: *mut f32,
        acc_y: *mut f32,
        n: i32,
        theta: f32,
        epsilon: f32,
        G: f32,
    );
}

#[cfg(feature = "vec3")]
unsafe extern "C" {
    fn cuda_barnes_hut_forces(
        pos_x: *const f32,
        pos_y: *const f32,
        pos_z: *const f32,
        masses: *const f32,
        acc_x: *mut f32,
        acc_y: *mut f32,
        acc_z: *mut f32,
        n: i32,
        theta: f32,
        epsilon: f32,
        G: f32,
    );
}

/// @brief Computes gravitational forces using the CUDA N² algorithm.
///
/// @param bodies Mutable slice of bodies; accelerations are updated in place
/// @param epsilon Softening length
/// @param G Gravitational constant
pub fn compute_forces_cuda(bodies: &mut [Body], epsilon: f32, G: f32) {
    let n = bodies.len() as i32;
    let epsilon_sq = epsilon * epsilon;

    let pos_x: Vec<f32> = bodies.iter().map(|b| b.pos.x).collect();
    let pos_y: Vec<f32> = bodies.iter().map(|b| b.pos.y).collect();
    let masses: Vec<f32> = bodies.iter().map(|b| b.mass).collect();

    let mut acc_x = vec![0.0f32; n as usize];
    let mut acc_y = vec![0.0f32; n as usize];

    #[cfg(feature = "vec2")]
    unsafe {
        cuda_compute_forces_nsquare(
            pos_x.as_ptr(),
            pos_y.as_ptr(),
            masses.as_ptr(),
            acc_x.as_mut_ptr(),
            acc_y.as_mut_ptr(),
            n,
            epsilon_sq,
            G,
        );
    }

    #[cfg(feature = "vec3")]
    {
        let pos_z: Vec<f32> = bodies.iter().map(|b| b.pos.z).collect();
        let mut acc_z = vec![0.0f32; n as usize];

        unsafe {
            cuda_compute_forces_nsquare(
                pos_x.as_ptr(),
                pos_y.as_ptr(),
                pos_z.as_ptr(),
                masses.as_ptr(),
                acc_x.as_mut_ptr(),
                acc_y.as_mut_ptr(),
                acc_z.as_mut_ptr(),
                n,
                epsilon_sq,
                G,
            );
        }

        for (i, body) in bodies.iter_mut().enumerate() {
            body.acc.x = acc_x[i];
            body.acc.y = acc_y[i];
            body.acc.z = acc_z[i];
        }
        return;
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        body.acc.x = acc_x[i];
        body.acc.y = acc_y[i];
    }
}

/// @brief Computes gravitational forces using the CUDA Barnes-Hut algorithm.
///
/// @param bodies Mutable slice of bodies; accelerations are updated in place
/// @param theta Barnes-Hut opening angle
/// @param epsilon Softening length
/// @param G Gravitational constant
pub fn compute_forces_barnes_hut_cuda(bodies: &mut [Body], theta: f32, epsilon: f32, G: f32) {
    let n = bodies.len() as i32;

    let pos_x: Vec<f32> = bodies.iter().map(|b| b.pos.x).collect();
    let pos_y: Vec<f32> = bodies.iter().map(|b| b.pos.y).collect();
    let masses: Vec<f32> = bodies.iter().map(|b| b.mass).collect();

    let mut acc_x = vec![0.0f32; n as usize];
    let mut acc_y = vec![0.0f32; n as usize];

    #[cfg(feature = "vec2")]
    unsafe {
        cuda_barnes_hut_forces(
            pos_x.as_ptr(),
            pos_y.as_ptr(),
            masses.as_ptr(),
            acc_x.as_mut_ptr(),
            acc_y.as_mut_ptr(),
            n,
            theta,
            epsilon,
            G,
        );
    }

    #[cfg(feature = "vec3")]
    {
        let pos_z: Vec<f32> = bodies.iter().map(|b| b.pos.z).collect();
        let mut acc_z = vec![0.0f32; n as usize];

        unsafe {
            cuda_barnes_hut_forces(
                pos_x.as_ptr(),
                pos_y.as_ptr(),
                pos_z.as_ptr(),
                masses.as_ptr(),
                acc_x.as_mut_ptr(),
                acc_y.as_mut_ptr(),
                acc_z.as_mut_ptr(),
                n,
                theta,
                epsilon,
                G,
            );
        }

        for (i, body) in bodies.iter_mut().enumerate() {
            body.acc.x = acc_x[i];
            body.acc.y = acc_y[i];
            body.acc.z = acc_z[i];
        }
        return;
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        body.acc.x = acc_x[i];
        body.acc.y = acc_y[i];
    }
}

/// @brief Updates body positions and velocities using CUDA (Euler integration).
///
/// @param bodies Mutable slice of bodies to integrate
/// @param dt Time step
pub fn update_bodies_cuda(bodies: &mut [Body], dt: f32) {
    let n = bodies.len() as i32;

    let mut pos_x: Vec<f32> = bodies.iter().map(|b| b.pos.x).collect();
    let mut pos_y: Vec<f32> = bodies.iter().map(|b| b.pos.y).collect();
    let mut vel_x: Vec<f32> = bodies.iter().map(|b| b.vel.x).collect();
    let mut vel_y: Vec<f32> = bodies.iter().map(|b| b.vel.y).collect();
    let acc_x: Vec<f32> = bodies.iter().map(|b| b.acc.x).collect();
    let acc_y: Vec<f32> = bodies.iter().map(|b| b.acc.y).collect();

    #[cfg(feature = "vec2")]
    unsafe {
        cuda_update_bodies(
            pos_x.as_mut_ptr(),
            pos_y.as_mut_ptr(),
            vel_x.as_mut_ptr(),
            vel_y.as_mut_ptr(),
            acc_x.as_ptr(),
            acc_y.as_ptr(),
            n,
            dt,
        );
    }

    #[cfg(feature = "vec3")]
    {
        let mut pos_z: Vec<f32> = bodies.iter().map(|b| b.pos.z).collect();
        let mut vel_z: Vec<f32> = bodies.iter().map(|b| b.vel.z).collect();
        let acc_z: Vec<f32> = bodies.iter().map(|b| b.acc.z).collect();

        unsafe {
            cuda_update_bodies(
                pos_x.as_mut_ptr(),
                pos_y.as_mut_ptr(),
                pos_z.as_mut_ptr(),
                vel_x.as_mut_ptr(),
                vel_y.as_mut_ptr(),
                vel_z.as_mut_ptr(),
                acc_x.as_ptr(),
                acc_y.as_ptr(),
                acc_z.as_ptr(),
                n,
                dt,
            );
        }

        for (i, body) in bodies.iter_mut().enumerate() {
            body.pos.x = pos_x[i];
            body.pos.y = pos_y[i];
            body.pos.z = pos_z[i];
            body.vel.x = vel_x[i];
            body.vel.y = vel_y[i];
            body.vel.z = vel_z[i];
        }
        return;
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        body.pos.x = pos_x[i];
        body.pos.y = pos_y[i];
        body.vel.x = vel_x[i];
        body.vel.y = vel_y[i];
    }
}
