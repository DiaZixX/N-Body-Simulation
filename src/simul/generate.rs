use crate::body::Body;
use crate::geom::Vec2;
use rand_distr::{Distribution, Normal};

pub fn generate_gaussian(n: usize, center: Vec2, sigma: f32, mass: f32, radius: f32) -> Vec<Body> {
    let normal_x = Normal::new(center.x, sigma).unwrap();
    let normal_y = Normal::new(center.y, sigma).unwrap();
    let mut rng = rand::thread_rng();

    let mut bodies = Vec::with_capacity(n);
    for _ in 0..n {
        let pos = Vec2::new(normal_x.sample(&mut rng), normal_y.sample(&mut rng));
        let vel = Vec2::zero();
        bodies.push(Body::new(pos, vel, mass, radius));
    }
    bodies
}
