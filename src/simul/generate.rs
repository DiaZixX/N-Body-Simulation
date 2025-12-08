use crate::body::Body;
use crate::geom::Vec2;
use rand_distr::{Distribution, Normal};

pub fn generate_gaussian(n: usize, center: Vec2, sigma: f32, mass: f32, radius: f32) -> Vec<Body> {
    let normal_x = Normal::new(center.x, sigma).unwrap();
    let normal_y = Normal::new(center.y, sigma).unwrap();
    let mut rng = rand::thread_rng();
    let mut bodies = Vec::with_capacity(n);

    for _ in 0..n {
        let mut pos = Vec2::new(normal_x.sample(&mut rng), normal_y.sample(&mut rng));

        // Clamper la position dans [-1, 1]
        pos.x = pos.x.clamp(-10.0, 10.0);
        pos.y = pos.y.clamp(-10.0, 10.0);

        // Calculer le vecteur depuis le centre
        let delta = pos - center;
        let distance = delta.length();

        // Vitesse orbitale circulaire : v = sqrt(G * M_central / r)
        let orbital_speed = if distance > 1e-6 {
            // Formule simplifiée : vitesse proportionnelle à 1/sqrt(distance)
            let speed_factor = 0.5;
            speed_factor / distance.sqrt()
        } else {
            0.0
        };

        // Vecteur perpendiculaire pour rotation (sens antihoraire)
        let vel = Vec2::new(-delta.y, delta.x).normalize() * orbital_speed;

        bodies.push(Body::new(pos, vel, mass, radius));
    }
    bodies
}
