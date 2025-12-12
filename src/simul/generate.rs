use crate::body::Body;
use crate::geom::Vec2;
use rand::Rng;
use rand_distr::Uniform;
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

pub fn generate_solar_system_varied(
    n_planets: usize,
    center: Vec2,
    star_mass: f32,
    star_radius: f32,
    min_orbit_radius: f32,
    max_orbit_radius: f32,
) -> Vec<Body> {
    let mut bodies = Vec::with_capacity(n_planets + 1);
    let mut rng = rand::thread_rng();

    // Étoile centrale
    bodies.push(Body::new(center, Vec2::zero(), star_mass, star_radius));

    // Planètes avec masses et tailles variées
    for i in 0..n_planets {
        let orbit_radius = min_orbit_radius
            + (max_orbit_radius - min_orbit_radius) * (i as f32 / n_planets.max(1) as f32);

        let angle = rng.gen_range(0.0..std::f32::consts::TAU);

        let pos = Vec2::new(
            center.x + orbit_radius * angle.cos(),
            center.y + orbit_radius * angle.sin(),
        );

        // Masse planétaire proportionnelle à la distance (planètes plus lointaines = plus massives)
        let planet_mass = 0.001 * star_mass * (1.0 + (i as f32 / n_planets as f32));
        let planet_radius = 0.01 + 0.02 * (planet_mass / star_mass).sqrt();

        let orbital_speed = (star_mass / orbit_radius).sqrt();
        let vel = Vec2::new(-angle.sin(), angle.cos()) * orbital_speed;

        bodies.push(Body::new(pos, vel, planet_mass, planet_radius));
    }

    bodies
}
