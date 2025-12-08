mod body;
mod cuda;
mod geom;
mod kdtree;
mod renderer;
mod simul;

use crate::body::Body;
use crate::geom::Vec2;
use crate::simul::generate::generate_gaussian;

/// Constante gravitationnelle (à ajuster selon vos besoins)
const G: f32 = 6.674e-11; // ou une valeur plus adaptée à votre simulation

/// Calcule les accélérations gravitationnelles pour tous les corps
/// Complexité: O(n²)
pub fn compute_nsquares(bodies: &mut [Body]) {
    let n = bodies.len();

    // Réinitialiser les accélérations
    for body in bodies.iter_mut() {
        body.reset_acceleration();
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let delta = bodies[j].pos - bodies[i].pos;
            let distance_sq = delta.x * delta.x + delta.y * delta.y;

            if distance_sq < 1e-10 {
                continue;
            }

            let distance = distance_sq.sqrt();

            let force_magnitude = G * bodies[i].mass * bodies[j].mass / distance_sq;

            let force_direction = delta / distance;

            bodies[i].acc += force_direction * (force_magnitude / bodies[i].mass);
            bodies[j].acc -= force_direction * (force_magnitude / bodies[j].mass);
        }
    }
}

fn main() -> anyhow::Result<()> {
    // let dt = 0.01;

    // for n in 0..1000 {
    // compute_nsquares(&mut bodies);

    //// Mettre à jour les positions et vitesses
    // for body in bodies.iter_mut() {
    // body.update(dt);
    // }

    // println!("Itération {}", n);
    // for b in bodies.iter().take(5) {
    // println!("{}", b);
    // }
    // }

    renderer::run()
}
