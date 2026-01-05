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

pub fn generate_solar_system() -> Vec<Body> {
    // Constante gravitationnelle (m³/kg/s²)
    const G: f32 = 6.674e-11;

    // Centre du système (le Soleil sera au centre)
    let center = Vec2::new(0.0, 0.0);
    // Masse du Soleil pour le calcul des vitesses orbitales
    let m_sun = 1.989e30;

    let mut bodies = Vec::new();

    // ========== SOLEIL ==========
    // Masse: 1.989 × 10^30 kg
    // On le place au centre avec vitesse nulle
    bodies.push(Body::new(
        center,
        Vec2::zero(),
        m_sun, // masse réelle
        0.010, // rayon visuel (agrandi pour être visible)
    ));

    // ========== PLANÈTES ==========
    // Format: (nom, distance_au_soleil_en_m, masse_en_kg, rayon_visuel)
    let planets = [
        // Mercure
        ("Mercure", 57.9e9, 3.301e23, 0.0016),
        // Vénus
        ("Venus", 108.2e9, 4.867e24, 0.0024),
        // Terre
        ("Terre", 149.6e9, 5.972e24, 0.0026),
        // Mars
        ("Mars", 227.9e9, 6.417e23, 0.0018),
        // Jupiter (géante gazeuse)
        ("Jupiter", 778.5e9, 1.898e27, 0.0070),
        // Saturne (géante gazeuse)
        ("Saturne", 1.434e12, 5.683e26, 0.0060),
        // Uranus (géante de glace)
        ("Uranus", 2.871e12, 8.681e25, 0.0040),
        // Neptune (géante de glace)
        ("Neptune", 4.495e12, 1.024e26, 0.0038),
    ];

    for &(name, distance, mass, visual_radius) in planets.iter() {
        // Position de la planète
        // On place chaque planète à un angle différent pour éviter les collisions initiales
        let angle = 2.0 * std::f32::consts::PI / planets.len() as f32;
        let scaled_distance = distance;

        let pos = Vec2::new(
            center.x + scaled_distance * angle.cos(),
            center.y + scaled_distance * angle.sin(),
        );

        // Calcul de la vitesse orbitale circulaire
        // Formule: v = sqrt(G * M_soleil / r)
        let orbital_velocity = (G * m_sun / scaled_distance).sqrt();

        // Vecteur vitesse perpendiculaire au rayon (mouvement circulaire)
        // Pour un mouvement antihoraire, on utilise (-sin, cos) au lieu de (cos, sin)
        let vel = Vec2::new(
            -orbital_velocity * angle.sin(),
            orbital_velocity * angle.cos(),
        );

        bodies.push(Body::new(pos, vel, mass, visual_radius));
    }

    println!("=== Système solaire mini généré ===");
    println!("Masse du Soleil: {}", m_sun);
    println!("Constante G: {}", G);
    for (i, body) in bodies.iter().enumerate().skip(1) {
        let r = (body.pos.x * body.pos.x + body.pos.y * body.pos.y).sqrt();
        let v = (body.vel.x * body.vel.x + body.vel.y * body.vel.y).sqrt();
        let v_theorique = (G * m_sun / r).sqrt();
        println!(
            "Planète {}: r={:.3}, v={:.6}, v_théorique={:.6}",
            i, r, v, v_theorique
        );
    }

    bodies
}

pub fn generate_solar_system_mini() -> Vec<Body> {
    let center = Vec2::new(0.0, 0.0);
    let mut bodies = Vec::new();

    // Constante gravitationnelle utilisée dans votre simulation
    // Elle doit correspondre à celle dans compute_nsquares (main.rs)
    const G: f32 = 6.674e-11;

    // Masse du Soleil (ajustée pour l'échelle de la simulation)
    let sun_mass = 1000.0;

    // Soleil au centre (gros et lourd)
    bodies.push(Body::new(
        center,
        Vec2::zero(),
        sun_mass,
        0.08, // rayon visuel
    ));

    // Planètes avec distances et masses simplifiées
    // Format: (distance, masse, rayon_visuel)
    let planets = [
        (0.15, 0.1, 0.015), // Mercure
        (0.25, 0.5, 0.020), // Vénus
        (0.35, 0.6, 0.022), // Terre
        (0.50, 0.3, 0.018), // Mars
        (0.70, 5.0, 0.045), // Jupiter
        (0.90, 3.0, 0.040), // Saturne
    ];

    for (i, &(distance, mass, radius)) in planets.iter().enumerate() {
        let angle = (i as f32) * (2.0 * std::f32::consts::PI / planets.len() as f32);

        let pos = Vec2::new(
            center.x + distance * angle.cos(),
            center.y + distance * angle.sin(),
        );

        // CALCUL CORRECT de la vitesse orbitale pour une orbite circulaire stable
        // Formule: v = sqrt(G * M_soleil / r)
        // Pour une orbite stable, la force gravitationnelle = force centripète
        let orbital_speed = (G * sun_mass / distance).sqrt();

        // Vitesse perpendiculaire au rayon (sens antihoraire)
        // Le vecteur (-sin(θ), cos(θ)) est perpendiculaire à (cos(θ), sin(θ))
        let vel = Vec2::new(-orbital_speed * angle.sin(), orbital_speed * angle.cos());

        bodies.push(Body::new(pos, vel, mass, radius));
    }

    println!("=== Système solaire mini généré ===");
    println!("Masse du Soleil: {}", sun_mass);
    println!("Constante G: {}", G);
    for (i, body) in bodies.iter().enumerate().skip(1) {
        let r = (body.pos.x * body.pos.x + body.pos.y * body.pos.y).sqrt();
        let v = (body.vel.x * body.vel.x + body.vel.y * body.vel.y).sqrt();
        let v_theorique = (G * sun_mass / r).sqrt();
        println!(
            "Planète {}: r={:.3}, v={:.6}, v_théorique={:.6}",
            i, r, v, v_theorique
        );
    }

    bodies
}
