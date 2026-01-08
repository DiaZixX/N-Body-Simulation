//! @file vector_tests.rs
//! @brief Integration tests for Vector operations

use n_body_simulation::{Body, Vector};

#[test]
fn test_vector_creation() {
    #[cfg(feature = "vec2")]
    {
        let v = Vector::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[cfg(feature = "vec3")]
    {
        let v = Vector::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }
}

#[test]
fn test_vector_zero() {
    let v = Vector::zero();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);

    #[cfg(feature = "vec3")]
    assert_eq!(v.z, 0.0);
}

#[test]
fn test_vector_addition() {
    #[cfg(feature = "vec2")]
    {
        let v1 = Vector::new(1.0, 2.0);
        let v2 = Vector::new(3.0, 4.0);
        let result = v1 + v2;
        assert_eq!(result.x, 4.0);
        assert_eq!(result.y, 6.0);
    }

    #[cfg(feature = "vec3")]
    {
        let v1 = Vector::new(1.0, 2.0, 3.0);
        let v2 = Vector::new(3.0, 4.0, 5.0);
        let result = v1 + v2;
        assert_eq!(result.x, 4.0);
        assert_eq!(result.y, 6.0);
        assert_eq!(result.z, 8.0);
    }
}

#[test]
fn test_vector_subtraction() {
    #[cfg(feature = "vec2")]
    {
        let v1 = Vector::new(5.0, 7.0);
        let v2 = Vector::new(2.0, 3.0);
        let result = v1 - v2;
        assert_eq!(result.x, 3.0);
        assert_eq!(result.y, 4.0);
    }

    #[cfg(feature = "vec3")]
    {
        let v1 = Vector::new(5.0, 7.0, 9.0);
        let v2 = Vector::new(2.0, 3.0, 4.0);
        let result = v1 - v2;
        assert_eq!(result.x, 3.0);
        assert_eq!(result.y, 4.0);
        assert_eq!(result.z, 5.0);
    }
}

#[test]
fn test_vector_scalar_multiplication() {
    #[cfg(feature = "vec2")]
    {
        let v = Vector::new(2.0, 3.0);
        let result = v * 2.0;
        assert_eq!(result.x, 4.0);
        assert_eq!(result.y, 6.0);
    }

    #[cfg(feature = "vec3")]
    {
        let v = Vector::new(2.0, 3.0, 4.0);
        let result = v * 2.0;
        assert_eq!(result.x, 4.0);
        assert_eq!(result.y, 6.0);
        assert_eq!(result.z, 8.0);
    }
}

#[test]
fn test_vector_scalar_division() {
    #[cfg(feature = "vec2")]
    {
        let v = Vector::new(4.0, 6.0);
        let result = v / 2.0;
        assert_eq!(result.x, 2.0);
        assert_eq!(result.y, 3.0);
    }

    #[cfg(feature = "vec3")]
    {
        let v = Vector::new(4.0, 6.0, 8.0);
        let result = v / 2.0;
        assert_eq!(result.x, 2.0);
        assert_eq!(result.y, 3.0);
        assert_eq!(result.z, 4.0);
    }
}

#[test]
fn test_vector_norm() {
    #[cfg(feature = "vec2")]
    {
        let v = Vector::new(3.0, 4.0);
        assert_eq!(v.norm(), 5.0);
    }

    #[cfg(feature = "vec3")]
    {
        let v = Vector::new(2.0, 3.0, 6.0);
        assert_eq!(v.norm(), 7.0);
    }
}

#[test]
fn test_vector_normalized() {
    #[cfg(feature = "vec2")]
    {
        let v = Vector::new(3.0, 4.0);
        let normalized = v.normalized();
        assert!((normalized.norm() - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "vec3")]
    {
        let v = Vector::new(2.0, 3.0, 6.0);
        let normalized = v.normalized();
        assert!((normalized.norm() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_vector_normalized_zero() {
    let v = Vector::zero();
    let normalized = v.normalized();
    assert_eq!(normalized.x, 0.0);
    assert_eq!(normalized.y, 0.0);

    #[cfg(feature = "vec3")]
    assert_eq!(normalized.z, 0.0);
}

#[test]
fn test_vector_distance() {
    #[cfg(feature = "vec2")]
    {
        let v1 = Vector::new(0.0, 0.0);
        let v2 = Vector::new(3.0, 4.0);
        assert_eq!(Vector::distance(v1, v2), 5.0);
    }

    #[cfg(feature = "vec3")]
    {
        let v1 = Vector::new(0.0, 0.0, 0.0);
        let v2 = Vector::new(2.0, 3.0, 6.0);
        assert_eq!(Vector::distance(v1, v2), 7.0);
    }
}

#[test]
fn test_vector_assignment_operators() {
    #[cfg(feature = "vec2")]
    {
        let mut v = Vector::new(1.0, 2.0);
        v += Vector::new(1.0, 1.0);
        assert_eq!(v.x, 2.0);
        assert_eq!(v.y, 3.0);

        v -= Vector::new(0.5, 0.5);
        assert_eq!(v.x, 1.5);
        assert_eq!(v.y, 2.5);

        v *= 2.0;
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 5.0);

        v /= 2.0;
        assert_eq!(v.x, 1.5);
        assert_eq!(v.y, 2.5);
    }

    #[cfg(feature = "vec3")]
    {
        let mut v = Vector::new(1.0, 2.0, 3.0);
        v += Vector::new(1.0, 1.0, 1.0);
        assert_eq!(v.x, 2.0);
        assert_eq!(v.y, 3.0);
        assert_eq!(v.z, 4.0);

        v *= 2.0;
        assert_eq!(v.x, 4.0);
        assert_eq!(v.y, 6.0);
        assert_eq!(v.z, 8.0);
    }
}

#[cfg(feature = "vec3")]
#[test]
fn test_vec3_dot_product() {
    let v1 = Vector::new(1.0, 2.0, 3.0);
    let v2 = Vector::new(4.0, 5.0, 6.0);
    assert_eq!(v1.dot(v2), 32.0); // 1*4 + 2*5 + 3*6 = 32
}

#[cfg(feature = "vec3")]
#[test]
fn test_vec3_cross_product() {
    let v1 = Vector::new(1.0, 0.0, 0.0);
    let v2 = Vector::new(0.0, 1.0, 0.0);
    let result = v1.cross(v2);
    assert_eq!(result.x, 0.0);
    assert_eq!(result.y, 0.0);
    assert_eq!(result.z, 1.0);
}

#[test]
fn test_body_creation() {
    #[cfg(feature = "vec2")]
    {
        let body = Body::new(Vector::new(1.0, 2.0), Vector::new(0.5, 0.3), 10.0, 1.5);
        assert_eq!(body.pos.x, 1.0);
        assert_eq!(body.mass, 10.0);
        assert_eq!(body.radius, 1.5);
    }

    #[cfg(feature = "vec3")]
    {
        let body = Body::new(
            Vector::new(1.0, 2.0, 3.0),
            Vector::new(0.5, 0.3, 0.1),
            10.0,
            1.5,
        );
        assert_eq!(body.pos.z, 3.0);
    }
}

#[test]
fn test_body_update() {
    #[cfg(feature = "vec2")]
    {
        let mut body = Body::new(Vector::new(0.0, 0.0), Vector::new(1.0, 0.0), 1.0, 0.5);
        body.acc = Vector::new(0.0, -9.81);
        body.update(1.0);

        assert_eq!(body.vel.x, 1.0);
        assert_eq!(body.vel.y, -9.81);
        assert_eq!(body.pos.x, 1.0);
    }

    #[cfg(feature = "vec3")]
    {
        let mut body = Body::new(
            Vector::new(0.0, 0.0, 0.0),
            Vector::new(1.0, 0.0, 0.0),
            1.0,
            0.5,
        );
        body.acc = Vector::new(0.0, -9.81, 0.0);
        body.update(1.0);

        assert_eq!(body.vel.y, -9.81);
    }
}
