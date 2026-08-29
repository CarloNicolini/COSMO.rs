#![allow(non_snake_case)]

use cosmo::{
    DualExponentialCone, DualPowerCone, ExponentialCone, NonnegativeCone, PowerCone,
    SecondOrderCone, ZeroCone,
};

fn inf_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn zero_projection() {
    let c = ZeroCone::new(3);
    let mut x = [1.0, -2.0, 3.5];
    c.project(&mut x);
    assert_eq!(x, [0.0, 0.0, 0.0]);
    assert!(c.in_cone(&[0.0, 0.0, 0.0], 1e-12));
    assert!(!c.in_cone(&[1e-6, 0.0, 0.0], 1e-12));
}

#[test]
fn nonnegative_projection_idempotent() {
    let c = NonnegativeCone::new(4);
    let mut x = [1.0, -2.0, 0.0, 3.5];
    c.project(&mut x);
    assert_eq!(x, [1.0, 0.0, 0.0, 3.5]);
    let y = x;
    c.project(&mut x);
    assert_eq!(x, y);
    assert!(c.in_cone(&x, 0.0));
}

#[test]
fn soc_inside_unchanged() {
    let c = SecondOrderCone::new(3);
    let mut x = [2.0, 0.6, 0.8];
    let orig = x;
    c.project(&mut x);
    assert!(inf_diff(&x, &orig) < 1e-14);
}

#[test]
fn soc_outside() {
    let c = SecondOrderCone::new(3);
    let mut x = [0.0, 3.0, 4.0];
    c.project(&mut x);
    // should land on the boundary
    let n = (x[1] * x[1] + x[2] * x[2]).sqrt();
    assert!((x[0] - n).abs() < 1e-10);
    let mut y = x;
    c.project(&mut y);
    assert!(inf_diff(&x, &y) < 1e-12);
}

#[test]
fn soc_negative_ray() {
    let c = SecondOrderCone::new(3);
    let mut x = [-10.0, 1.0, 1.0];
    c.project(&mut x);
    assert!(x.iter().all(|&v| v.abs() < 1e-14));
}

#[test]
fn exp_projection_in_cone() {
    let mut c = ExponentialCone::new();
    // y=1, x=0, z=1 is in the cone
    let mut x = [0.0, 1.0, 1.0];
    c.project(&mut x);
    assert!(c.in_cone(&x, 1e-8));
}

#[test]
fn exp_projection_boundary_and_origin() {
    let mut c = ExponentialCone::new();
    let mut z = [0.0, 0.0, 0.0];
    c.project(&mut z);
    assert!(z.iter().all(|v| v.abs() < 1e-12));

    let mut v = [-1.0, -1.0, 2.0];
    c.project(&mut v);
    assert!((v[0] + 1.0).abs() < 1e-12);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 2.0);
}

#[test]
fn exp_projection_idempotent_random() {
    let mut c = ExponentialCone::new();
    let pts = [
        [1.0, 0.5, 0.1],
        [-2.0, 3.0, 0.01],
        [0.0, -1.0, 1.0],
        [5.0, 5.0, 5.0],
        [1e-8, 1e-8, 1e-8],
        [10.0, 0.1, 100.0],
    ];
    for p in pts {
        let mut x = p;
        c.project(&mut x);
        let y = x;
        c.project(&mut x);
        assert!(inf_diff(&x, &y) < 1e-6, "not idempotent {:?} -> {:?}", p, x);
        assert!(x.iter().all(|v| v.is_finite()), "non-finite {:?}", p);
        assert!(
            c.in_cone(&x, 1e-5),
            "projection of {:?} not in cone: {:?}",
            p,
            x
        );
    }
}

#[test]
fn dual_exp_moreau() {
    let mut d = DualExponentialCone::new();
    let mut p = ExponentialCone::new();
    let v = [0.3, -0.2, 0.7];
    let mut dv = v;
    d.project(&mut dv);
    let mut pv = [-v[0], -v[1], -v[2]];
    p.project(&mut pv);
    for i in 0..3 {
        assert!((dv[i] - (v[i] + pv[i])).abs() < 1e-6);
    }
}

#[test]
fn power_projection_in_cone() {
    let c = PowerCone::new(0.6);
    let mut x = [1.0, 1.0, 0.1];
    c.project(&mut x);
    assert!(c.in_cone(&x, 1e-8));
}

#[test]
fn power_projection_idempotent() {
    let c = PowerCone::new(0.4);
    let pts = [[1.0, -1.0, 2.0], [0.5, 0.5, 0.9], [-0.2, 0.3, 0.0]];
    for p in pts {
        let mut x = p;
        c.project(&mut x);
        let y = x;
        c.project(&mut x);
        assert!(inf_diff(&x, &y) < 1e-6, "not idempotent {p:?}");
        assert!(x.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn dual_power_moreau() {
    let mut d = DualPowerCone::new(0.3);
    let p = PowerCone::new(0.3);
    let v = [0.4, 0.2, -0.5];
    let mut dv = v;
    d.project(&mut dv);
    let mut pv = [-v[0], -v[1], -v[2]];
    p.project(&mut pv);
    for i in 0..3 {
        assert!((dv[i] - (v[i] + pv[i])).abs() < 1e-6);
    }
}

#[test]
fn nan_inputs_do_not_panic() {
    let mut e = ExponentialCone::new();
    let mut x = [f64::NAN, 1.0, 1.0];
    e.project(&mut x);
    let c = PowerCone::new(0.5);
    let mut y = [f64::INFINITY, 1.0, 1.0];
    c.project(&mut y);
}
