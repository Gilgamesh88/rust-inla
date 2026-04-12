use inla_core::models::{IidModel, QFunc};
use inla_core::likelihood::{LogLikelihood, PoissonLikelihood};
use inla_core::problem::Problem;
use inla_core::inference::InlaModel;
use faer::sparse::SparseColMat;
use ndarray::Array1;

#[test]
fn test_synthetic_fremtpl2_poisson() {
    let n_data = 5;
    let n_fixed = 1;

    // Claims data
    let y = vec![0.0, 1.0, 0.0, 2.0, 0.0];
    
    // Fixed Matrix (Intercept)
    let fixed_matrix = vec![
        1.0, 
        1.0, 
        1.0, 
        1.0, 
        1.0
    ];

    // Offsets / exposures (log exposure)
    let offset = vec![0.0; n_data];

    // Single IID model (e.g. VehBrand) with 2 levels
    let mut iid = IidModel::new(2);
    let likelihood = PoissonLikelihood {};

    // Synthetic sparse mapping (COO tuple) mapping observations to Random Effects
    // y[0] -> Brand 0
    // y[1] -> Brand 1
    // y[2] -> Brand 0
    // y[3] -> Brand 1
    // y[4] -> Brand 0
    let a_i = vec![0, 1, 2, 3, 4];
    let a_j = vec![0, 1, 0, 1, 0];
    let a_x = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    // Initial hyperparameter params: theta = [log(tau)]
    let theta = vec![0.0]; 

    let model = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &iid,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed,
        fixed_matrix: Some(&fixed_matrix),
        n_latent: 2,
        theta_init: theta.clone(), // Vec<f64>
        extr_constr: None,
        n_constr: 0,
    };

    let mut prob = Problem::new(&model);

    // Call finding mode:
    let (beta, x, log_det_aug, _, schur_log_det) = prob.find_mode_with_fixed_effects(
        &model,
        &theta,
        &[],
        &[],
        50,
        1e-5
    ).unwrap();

    println!("Intercept Mode: {:?}", beta);
    println!("Random Effects Mode: {:?}", x);
    println!("Schur Log Det: {}", schur_log_det);
    println!("Augmented Log Det: {}", log_det_aug);

    assert!(beta.len() == 1);
    assert!(x.len() == 2);
}
