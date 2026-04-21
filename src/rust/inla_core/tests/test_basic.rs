use inla_core::inference::{InlaEngine, InlaModel, InlaParams};
use inla_core::likelihood::{
    GammaLikelihood, GaussianLikelihood, LogLikelihood, PoissonLikelihood,
};
use inla_core::models::{Ar1Model, CompoundQFunc, IidModel, QFunc, Rw1Model, Rw2Model};
use inla_core::problem::Problem;

fn dense_cholesky_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                assert!(sum > 1e-12, "dense reference matrix must remain SPD");
                a[i * n + j] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }

    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * n + k] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= a[k * n + i] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }

    b
}

fn dense_spd_inverse_diagonal(a: &[f64], n: usize) -> Vec<f64> {
    let mut diag = vec![0.0_f64; n];
    for col in 0..n {
        let mut rhs = vec![0.0_f64; n];
        rhs[col] = 1.0;
        let sol = dense_cholesky_solve(a.to_vec(), rhs, n);
        diag[col] = sol[col];
    }
    diag
}

fn dense_q_matrix(qfunc: &dyn QFunc, theta_model: &[f64], n: usize) -> Vec<f64> {
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        q[i * n + i] = qfunc.eval(i, i, theta_model);
        for j in (i + 1)..n {
            if qfunc.graph().are_neighbors(i, j) {
                let val = qfunc.eval(i, j, theta_model);
                q[i * n + j] = val;
                q[j * n + i] = val;
            }
        }
    }
    q
}

#[test]
fn test_synthetic_fremtpl2_poisson() {
    let n_data = 5;
    let n_fixed = 1;

    // Claims data
    let y = vec![0.0, 1.0, 0.0, 2.0, 0.0];

    // Fixed Matrix (Intercept)
    let fixed_matrix = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    // Offsets / exposures (log exposure)
    let offset = vec![0.0; n_data];

    // Single IID model (e.g. VehBrand) with 2 levels
    let iid = IidModel::new(2);
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
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: None,
        n_constr: 0,
    };

    let mut prob = Problem::new(&model);

    // Call finding mode:
    let (beta, x, log_det_aug, _, schur_log_det) = prob
        .find_mode_with_fixed_effects(&model, &theta, &[], &[], 50, 1e-5)
        .unwrap();

    println!("Intercept Mode: {:?}", beta);
    println!("Random Effects Mode: {:?}", x);
    println!("Schur Log Det: {}", schur_log_det);
    println!("Augmented Log Det: {}", log_det_aug);

    assert!(beta.len() == 1);
    assert!(x.len() == 2);
}

#[test]
fn test_compound_latent_blocks_with_offset_run_end_to_end() {
    let y = vec![0.1, -0.2, 1.0, 0.8, 0.5];
    let offset = vec![0.2, -0.1, 0.05, 0.0, -0.15];

    let qfunc = CompoundQFunc::new(vec![
        (0, Box::new(IidModel::new(2))),
        (2, Box::new(Ar1Model::new(3))),
    ]);
    let likelihood = GaussianLikelihood {};

    let a_i = vec![0, 1, 2, 3, 4];
    let a_j = vec![0, 1, 2, 3, 4];
    let a_x = vec![1.0; 5];

    let model = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 0,
        fixed_matrix: None,
        n_latent: 5,
        theta_init: vec![0.0; qfunc.n_hyperparams() + likelihood.n_hyperparams()],
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: None,
        n_constr: 0,
    };

    let res = InlaEngine::run(&model, &InlaParams::default()).unwrap();

    assert_eq!(res.theta_opt.len(), 4);
    assert_eq!(res.random.len(), 5);
    assert_eq!(res.fitted.len(), 5);
    assert_eq!(res.posterior_mean.len(), 5);
    assert_eq!(res.w_opt.len(), 5);
    assert!(res.fitted.iter().all(|m| m.mean().is_finite()));
    assert!(res.w_opt.iter().all(|w| *w > 0.0));
}

#[test]
fn test_gamma_iid_run_end_to_end() {
    let y = vec![1.4, 0.9, 2.1, 1.2, 1.8];
    let offset = vec![0.0; y.len()];
    let fixed_matrix = vec![1.0; y.len()];

    let qfunc = IidModel::new(2);
    let likelihood = GammaLikelihood {};

    let a_i = vec![0, 1, 2, 3, 4];
    let a_j = vec![0, 1, 0, 1, 0];
    let a_x = vec![1.0; y.len()];

    let model = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 1,
        fixed_matrix: Some(&fixed_matrix),
        n_latent: 2,
        theta_init: vec![0.0; qfunc.n_hyperparams() + likelihood.n_hyperparams()],
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: None,
        n_constr: 0,
    };

    let res = InlaEngine::run(&model, &InlaParams::default()).unwrap();

    assert_eq!(res.theta_opt.len(), 2);
    assert_eq!(res.fixed_means.len(), 1);
    assert_eq!(res.random.len(), 2);
    assert_eq!(res.fitted.len(), y.len());
    assert_eq!(res.posterior_mean.len(), 2);
    assert_eq!(res.w_opt.len(), 2);
    assert!(res.fixed_means[0].is_finite());
    assert!(res.fixed_sds[0].is_finite());
    assert!(res
        .fitted
        .iter()
        .all(|m| m.mean().is_finite() && m.mean() > 0.0));
    assert!(res.w_opt.iter().all(|w| *w > 0.0));
}

#[test]
fn test_rw1_constraint_changes_latent_covariance_diagonal() {
    let y = vec![1.5, 1.5, 1.5, 1.5];
    let offset = vec![0.0; y.len()];
    let fixed_matrix = vec![1.0; y.len()];
    let a_i = vec![0, 1, 2, 3];
    let a_j = vec![0, 1, 2, 3];
    let a_x = vec![1.0; y.len()];
    let theta = vec![0.0, 0.0];
    let likelihood = GaussianLikelihood {};
    let qfunc = Rw1Model::new(4);
    let extr_constr = vec![1.0; 4];

    let model_constrained = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 1,
        fixed_matrix: Some(&fixed_matrix),
        n_latent: 4,
        theta_init: theta.clone(),
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: Some(&extr_constr),
        n_constr: 1,
    };
    let model_unconstrained = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 1,
        fixed_matrix: Some(&fixed_matrix),
        n_latent: 4,
        theta_init: theta.clone(),
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: None,
        n_constr: 0,
    };

    let mut problem_constrained = Problem::new(&model_constrained);
    let (_, x_constrained, _, diag_constrained, _) = problem_constrained
        .find_mode_with_fixed_effects(&model_constrained, &theta, &[], &[], 50, 1e-8)
        .unwrap();
    let mut problem_unconstrained = Problem::new(&model_unconstrained);
    let (_, _, _, diag_unconstrained, _) = problem_unconstrained
        .find_mode_with_fixed_effects(&model_unconstrained, &theta, &[], &[], 50, 1e-8)
        .unwrap();

    let avg_diag_constrained = diag_constrained.iter().sum::<f64>() / diag_constrained.len() as f64;
    let avg_diag_unconstrained =
        diag_unconstrained.iter().sum::<f64>() / diag_unconstrained.len() as f64;

    assert!(x_constrained.iter().sum::<f64>().abs() < 1e-8);
    assert!(avg_diag_constrained < avg_diag_unconstrained * 0.5);
}

#[test]
fn test_rw2_gaussian_with_intrinsic_constraints_runs_end_to_end() {
    let y = vec![0.1, 0.4, 0.9, 1.0, 0.6, 0.2];
    let fixed_matrix = vec![1.0; y.len()];
    let offset = vec![0.0; y.len()];
    let a_i = vec![0, 1, 2, 3, 4, 5];
    let a_j = vec![0, 1, 2, 3, 4, 5];
    let a_x = vec![1.0; y.len()];
    let structure_values = [390.0_f64, 391.0, 393.0, 394.0, 396.0, 397.0];
    let structure_mean = structure_values.iter().sum::<f64>() / structure_values.len() as f64;
    let centered_idx: Vec<f64> = structure_values
        .iter()
        .map(|value| value - structure_mean)
        .collect();
    let mut extr_constr = vec![1.0; y.len()];
    extr_constr.extend(centered_idx.iter().copied());

    let qfunc = Rw2Model::new_with_values(&structure_values).unwrap();
    let likelihood = GaussianLikelihood {};

    let model = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 1,
        fixed_matrix: Some(&fixed_matrix),
        n_latent: 6,
        theta_init: vec![0.0; qfunc.n_hyperparams() + likelihood.n_hyperparams()],
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: Some(&extr_constr),
        n_constr: 2,
    };

    let res = InlaEngine::run(&model, &InlaParams::default()).unwrap();
    let random_mean_sum = res.posterior_mean.iter().sum::<f64>();
    let weighted_sum = res
        .posterior_mean
        .iter()
        .zip(centered_idx.iter())
        .map(|(value, weight)| value * weight)
        .sum::<f64>();

    assert_eq!(res.theta_opt.len(), 2);
    assert_eq!(res.random.len(), 6);
    assert_eq!(res.fitted.len(), y.len());
    assert!(res.fixed_means[0].is_finite());
    assert!(res.random.iter().all(|m| m.mean().is_finite()));
    assert!(res.random.iter().all(|m| m.variance().is_finite()));
    assert!(random_mean_sum.abs() < 1e-6);
    assert!(weighted_sum.abs() < 1e-6);
}

#[test]
fn test_latent_only_rw2_gaussian_mode_matches_dense_posterior() {
    let y = vec![0.15, -0.05, 0.30, 0.80, 0.40, -0.10];
    let offset = vec![0.0; y.len()];
    let a_i = vec![0, 1, 2, 3, 4, 5];
    let a_j = vec![0, 1, 2, 3, 4, 5];
    let a_x = vec![1.0; y.len()];
    let theta = vec![6.0, 3.5];

    let qfunc = Rw2Model::new(y.len());
    let likelihood = GaussianLikelihood {};
    let model = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 0,
        fixed_matrix: None,
        n_latent: y.len(),
        theta_init: theta.clone(),
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: None,
        n_constr: 0,
    };

    let mut problem = Problem::new(&model);
    let (mode_x, _, diag_aug_inv) = problem
        .find_mode_with_inverse(&model, &theta, &[], 50, 1e-10)
        .unwrap();

    let tau_obs = theta[1].exp();
    let mut q_aug = dense_q_matrix(&qfunc, &theta[..1], y.len());
    for i in 0..y.len() {
        q_aug[i * y.len() + i] += tau_obs;
    }

    let rhs: Vec<f64> = y.iter().map(|yi| tau_obs * yi).collect();
    let exact_mean = dense_cholesky_solve(q_aug.clone(), rhs, y.len());
    let exact_diag = dense_spd_inverse_diagonal(&q_aug, y.len());

    for i in 0..y.len() {
        assert!(
            (mode_x[i] - exact_mean[i]).abs() < 1e-8,
            "latent mode mismatch at {i}: got {}, expected {}",
            mode_x[i],
            exact_mean[i]
        );
        assert!(
            (diag_aug_inv[i] - exact_diag[i]).abs() < 1e-8,
            "latent variance mismatch at {i}: got {}, expected {}",
            diag_aug_inv[i],
            exact_diag[i]
        );
    }
}

#[test]
fn test_latent_only_irregular_rw2_gaussian_mode_matches_dense_posterior() {
    let y = vec![0.15, -0.05, 0.30, 0.80, 0.40, -0.10];
    let offset = vec![0.0; y.len()];
    let a_i = vec![0, 1, 2, 3, 4, 5];
    let a_j = vec![0, 1, 2, 3, 4, 5];
    let a_x = vec![1.0; y.len()];
    let theta = vec![6.0, 3.5];
    let structure_values = vec![390.0, 391.0, 393.0, 394.0, 396.0, 397.0];

    let qfunc = Rw2Model::new_with_values(&structure_values).unwrap();
    let likelihood = GaussianLikelihood {};
    let model = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 0,
        fixed_matrix: None,
        n_latent: y.len(),
        theta_init: theta.clone(),
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: None,
        n_constr: 0,
    };

    let mut problem = Problem::new(&model);
    let (mode_x, _, diag_aug_inv) = problem
        .find_mode_with_inverse(&model, &theta, &[], 50, 1e-10)
        .unwrap();

    let tau_obs = theta[1].exp();
    let mut q_aug = dense_q_matrix(&qfunc, &theta[..1], y.len());
    for i in 0..y.len() {
        q_aug[i * y.len() + i] += tau_obs;
    }

    let rhs: Vec<f64> = y.iter().map(|yi| tau_obs * yi).collect();
    let exact_mean = dense_cholesky_solve(q_aug.clone(), rhs, y.len());
    let exact_diag = dense_spd_inverse_diagonal(&q_aug, y.len());

    for i in 0..y.len() {
        assert!(
            (mode_x[i] - exact_mean[i]).abs() < 1e-8,
            "irregular latent mode mismatch at {i}: got {}, expected {}",
            mode_x[i],
            exact_mean[i]
        );
        assert!(
            (diag_aug_inv[i] - exact_diag[i]).abs() < 1e-8,
            "irregular latent variance mismatch at {i}: got {}, expected {}",
            diag_aug_inv[i],
            exact_diag[i]
        );
    }
}

#[test]
fn test_ccd_exports_keep_log_mlik_separate_from_log_weight() {
    let y = vec![0.1, -0.2, 1.0, 0.8, 0.5];
    let offset = vec![0.2, -0.1, 0.05, 0.0, -0.15];

    let qfunc = CompoundQFunc::new(vec![
        (0, Box::new(IidModel::new(2))),
        (2, Box::new(Ar1Model::new(3))),
    ]);
    let likelihood = GaussianLikelihood {};

    let a_i = vec![0, 1, 2, 3, 4];
    let a_j = vec![0, 1, 2, 3, 4];
    let a_x = vec![1.0; 5];

    let model = InlaModel {
        y: &y,
        offset: Some(&offset),
        qfunc: &qfunc,
        likelihood: &likelihood,
        a_i: Some(&a_i),
        a_j: Some(&a_j),
        a_x: Some(&a_x),
        n_fixed: 0,
        fixed_matrix: None,
        n_latent: 5,
        theta_init: vec![0.0; qfunc.n_hyperparams() + likelihood.n_hyperparams()],
        latent_init: vec![],
        fixed_init: vec![],
        extr_constr: None,
        n_constr: 0,
    };

    let res = InlaEngine::run(&model, &InlaParams::default()).unwrap();

    let n_points = res.ccd_base_weights.len();
    assert!(n_points > 1);
    assert_eq!(res.ccd_log_mlik.len(), n_points);
    assert_eq!(res.ccd_log_weight.len(), n_points);
    assert_eq!(res.ccd_weights.len(), n_points);
    assert!(res
        .ccd_base_weights
        .iter()
        .skip(1)
        .any(|w| (*w - 1.0).abs() > 1e-12));

    let max_log_weight = res
        .ccd_log_weight
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let unnormalized: Vec<f64> = res
        .ccd_log_weight
        .iter()
        .map(|log_weight| (*log_weight - max_log_weight).exp())
        .collect();
    let weight_sum: f64 = unnormalized.iter().sum();

    for i in 0..n_points {
        let expected_gap = res.ccd_base_weights[i].max(1e-300).ln();
        let observed_gap = res.ccd_log_weight[i] - res.ccd_log_mlik[i];
        assert!(
            (observed_gap - expected_gap).abs() < 1e-10,
            "ccd_log_weight should equal log(base_weight) + ccd_log_mlik at point {i}"
        );

        let expected_weight = unnormalized[i] / weight_sum;
        assert!(
            (res.ccd_weights[i] - expected_weight).abs() < 1e-10,
            "ccd_weights should softmax ccd_log_weight at point {i}"
        );
    }
}
