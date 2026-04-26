import os
import re

path = 'C:/Users/Antonio/.gemini/antigravity/scratch/rustyINLA/src/rust/inla_core/src/problem/mod.rs'

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update AugmentedQFunc to add 1e-6 to intrinsic diagonal
aug_q_old = '''    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let base = self.inner.eval(i, j, theta);
        let (min_j, max_j) = if i < j { (i, j) } else { (j, i) };
        base + self.atwa.get(&(min_j, max_j)).copied().unwrap_or(0.0)
    }'''
    
aug_q_new = '''    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let mut base = self.inner.eval(i, j, theta);
        if i == j && !self.inner.is_proper() {
            base += 1e-6; // Intrinsic regularization
        }
        let (min_j, max_j) = if i < j { (i, j) } else { (j, i) };
        base + self.atwa.get(&(min_j, max_j)).copied().unwrap_or(0.0)
    }'''

text = text.replace(aug_q_old, aug_q_new)

# 2. Inject KKT Kriging solver function at the bottom
kkt_fn = '''
// Kriging solver: u_new = u_old - V_c (C V_c)^-1 C u_old
fn apply_kriging_correction(
    u: &mut [f64],
    v_c: &[f64], // n_latent x n_constr
    c_vc_inv: &[f64], // n_constr x n_constr
    constr: &[f64], // n_constr x n_latent
    n_latent: usize,
    n_constr: usize,
) {
    let mut cu = vec![0.0_f64; n_constr];
    for c in 0..n_constr {
        for i in 0..n_latent {
            cu[c] += constr[c * n_latent + i] * u[i];
        }
    }
    let mut lambda = vec![0.0_f64; n_constr];
    for c1 in 0..n_constr {
        for c2 in 0..n_constr {
            lambda[c1] += c_vc_inv[c1 * n_constr + c2] * cu[c2];
        }
    }
    for i in 0..n_latent {
        for c in 0..n_constr {
            u[i] -= v_c[c * n_latent + i] * lambda[c];
        }
    }
}
'''
if "apply_kriging_correction" not in text:
    text += kkt_fn

# 3. Apply inside find_mode_with_fixed_effects
find_mode_logic = '''            let mut u = b_x.clone();
            self.solver.solve_llt(&mut u);

            let mut v = vec![0.0_f64; n_latent * n_fixed];
            for j in 0..n_fixed {
                let mut v_col = w_cross[j * n_latent .. (j+1) * n_latent].to_vec();
                self.solver.solve_llt(&mut v_col);
                for k in 0..n_latent { v[k + j * n_latent] = v_col[k]; }
            }'''

kriging_logic = '''            let mut u = b_x.clone();
            self.solver.solve_llt(&mut u);

            let mut v = vec![0.0_f64; n_latent * n_fixed];
            for j in 0..n_fixed {
                let mut v_col = w_cross[j * n_latent .. (j+1) * n_latent].to_vec();
                self.solver.solve_llt(&mut v_col);
                for k in 0..n_latent { v[k + j * n_latent] = v_col[k]; }
            }
            
            // KKT Kriging Core
            if let Some(constr) = model.extr_constr {
                let n_c = model.n_constr;
                if n_c > 0 {
                    let mut v_c = vec![0.0_f64; n_latent * n_c];
                    for c in 0..n_c {
                        let mut vc_col = constr[c * n_latent .. (c+1) * n_latent].to_vec();
                        self.solver.solve_llt(&mut vc_col);
                        for k in 0..n_latent { v_c[c * n_latent + k] = vc_col[k]; }
                    }
                    
                    let mut c_vc = vec![0.0_f64; n_c * n_c];
                    for c1 in 0..n_c {
                        for c2 in 0..n_c {
                            for k in 0..n_latent {
                                c_vc[c1 * n_c + c2] += constr[c1 * n_latent + k] * v_c[c2 * n_latent + k];
                            }
                        }
                    }
                    
                    let mut c_vc_inv = c_vc.clone();
                    let mut dummy_b = vec![0.0_f64; n_c]; // dense_cholesky modifies b
                    if dense_cholesky_solve(&mut c_vc_inv, &mut dummy_b, n_c).is_ok() {
                        // We must compute matrix inverse from the packed cholesky representation.
                        // Actually dense_cholesky_solve returns L in lower triangle.
                        // For simplicity, let's just do a poor-man's dense inverse by solving identity vectors.
                        let mut real_inv = vec![0.0_f64; n_c * n_c];
                        for c in 0..n_c {
                            let mut e = vec![0.0_f64; n_c];
                            e[c] = 1.0;
                            let mut cvc_tmp = c_vc.clone();
                            let _ = dense_cholesky_solve(&mut cvc_tmp, &mut e, n_c);
                            for c2 in 0..n_c { real_inv[c * n_c + c2] = e[c2]; }
                        }
                        
                        apply_kriging_correction(&mut u, &v_c, &real_inv, constr, n_latent, n_c);
                        
                        for j in 0..n_fixed {
                            let mut vj = v[j * n_latent .. (j+1)*n_latent].to_vec();
                            apply_kriging_correction(&mut vj, &v_c, &real_inv, constr, n_latent, n_c);
                            for k in 0..n_latent { v[k + j * n_latent] = vj[k]; }
                        }
                    }
                }
            }'''

text = text.replace(find_mode_logic, kriging_logic)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('KKT constraints injected!')
