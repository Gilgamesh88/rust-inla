//! Benchmark de rendimiento del solver de Cholesky.
//!
//! Mide el tiempo de factorización de Cholesky sparse en distintos tamaños
//! de matrices tridiagonales (RW1). Ejecutar con:
//!
//!   cargo bench
//!
//! Reporta ns/iter para cada tamaño.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_inla::graph::Graph;
use rust_inla::models::Rw1Model;
use rust_inla::solver::{FaerSolver, SparseSolver};

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_rw1");

    for &n in &[100usize, 500, 1_000, 5_000] {
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, &n| {
            // Setup fuera del benchmark: reorder (AMD) se hace una sola vez
            let model = Rw1Model::new(n);
            let mut graph = Graph::linear(n);
            let mut solver = FaerSolver::new();
            solver.reorder(&mut graph);

            // Solo medimos build + factorize (lo que BFGS paga en cada eval)
            let theta = [1.0f64]; // log-tau = 1
            b.iter(|| {
                solver.build(&graph, &model, &theta);
                solver.factorize().unwrap();
                solver.log_determinant()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cholesky);
criterion_main!(benches);
