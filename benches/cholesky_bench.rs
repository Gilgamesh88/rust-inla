// Benchmark de rendimiento del solver de Cholesky.
// Implementación completa en Fase A.5.
//
// Este archivo existe para satisfacer la declaración [[bench]] en Cargo.toml.
// Criterion requiere un punto de entrada aunque el benchmark esté vacío.

use criterion::criterion_main;

// En Fase A.5 se añadirá:
// criterion_group!(benches, bench_cholesky_10k);
// criterion_main!(benches);

// Por ahora: punto de entrada vacío para que cargo build no falle.
criterion_main!();
