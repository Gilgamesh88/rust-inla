use faer::prelude::*;
use faer::Mat;

#[test]
fn test_faer_svd() {
    let mut a: Mat<f64> = Mat::zeros(2, 2);
    a.write(0, 0, 2.0);
    a.write(1, 1, 3.0);
    let svd = a.svd();
    let s = svd.s();
    let u = svd.u();
}
